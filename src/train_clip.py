import os
import random
import numpy as np
import open_clip
import torch
import yaml
from torch.utils.data import DataLoader

from flickr30k_datasets import Flickr30KDataset

#固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)#控制在cpu上的随机操作
    torch.cuda.manual_seed_all(seed)#控制在GPU上的随机操作

def load_config(config_path):
    with open(config_path,"r",encoding="utf-8") as file:
        config=yaml.safe_load(file)
    return config

def compute_clip_loss(model,pixel_values,input_ids):
    image_features=model.encode_image(pixel_values)
    text_features=model.encode_text(input_ids)

    image_features=image_features/image_features.norm(
        dim=-1,
        keepdim=True,
    )
    text_features=text_features/text_features.norm(
        dim=-1,
        keepdim=True,
    )#对两个特征都做L2归一化

    logits_per_image=(
        model.logit_scale.exp()*image_features@text_features.T

    )#model.logit._scale.exp()是放大系数，把相似度放大到更适合交叉熵损失的范围
    
    batch_size=pixel_values.shape[0]

    labels=torch.arange(
        batch_size,
        device=pixel_values.device,
    )#arange表示生成一个数值范围连续的张量

    #计算双向对比损失
    loss_image=torch.nn.functional.cross_entropy(
        logits_per_image,
        labels
    )

    loss_text=torch.nn.functional.cross_entropy(
        logits_per_image.T,
        labels
    )
    loss=(loss_image+loss_text)/2

    return loss,logits_per_image


def main():
    config_path="configs/flickr30k-debug.yaml"
    config=load_config(config_path)

    set_seed(config["seed"])

    device="cuda" if torch.cuda.is_available() else "cpu"
    

    storage_root = f"/data/{os.environ['USER']}/vlm_learning"
    model_cache = f"{storage_root}/models/open_clip"

    model, _, preprocess=open_clip.create_model_and_transforms(
        config["model_name"],
        pretrained=config["pretrained"],
        cache_dir=model_cache,
        device=device
    )

    tokenizer=open_clip.get_tokenizer(config["model_name"])#把caption的text映射成文本向量库中数值的方法
    train_dataset=Flickr30KDataset(
        split_init="train",
        preprocess=preprocess,
        tokenizer=tokenizer,
    )

    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,#现在要训练了，需要打乱
        num_workers=config["num_workers"],
        pin_memory=True,#让 CPU 到 GPU 的数据传输更高效
    )

   
    

    
    #优化器
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01
    )

    model.train()

    num_epochs=config["epochs"]

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        print(f"starting epoch {epoch + 1}/{num_epochs}")


        for step, batch in train_loader:
            pixel_values=batch["pixel_values"].to(
                device,
                non_blocking=True,#使用固定内存时，让cpu到gpu更快
            )

            input_ids=batch["input_ids"].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()#清空上一次的梯度

            loss, logits_per_image=compute_clip_loss(
                model,
                pixel_values,
                input_ids
            )

            loss.backward()
            optimizer.step()

            batch_size=pixel_values.shape[0]

            predictions=logits_per_image.argmax(dim=1)
            labels=torch.arange(
                batch_size,
                device=device
            )
        
            total_correct += (predictions == labels).sum().item()#计算Recall@1
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            #打印batch信息
            if (step + 1) % 100 == 0:
                current_loss = total_loss / total_samples
                current_recall = total_correct / total_samples

                print(
                    f"epoch={epoch + 1}, "
                    f"step={step + 1}/{len(train_loader)}, "
                    f"loss={current_loss:.4f}, "
                    f"recall@1={current_recall:.4f}"
                )
                epoch_loss = total_loss / total_samples

        #打印epoch信息        
        epoch_recall = total_correct / total_samples

        print(
            f"epoch {epoch + 1} finished: "
            f"loss={epoch_loss:.4f}, "
            f"recall@1={epoch_recall:.4f}"
        )
    


if __name__ == "__main__":
    main()