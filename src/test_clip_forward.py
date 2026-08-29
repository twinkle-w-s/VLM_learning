import os

import open_clip
import torch
from torch.utils.data import DataLoader

from flickr30k_datasets import Flickr30KDataset

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    model_name="ViT-B-32"

    storage_root = f"/data/{os.environ['USER']}/vlm_learning"
    model_cache = f"{storage_root}/models/open_clip"

    model, _, preprocess=open_clip.create_model_and_transforms(
        model_name,
        pretrained="openai",
        cache_dir=model_cache,
        device=device
    )
    tokenizer=open_clip.get_tokenizer(model_name)#把caption的text映射成文本向量库中数值的方法
    dataset=Flickr30KDataset(
        split_init="train",
        preprocess=preprocess,
        tokenizer=tokenizer,
    )

    dataloader=DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,#这里只做前向传播的验证，不打乱
        num_workers=2,
    )

    batch=next(iter(dataloader))#把DataLoader变成迭代器，然后取出第一个

    print("dataset size:", len(dataset))
    print("pixel_values shape:", batch["pixel_values"].shape)
    print("input_ids shape:", batch["input_ids"].shape)

    print("captions:")
    for caption in batch["caption"]:
        print("-", caption)

    pixel_values=batch["pixel_values"].to(device)
    input_ids=batch["input_ids"].to(device)

    print("model device:", next(model.parameters()).device)
    print("image device:", pixel_values.device)
    print("text device:", input_ids.device)
    #优化器
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
    )

    model.train()

    optimizer.zero_grad()#清空上一次的梯度

    

    with torch.enable_grad():
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
        print("logits形状：",logits_per_image.shape)

        batch_size=pixel_values.shape[0]

        labels=torch.arange(
            batch_size,
            device=device,
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


        loss.backward()
        optimizer.step()

        first_parameter=next(model.parameters())#取出模型的第一个参数

        print("first parameter requires_grad:", first_parameter.requires_grad)
        print("first parameter has gradient:", first_parameter.grad is not None)

        if first_parameter.grad is not None:
            print("first parameter gradient norm:", first_parameter.grad.norm().item())



    #重新计算一遍特征，确保使用的是参数更新后
    model.eval()

    with torch.inference_mode():
        image_features_after = model.encode_image(pixel_values)
        text_features_after = model.encode_text(input_ids)

        image_features_after = image_features_after / image_features_after.norm(
            dim=-1,
            keepdim=True,
        )
        text_features_after = text_features_after / text_features_after.norm(
            dim=-1,
            keepdim=True,
        )

        logits_after = (
            model.logit_scale.exp()
            * image_features_after
            @ text_features_after.T
        )

        loss_image_after = torch.nn.functional.cross_entropy(
            logits_after,
            labels,
        )
        loss_text_after = torch.nn.functional.cross_entropy(
            logits_after.T,
            labels,
        )
        loss_after = (loss_image_after + loss_text_after) / 2









    print("image feature shape:", image_features.shape)
    print("text feature shape:", text_features.shape)
    print("similarity matrix shape:", logits_per_image.shape)
    print("similarity matrix:")
    print(logits_per_image)

    print("diagonal scores:")
    print(logits_per_image.diag())


    predicted_text = logits_per_image.argmax(dim=1)

    print("loss_image:", loss_image.item())
    print("loss_text:", loss_text.item())

    print("loss before update:", loss.item())
    print("loss after update:", loss_after.item())
    print("loss change:", loss_after.item() - loss.item())
    
    print("predicted text index:", predicted_text)
    print("correct text index:", labels)


if __name__ == "__main__":
    main()