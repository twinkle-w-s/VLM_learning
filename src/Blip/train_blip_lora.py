import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
)

src_dir=Path(__file__).resolve().parents[1]#parents[0]是当前目录


sys.path.insert(
    0,
    str(src_dir),
)#把父目录加入当前的sys.path中，这样就可以导入父目录中的模块了

from blip_datasets import Flickr30KBLIPDataset
from lora_layers import (
    LoRALinear,
    add_lora_to_linear_layers,
    count_trainable_parameters,
    get_lora_state_dict,
)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def move_batch_to_device(batch, device):
    return {
        name: value.to(device)
        for name, value in batch.items()
        if torch.is_tensor(value)
    }

def compute_blip_loss(model, batch):
    input_ids = batch["input_ids"]
    pixel_values = batch["pixel_values"]
    attention_mask = batch["attention_mask"]

    labels = input_ids.clone()
    labels[batch["attention_mask"] == 0] = -100

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        labels=labels,
        attention_mask=attention_mask,
    )#输入四个关键词，来自batch中，返回一个包含loss的outputs对象
    #不同于CLIP模型，BLIP模型的forward函数直接返回loss，而不是logits，logits还要自己手动实现交叉熵，这里不用

    return outputs.loss


def evaluate_loss(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            loss = compute_blip_loss(model, batch)
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    model.train()

    average_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return average_loss


def main():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "blip-debug.yaml"
    config = load_config(config_path)
    set_seed(config["seed"])

    storage_root = Path("/data") / os.environ["USER"] / "vlm_learning"

    model_path = (
        storage_root
        / config["model_dir"]
    )

    dataset_cache = (
        storage_root
        / config["dataset_cache"]
    )

    output_dir = (
        storage_root
        / config["output_dir"]
    )

    output_dir.mkdir(
        parents=True,#如果父目录不存在，就创建父目录
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)
    print("model path:", model_path)
    print("dataset cache:", dataset_cache)
    print("output dir:", output_dir)


    ################################加载模型和处理器################################
    model = BlipForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
    ).to(device)

    processor = BlipProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model.train()

    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_config = config["lora"]

    target_names = lora_config["target_names"]
    rank = lora_config["rank"]
    alpha = lora_config["alpha"]
    dropout = lora_config["dropout"]

    print("LoRA target names:", target_names)
    print("LoRA rank:", rank)
    print("LoRA alpha:", alpha)

    add_lora_to_linear_layers(
        module=model,  
        target_names=target_names,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )
    model = model.to(device)  # 再装载一次模型到设备上

    replaced_layers=sum(
        isinstance(module, LoRALinear)#如果module是LoRALinear的实例，就返回True，否则返回False，然后通过sum()函数统计True的数量
        for module in model.modules()#遍历所有子模块
    )

    if replaced_layers == 0:
        raise RuntimeError(
            "No linear layers were replaced with LoRA layers. "
            "Please check the target_names in the configuration."
        )
    trainable, total = count_trainable_parameters(model)
    # print(f"Trainable parameters: {trainable:,}, Total parameters: {total:,}")#按照千位分隔符格式化输出
    # print("replaced layers:", replaced_layers)


    # print("trainable parameter names:")
    # for name, parameter in model.named_parameters():
    #     if parameter.requires_grad:
    #         print(name, tuple(parameter.shape))#tuple(parameter.shape)返回参数的形状，例如torch.Size([512, 512])，然后转换为元组(512, 512)
    ######################导入数据集####################################
    train_dataset = Flickr30KBLIPDataset(
        split=config["train_split"],
        processor=processor,
        dataset_cache=dataset_cache,
    )

    validation_dataset = Flickr30KBLIPDataset(
        split=config["validation_split"],
        processor=processor,
        dataset_cache=dataset_cache,
    )
    ############创建数据加载器#######################

    train_loader=DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    validation_loader=DataLoader(
        validation_dataset, 
        batch_size=config["batch_size"],
        shuffle=False,  
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    ###定义优化器###
    optimizer = torch.optim.AdamW(
        params=trainable_parameters,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    num_epochs = config["epochs"]

    print("number of epochs:", num_epochs)
    print("number of training batches:", len(train_loader))
    print(
        "number of validation batches:",
        len(validation_loader),
    )
    ####################训练循环########################
    best_validation_loss = float("inf")
    training_history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            
            loss = compute_blip_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()#计算梯度
            optimizer.step()#更新参数

            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if (step + 1) % 100 == 0:
                current_loss = (
                    total_loss
                    / total_samples
                )

                print(
                    f"epoch={epoch + 1}, "
                    f"step={step + 1}/"
                    f"{len(train_loader)}, "
                    f"loss={current_loss:.4f}"
                )

        average_train_loss = (
            total_loss / total_samples if total_samples > 0 else 0.0
        )

        average_validation_loss = evaluate_loss(
            model, validation_loader, device
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {average_train_loss:.4f}, "
            f"Validation Loss: {average_validation_loss:.4f}"
        )

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": average_train_loss,
                "validation_loss": average_validation_loss,
            }
        )

        checkpoint={
            "epoch": epoch + 1,
            "lora_state_dict": get_lora_state_dict(model),
            "config": config,
            
            "train_loss": average_train_loss,
            "validation_loss": average_validation_loss,
            "training_history": training_history,
        }

        checkpoint_path = (
            output_dir / f"epoch_{epoch + 1:02d}_lora.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            best_checkpoint_path = (
                output_dir / "best_model_lora.pt"
            )
            torch.save(checkpoint, best_checkpoint_path)
            print(
                f"New best model saved to {best_checkpoint_path}"
            )
    history_path = output_dir / "training_history.yaml"
    with open(history_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(training_history, f,allow_unicode=True)#这里会把训练历史保存为yaml文件，allow_unicode=True允许保存中文字符

    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()