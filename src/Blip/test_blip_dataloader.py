import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration

from blip_datasets import Flickr30KBLIPDataset

# def get_storage_root():
#     return (
#         Path("/data")
#         / os.environ["USER"]
#         / "vlm_learning"
#     )

def main():
    storage_root = Path("/data") / os.environ["USER"] / "vlm_learning"

    model_path = (
        storage_root
        / "models"
        / "blip-image-captioning-base"
    )# f"{storage_root}/models/open_clip"

    dataset_cache = (
        storage_root
        / "datasets"
        / "flickr30k"
    )
    processor=BlipProcessor.from_pretrained(
        model_path,
        cache_dir=str(dataset_cache),#如果本地没有，还是回从网络下载到这个目录
        trust_remote_code=True,
    )   
    
    model = BlipForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model = model.to(device)
    model.train()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.train()

    dataset=Flickr30KBLIPDataset(
        split="val",
        processor=processor,
        dataset_cache=dataset_cache,
        
    )       
    print("dastaset size:", len(dataset)) 

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False,num_workers=0)

    batch = next(iter(dataloader))#取出一个

    print("batch keys:", batch.keys())
    print("pixel_values shape:", batch["pixel_values"].shape)
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("captions:")

    for caption in batch["caption"]:
        print("-", caption)

    print("input ids of first sample:")
    print(batch["input_ids"][0])

    print("attention mask of first sample:")
    print(batch["attention_mask"][0])

    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    labels=input_ids.clone()# clone()是为了避免修改原来的张量
    labels[attention_mask==0]=-100#把padding位置的标签设置为-100，表示不计算损失


    outputs=model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss=outputs.loss

    print("loss:", loss.item())
    print("logits shape:", outputs.logits.shape)

    loss.backward()#计算梯度

    first_param = next(model.parameters())
    print("first param requires grad:", first_param.requires_grad)

    print("first param has grad:", first_param.grad is not None)


if __name__ == "__main__":
    main()