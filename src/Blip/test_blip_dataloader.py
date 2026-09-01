import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor

from blip_datasets import Flickr30KBLIPDataset

# def get_storage_root():
#     return (
#         Path("/data")
#         / os.environ["USER"]
#         / "vlm_learning"
#     )

def main():
    storage_root = f"/data / {os.environ['USER'] }/ vlm_learning"

    model_path = (
        storage_root
        / "models"
        / "blip-image-captioning-base"
    )

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

if __name__ == "__main__":
    main()