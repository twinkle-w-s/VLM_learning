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
        shuffule=False,#这里只做前向传播的验证，不打乱
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

if __name__ == "__main__":
    main()