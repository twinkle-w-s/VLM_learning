import open_clip
from torch.utils.data import DataLoader
from flickr30k_datasets import Flickr30KDataset

def main():
    model_name="VIT-B-32"

    _,_,preprocess=open_clip.create_model_and_transforms(
        model_name,
        pretrained=None,
    )

    tokenizer=open_clip.get_tokenizer(model_name)

    train_dataset=Flickr30KDataset(
        split_init="train",
        preprocess=preprocess,
        tokenizer=tokenizer
    )

    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2# 两个进程同步读

    )



    batch=next(iter(train_loader))#iter创建迭代器，next取出第一个

    print("Number of training images:", len(train_dataset))
    print("pixel_values shape:", batch["pixel_values"].shape)
    print("input_ids shape:", batch["input_ids"].shape)

    print("\nCaptions in this batch:")
    for caption in batch["caption"]:
        print("-", caption)