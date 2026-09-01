import os
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from blip_datasets import Flickr30KBLIPDataset


def get_storage_root():
    return (
        Path("/data")
        / os.environ["USER"]
        / "vlm_learning"
        # storage_root = f"/data/{os.environ['USER']}/vlm_learning"
    )

# def load_one_flickr30k_sample(dataset_cache):
#     dataset=load_dataset(
#         "nlphuji/flickr30k",
#         split="test",#这里的test好像仅表示全部数据是仓库中的外层分片 
#         cache_dir=str(dataset_cache),
#         trust_remote_code=True,
#     )
    
#     for sample in dataset:
#         if sample["split"]=="val":
#             image=sample["image"].convert("RGB")
#             caption=sample["caption"][0]

#             return image,caption
#     raise RuntimeError("no val data")#return执行了就到不了这里

def main():
    storage_root = get_storage_root()

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_path.exists():
        raise FileNotFoundError(
            f"BLIP model directory was not found: {model_path}"
        )

    print("device:", device)
    print("model path:", model_path)

    processor=BlipProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model= BlipForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
    ).to(device)

    model.eval()
    #这里把process和model取出来，本质和之前没有区别

    dataset = Flickr30KBLIPDataset(
        split="val",
        processor=processor,
        dataset_cache=dataset_cache,
    )

    sample = dataset[0]

    image = sample["pixel_values"].unsqueeze(0).to(device)
    reference_caption = sample["caption"]

    print("dataset size:", len(dataset))
    print("image shape:", image.shape)


    with torch.inference_mode():
        generated_ids=model.generate(
            pixel_values=image,
            max_new_tokens=30,
        )
    generated_caption=processor.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )#把token数字编号解码回文本

    print("generated token shape:", generated_ids.shape)
    print("reference caption:", reference_caption)
    print("BLIP generated caption:", generated_caption)

if __name__ == "__main__":
    main()