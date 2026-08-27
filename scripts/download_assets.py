import argparse
import json
import os
from collections import Counter
from pathlib import Path


def configure_cache(storage_root: Path) -> tuple[Path, Path]:
    dataset_cache = storage_root / "datasets" / "flickr30k"
    model_cache = storage_root / "models" / "open_clip"
    hf_home = storage_root / "cache" / "huggingface"
    torch_home = storage_root / "cache" / "torch"

    for path in (dataset_cache, model_cache, hf_home, torch_home):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(dataset_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TORCH_HOME"] = str(torch_home)
    return dataset_cache, model_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Flickr30K and OpenAI CLIP weights to /data."
    )
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()

    storage_root = args.storage_root.expanduser().resolve()
    dataset_cache, model_cache = configure_cache(storage_root)

    print(f"Storage root: {storage_root}", flush=True)
    print(f"Flickr30K cache: {dataset_cache}", flush=True)
    print(f"OpenCLIP model cache: {model_cache}", flush=True)

    from datasets import load_dataset

    print("Downloading Flickr30K...", flush=True)
    dataset = load_dataset(
        "nlphuji/flickr30k",
        split="test",
        cache_dir=str(dataset_cache),
        trust_remote_code=True,
    )
    split_counts = (
        dict(Counter(dataset["split"])) if "split" in dataset.column_names else {}
    )
    print(f"Flickr30K examples: {len(dataset)}", flush=True)
    print(f"Internal split counts: {split_counts}", flush=True)

    import open_clip
    import torch

    print("Downloading OpenAI CLIP ViT-B/32...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        cache_dir=str(model_cache),
        device="cpu",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    sample = dataset[0]
    captions = sample["caption"]
    caption = captions[0] if isinstance(captions, list) else captions
    image = preprocess(sample["image"]).unsqueeze(0)
    text = tokenizer([caption])

    with torch.inference_mode():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    manifest = {
        "storage_root": str(storage_root),
        "dataset_id": "nlphuji/flickr30k",
        "dataset_loader_split": "test",
        "dataset_cache": str(dataset_cache),
        "dataset_examples": len(dataset),
        "internal_split_counts": split_counts,
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "model_cache": str(model_cache),
        "image_feature_shape": list(image_features.shape),
        "text_feature_shape": list(text_features.shape),
    }
    manifest_path = storage_root / "assets.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(f"Image feature shape: {tuple(image_features.shape)}", flush=True)
    print(f"Text feature shape: {tuple(text_features.shape)}", flush=True)
    print(f"Manifest written to: {manifest_path}", flush=True)
    print("All assets downloaded and verified.", flush=True)


if __name__ == "__main__":
    main()
