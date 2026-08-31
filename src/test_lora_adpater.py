import os
from pathlib import Path

import open_clip
import torch
import yaml

from lora_layers import(
    add_lora_to_linear_layers,
    count_trainable_parameters
)
#和train脚本的导入逻辑完全一致
def load_config(config_path):
    with open(config_path,"r",encoding="utf-8") as file:
        config =yaml.safe_load(file)

    return config

def main():
    config_path = "configs/flickr30k-debug.yaml"
    config = load_config(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    storage_root = f"/data/{os.environ['USER']}/vlm_learning"
    model_cache = f"{storage_root}/models/open_clip"

    adapter_path = (
        Path(storage_root)
        / config["output_dir"]
        / "epoch_01_lora.pt"
    )

    print("device:", device)
    print("adapter path:", adapter_path)

    model, _, _ = open_clip.create_model_and_transforms(
        config["model_name"],
        pretrained=config["pretrained"],
        cache_dir=model_cache,
        device=device,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    target_names = [
        "c_fc",
        "c_proj",
    ]

    add_lora_to_linear_layers(
        module=model,
        target_names=target_names,
        rank=8,
        alpha=16,
        dropout=0.0,
    )

    trainable, total = count_trainable_parameters(model)

    print(f"total parameters: {total:,}")
    print(f"trainable parameters: {trainable:,}")

        checkpoint = torch.load(
        adapter_path,
        map_location=device,
    )

    lora_state_dict = checkpoint["lora_state_dict"]

    model_lora_names = {
        name
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }

    saved_lora_names = set(lora_state_dict.keys())

    if model_lora_names != saved_lora_names:
        raise ValueError(
            "The LoRA adapter does not match the current model."
        )

    incompatible_keys = model.load_state_dict(
        lora_state_dict,
        strict=False,
    )

    if incompatible_keys.unexpected_keys:
        raise ValueError(
            f"Unexpected keys: {incompatible_keys.unexpected_keys}"
        )

    model.eval()

    print("LoRA adapter loaded successfully.")
    print(f"loaded epoch: {checkpoint['epoch']}")
    print(f"saved LoRA tensors: {len(lora_state_dict)}")

if __name__ == "__main__":
    main()