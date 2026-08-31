import os
from pathlib import Path

import open_clip
import torch
import yaml

from lora_layers import(
    add_lora_to_linear_layers,
    count_trainable_parameters
)

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

if __name__ == "__main__":
    main()