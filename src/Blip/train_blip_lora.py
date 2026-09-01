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



def main():
    config = load_config(
        "blip-debug.yaml"
    )

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
        cache_dir=dataset_cache,
    ).to(device)

    processor = BlipProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model.train()

    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_config = config["lora"]

    target_names = lora_config("target_names")
    rank = lora_config("rank")
    alpha = lora_config("alpha")
    dropout = lora_config("dropout")

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
    print(f"Trainable parameters: {trainable:,}, Total parameters: {total:,}")#按照千位分隔符格式化输出
    print("replaced layers:", replaced_layers)


    print("trainable parameter names:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name, tuple(parameter.shape))#tuple(parameter.shape)返回参数的形状，例如torch.Size([512, 512])，然后转换为元组(512, 512)

if __name__ == "__main__":
    main()