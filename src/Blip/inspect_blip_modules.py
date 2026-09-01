import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BlipForConditionalGeneration

src_dir=Path(__file__).resolve().parent.parent

sys.path.insert(0,str(src_dir))

from lora_layers import add_lora_to_linear_layers, count_trainable_parameters, get_lora_state_dict

def main():
    storage_root = Path("/data") / os.environ["USER"] / "vlm_learning"

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

    model = BlipForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    print("linear layers in the BLIP model:")
    leaf_name_counts = {}#这是一个字典，用来统计叶子名称的数量

    for name,module in model.named_modules():
        if isinstance(module, torch.nn.Linear):#isinstance判断module是否是torch.nn.Linear的实例
            print(
                name,"infeatures=",module.in_features,
                "out_features=",module.out_features
            )
            leaf_name = name.rsplit(".",1)[-1]#取最后一个点之后的部分作为叶子名称
            leaf_name_counts[leaf_name] = leaf_name_counts.get(leaf_name,0)+1#如果叶子名称已经存在，就加1，否则就初始化为1

    print("Linear leaf name counts:")
    for name,count in sorted(leaf_name_counts.items()):#对字典的items()进行排序，返回一个列表，每个元素是一个(name,count)元组
        print(name,count)


if __name__ == "__main__":
    main()