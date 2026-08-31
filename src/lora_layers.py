import math
import torch
from torch import nn
from torch.nn import functional as F

class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer,
        rank=8,#rank表示低秩维度，rank越小训练参数越小
        alpha=16,
        dropout=0.0
    ):
        super().__init__()

        self.base_layer=base_layer
        self.rank=rank
        self.alpha=alpha
        self.scaling=alpha/rank
        self.dropout=nn.Dropout(dropout)

        self.lora_A=nn.Parameter(
            torch.empty(
                rank,
                base_layer.in_features,
            )
        )

        self.lora_B=nn.Parameter(
            torch.empty(
                base_layer.out_features,
                rank,
            )
        )
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )#kaiming初始化，设置极小的初值
        nn.init.zeros_(self.lora_B)#B设置为全0

        for parameter in self.base_layer.parameters():
            parameter.requires_grad=False

    def forward(self,x):
        base_output=self.base_layer(x)

        lora_output=F.linear(
            self.dropout(x),
            self.lora_A,
        )
        lora_output=F.linear(
            lora_output,
            self.lora_B
        )

        return base_output+self.scaling*lora_output


def add_lora_to_linear_layers(
        module,
        target_names,
        rank=8,
        alpha=16,
        dropout=0.0,
):
    for child_name, child_module in module.named_children():
        if(#named_children可以返回模块的直接子模块名称和子模块对象
            isinstance(child_module,nn.Linear)
            and child_name in target_names
        ):#是线性层，且name在list中
            lora_layer = LoRALinear(
                base_layer=child_module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            setattr(
                module,
                child_name,
                lora_layer,
            )#module对象的child_name,改为lora_layer
        else:
            add_lora_to_linear_layers(
                module=child_module,
                target_names=target_names,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )#递归调用，继续寻找线性层，逐层访问子模块

def count_trainable_parameters(model):
    trainable=0
    total=0

    for parameter in model.parameters():
        total+=parameter.numel()#统计当前参数张量里有多少个数字

        if parameter.requires_grad:
            trainable+=parameter.numel()#表示一个参数张量中有多少个元素
    return trainable,total

    
    