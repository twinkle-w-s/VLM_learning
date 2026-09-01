# 项目结构与学习进度

## 当前目录

```text
VLM_learning/
├── configs/                 # 训练配置文件
│   └── flickr30k-debug.yaml
├── data/                    # 数据集缓存或本地处理后的数据
├── checkpoints/             # 完整模型 checkpoint
├── outputs/                 # 训练输出、LoRA adapter 和评估结果
├── notebooks/               # 探索性实验 notebook
├── scripts/                 # 数据下载等辅助脚本
├── src/                     # 项目主要 Python 代码
│   ├── flickr30k_datasets.py
│   ├── inspect_flickr30k.py
│   ├── test_dataloader.py
│   ├── test_clip_forward.py
│   ├── evaluate_retrieval.py
│   ├── lora_layers.py
│   ├── train_clip.py
│   └── test_lora_adpater.py
├── requirements.txt         # Python 依赖
└── readme.md                # 项目简介
```

## 当前学习进度

```text
[已完成] Flickr30k 数据读取
[已完成] CLIP 图文对比损失
[已完成] CLIP 全量微调流程
[已完成] 手写 LoRA Linear 层
[已完成] CLIP Linear 层 LoRA 注入
[已完成] 冻结基础模型并只训练 LoRA 参数
[已完成] 保存和加载 LoRA adapter
[下一步] BLIP 原始模型图像描述
[后续] BLIP LoRA 监督微调
```

## 文件职责

- `flickr30k_datasets.py`：读取 Flickr30k 并返回图像、token 和 caption。
- `evaluate_retrieval.py`：计算图文检索 Recall@K。
- `lora_layers.py`：实现 LoRA 线性层、递归替换和参数统计。
- `train_clip.py`：训练 CLIP、验证并保存 checkpoint。
- `test_lora_adpater.py`：重新创建 CLIP 并验证 LoRA adapter 是否能加载。
- `inspect_flickr30k.py`：查看原始数据集结构。
- `test_dataloader.py`：检查 DataLoader 输出形状。
- `test_clip_forward.py`：检查 CLIP 前向传播和梯度。

## 目录约定

目前保留 `src` 下的平面文件结构，不移动已有 Python 文件，以保证服务器上的导入命令继续有效。后续新增 BLIP 脚本也先放在 `src/` 中，等 CLIP 和 BLIP 教学主线完成后再统一重构目录。
