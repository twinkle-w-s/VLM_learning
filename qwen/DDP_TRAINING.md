# 单机多卡 LoRA：先短测，再决定是否使用 8 卡

当前入口 `qwen.src.training.train_sft` 支持单卡及单机多卡 DDP，默认仍为单卡。
本次不重新准备数据、不改变数据配比、不改变 LoRA 范围，也不做精确 resume。
R1 和 smoke 都从原始模型新建 adapter。

## 改动的文件

- `src/training/train_sft.py`：按卡数计算梯度累积，多卡调用 torchrun；检查显式
  GPU 选择；允许指定新输出目录；保存完整子进程 stdout/stderr 和退出码。
- `tests/test_training_ddp.py`：CPU 回归测试，检查 1/2/4/8 卡 batch 公式、预算不变、
  单卡兼容、输出隔离、GPU 数量校验和错误日志。不是实际 DDP 通信或训练效果测试。
- 本文档及 README：记录启动、看日志和断线后重连方法。

## Batch 与预算

```text
有效 global batch = 每卡 micro-batch × GPU 数量 × 梯度累积次数
```

| 卡数 | 每卡 micro-batch | 梯度累积 | 有效 batch |
| --- | --- | --- | --- |
| 1 | 2 | 16 | 32 |
| 2 | 2 | 8 | 32 |
| 4 | 2 | 4 | 32 |
| 8 | 2 | 2 | 32 |

因此正式 R1 仍计划 12,159 optimizer steps、389,088 次样本曝光，nominal_epochs=1。
学习率、长度上限、图片预算和来源配比不变。DDP 的数据采样顺序、dropout 和数值
归约与单卡可能不同；保持有效 batch 和训练预算不是逐步数值完全等价的保证。

DDP 在每张卡放完整基础模型，训练时同步可训练 LoRA 梯度；不是将模型权重切成8份，
每张卡仍需装下原模型与 micro-batch=2 的激活。不会使用 DeepSpeed 或 TP。

新增参数：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--num-gpus` | 1 | 单机进程/卡数；与 CUDA_VISIBLE_DEVICES 数量一致。 |
| `--micro-batch-size` | YAML 值 | 临时覆盖每卡 batch，并写入 launch 配置快照；不修改原 YAML。 |
| `--smoke-steps` | YAML 值 | 只允许 smoke 使用，用于短测吞吐，不改变正式 R1 预算。 |
| `--output-dir` | 原默认目录 | 新尝试写到新目录，复用原训练清单、保留旧失败记录。 |

不要对项目启动器再套一层 torchrun，它内部会启动工作进程。
多卡命令使用 `python -m torch.distributed.run --standalone ... --module swift.cli.sft`。
已核对 ms-swift v4.5.3 的 SFT 模块可作为进程入口，torchrun 为每个 worker 提供 rank。
启动器清理旧的分布式启动环境，使用单机 standalone rendezvous，避免继承旧 world size
或在 Swift CLI 中重复分派。多卡设置 `ddp_find_unused_parameters=false`，与当前
仅 LLM LoRA 可训练、gradient checkpointing 的配置配合。

## 服务器：必须先确认8张卡均已分配

代码同步后，从服务器项目根目录执行。只使用明确获分配的卡，不能把别人占用的
卡直接加入列表，也不能因为 nvidia-smi 暂时显示空闲就认定获得使用权限。

```bash
nvidia-smi
nvidia-smi topo -m
```

推荐在 tmux 中启动，避免 SSH 断开终止前台任务：

```bash
tmux new -s qwen8
cd /home/wangshuchang/projects/VLM_learning
conda activate qwen-mvp

read -r -p "请输入已分配的8张GPU编号，逗号分隔: " QWEN_GPUS
export CUDA_VISIBLE_DEVICES="$QWEN_GPUS"
export QWEN_FULL_CONFIG="/data/${USER}/vlm_learning/qwen_full_v1/full_runtime.yaml"
```

没有 tmux 时先确认可用的会话保持工具，不要为这个任务盲目执行全局安装。

## 先预览和短测

此 smoke 使用同一份全量训练清单，只训练60个 optimizer steps；预处理仍可能扫描
全部清单，所以总耗时不只有60步。当前保留原有 eager tokenization 和 strict 数据
处理策略，不为提速静默更改过滤规则或长度预算。

```bash
python -m qwen.src.training.train_sft \
  --config "${QWEN_FULL_CONFIG:?未设置配置路径}" \
  --mode smoke --num-gpus 8 --micro-batch-size 2 --smoke-steps 60 \
  --output-dir "/data/${USER}/vlm_learning/qwen_full_v1/smoke/sft_ddp8"
```

预览应看到 world_size=8、micro_batch_size=2、gradient_accumulation_steps=2、
global_batch_size=32、max_steps=60。不加载模型、不创建输出目录。

确认无误后执行：

```bash
python -m qwen.src.training.train_sft \
  --config "${QWEN_FULL_CONFIG:?未设置配置路径}" \
  --mode smoke --num-gpus 8 --micro-batch-size 2 --smoke-steps 60 \
  --output-dir "/data/${USER}/vlm_learning/qwen_full_v1/smoke/sft_ddp8" \
  --execute
```

8卡日志交错是正常的。检查模型仅训练 LLM LoRA、无 OOM/NCCL 错误、所有 worker 完成、
checkpoint 可保存，并验收 adapter。首次短测遇到问题时保留目录，后续用新名称尝试。

若 micro-batch=2 OOM，可在新的输出目录用 `--micro-batch-size 1`；8卡会自动累积4次，
有效 batch 仍为32。遇到 NCCL 错误要保留原日志并检查拓扑/资源，不默认禁用 P2P，
不杀其他用户的进程。

吞吐比较时排除前10步预热，固定 micro-batch=2、global batch=32，比较 optimizer-step
秒数、samples/s、峰值显存及准备阶段耗时。数据长度和采样顺序会造成波动；不能只拿
一个训练 step 比较。正式步数乘稳定段平均 step 时间只是训练阶段的近似耗时，
还需加上预处理、加载和 checkpoint 保存时间。

可把同样短测中的 `--num-gpus` 改为4/2/1，并为该次任务重新设置相应数量的已分配
GPU、使用新输出目录。不要同时启动多个对照占用同一组卡。
4090 没有 NVLink，PCIe/NUMA、CPU 数据预处理和磁盘读取会影响扩展效率；不保证8倍加速。

## 短测通过后正式 R1

```bash
python -m qwen.src.training.train_sft \
  --config "${QWEN_FULL_CONFIG:?未设置配置路径}" \
  --mode round1 --num-gpus 8 --micro-batch-size 2 \
  --output-dir "/data/${USER}/vlm_learning/qwen_full_v1/rounds/round01/sft_ddp8" \
  --execute
```

新 checkpoint 位于 `rounds/round01/sft_ddp8/`，旧 `rounds/round01/sft/` 保留。
不加载短测 adapter，而是从 base 新建正式 LoRA。后续评估将实际保存的 checkpoint
作为 `--checkpoint` 传入；probe/val 数据和动态历史目录不变。
例如用以下命令寻找本次 adapter，不能按计划步数认定一定保存成功：

```bash
find "/data/${USER}/vlm_learning/qwen_full_v1/rounds/round01/sft_ddp8" \
  -type f -name adapter_config.json
```

## 看日志、断线与失败

按 Ctrl+B 然后 D 退出 tmux 界面，不会终止内部训练。重新连接服务器后：

```bash
tmux attach -t qwen8
```

正式训练完整日志：

```bash
tail -n 60 -f "/data/${USER}/vlm_learning/qwen_full_v1/rounds/round01/sft_ddp8/console.log"
```

短测则看 `smoke/sft_ddp8/console.log`。Swift 的 `logging.jsonl` 通常在训练开始后出现；
模型加载、数据编码和 NCCL 初始化的错误应先看 console.log。

输出目录：

```text
sft_ddp8/
├── launch.json          # 计划、实际覆盖后的配置、启动命令和GPU列表
├── console.log          # 子进程stdout+stderr，及时刷新
├── process_result.json  # 子进程结束后记录returncode
├── logging.jsonl        # Swift训练指标，训练进入更新阶段才可能出现
└── checkpoint-*/        # Swift保存的真实训练产物
```

returncode=0 表示子进程正常结束，仍需要验收 checkpoint；非0表示失败。
若父进程遭到断线/kill，process_result.json 可能不存在；launch.json 存在也不表示
训练成功。console.log 不是 SSH 会话保持机制，必须配合 tmux 等工具。

## 本地验证与参考

```bash
python -m unittest discover -s qwen/tests -v
python -m compileall -q qwen/src qwen/tests
```

本地只能验证命令生成、batch预算、预览、日志和退出码，不代表服务器8卡已成功通信。

- [ms-swift v4.5.3 CLI 分派](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/cli/main.py)
- [ms-swift v4.5.3 SFT 入口](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/cli/sft.py)
- [PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)
