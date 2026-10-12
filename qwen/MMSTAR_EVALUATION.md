# MMStar：观察 SFT 前后的通用多模态能力变化

这部分实现固定 MMStar 图片选择题的 base→R1（或后续锁定 adapter）对比。
不会训练、不会更新配方，不使用外部 LLM judge，也不计算 MMStar 官方的 MG/ML。
这里的分数是本项目固定图片预算和解析规则下的准确率，不直接对照官方榜单。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `src/data/prepare_mmstar.py` | 下载一个固定版本 Parquet，验证1500题和6类各250题，解码图片并按原始字节导出，保存评估清单。 |
| `src/evaluation/mmstar_metrics.py` | 固定数据版本、构建提示、解析选项字母、汇总分类分数，以及逐样本训练前后对比。纯 CPU。 |
| `src/evaluation/evaluate_mmstar.py` | 沿用已验证的 Swift Qwen/LoRA 加载方式，单卡批量生成，只输入问题和图片，保存预测与指标。 |
| `src/evaluation/compare_mmstar.py` | 读取两份完整结果，校验数据/协议一致，保存差值和逐样本得失。不加载模型。 |
| `tests/test_mmstar.py` | 用合成数据验证图片导出、字母解析、无效答案分母、batch 对齐、预览和退化对比。不是模型性能结果。 |

原来的 `full_runtime.yaml` 已包含 `evaluation.mmstar`，这次无需重建全量训练数据或配置。
数据来自官方 `Lin-Chen/MMStar` 的 `val` 配置/划分。名称 val 是上游命名，
在本项目中视为外部保留评估集，不放入 train、replay 或配方控制器。

固定版本：`bc98d668301da7b14f648724866e57302778ab27`。
只下载 `mmstar.parquet`，约41.8MB，不下载 TSV、其他项目或整个数据仓库。
图片再导出一份，因此需为原 Parquet、导出图片和少量元数据预留空间。

## 1. 准备数据

代码同步到服务器，在 qwen-mvp 环境执行：

```bash
cd /home/wangshuchang/projects/VLM_learning
conda activate qwen-mvp
python -c "import pyarrow, PIL, huggingface_hub; print('MMSTAR DEPENDENCIES: PASS')"

python -m qwen.src.data.prepare_mmstar --execute
```

现有 replay 处理用 PyArrow/Pillow，模型下载用 HF Hub；不要求额外安装 datasets 或
VLMEvalKit，不升级既有环境依赖。默认无 `--execute` 时只预览，不访问网络。

默认数据位置：

```text
/data/${USER}/vlm_learning/benchmarks/mmstar/
├── raw/mmstar.parquet   # 原始41.8MB数据，含内嵌图片
├── images/              # 每题原图，不进行重压缩或 resize
├── manifest.jsonl       # 固定题目、答案、能力标签、图片路径
└── report.json          # 数据版本、数量及6类分布
```

成功状态为 `MMSTAR_READY`。已有完整数据就复用，不重复准备；导出中断的目录保留，
需要时用 `--data-root` 指定新目录，并在后续评估中使用相同目录。
如果只在首条图片处理时失败而留下空 images 目录，可以直接重试；脚本兼容原始
Parquet 的 bytes 图片和含 bytes 的字典。已有 raw/mmstar.parquet 时离线读取，
不重新下载；非空图片目录或已有 manifest/report 仍拒绝覆盖。
官方下载地址也可用于核对：[固定版本 mmstar.parquet](https://huggingface.co/datasets/Lin-Chen/MMStar/resolve/bc98d668301da7b14f648724866e57302778ab27/mmstar.parquet)。

## 2. 配置及短测

两次评估都使用同一份 `full_runtime.yaml` 和同一套准备后的数据。现有配置：

```yaml
evaluation:
  mmstar:
    dataset: Lin-Chen/MMStar
    config: val
    split: val
    max_new_tokens: 8
    feedback: false
```

以上只是相关片段，不要覆盖其他配置字段。固定 BF16、SDPA、temperature=0、图片
预算与 max_length 沿用 train 配置。默认 batch=4，base/adapter 必须一致；若因显存
需要改为2，则两份正式评估都用2。图片预算会影响小字/细节识别，不改变预算后再
将变化归因于 SFT。

确认另一张卡也已分配给你。若 CLEVR 正在3号卡上跑，MMStar 可使用另一张已分配
且有足够显存的卡；不要让两个评估在同一张卡上争用显存。

```bash
read -r -p "请输入用于MMStar的一张已分配GPU卡号: " MMSTAR_GPU
export CUDA_VISIBLE_DEVICES="$MMSTAR_GPU"
export QWEN_FULL_CONFIG="/data/${USER}/vlm_learning/qwen_full_v1/full_runtime.yaml"

python -m qwen.src.evaluation.evaluate_mmstar \
  --config "${QWEN_FULL_CONFIG:?未设置配置路径}" \
  --batch-size 4 --limit 32 \
  --output "/data/${USER}/vlm_learning/qwen_full_v1/smoke/mmstar_base_b4" \
  --execute
```

检查无 OOM、响应与题目对齐、无效输出数及截断数。带 limit 的结果为 SMOKE_COMPLETE，
即使恰好覆盖1500条也不用于正式对比。不要重复覆盖旧目录。

## 3. 正式基础模型评估

不传 checkpoint，就评估原始 Instruct 权重。即使 R1 已训练完，也能现在补测 base：

```bash
python -m qwen.src.evaluation.evaluate_mmstar \
  --config "${QWEN_FULL_CONFIG:?未设置配置路径}" \
  --batch-size 4 \
  --output "/data/${USER}/vlm_learning/qwen_full_v1/baseline/mmstar" \
  --execute
```

## 4. 正式 adapter 评估

复用 CLEVR 评估中已确认的正式 `QWEN_CHECKPOINT`，不能使用 smoke adapter，也不要
从计划步数猜目录。base 和 adapter 两次评估可以依次在同一张卡运行：

```bash
python -m qwen.src.evaluation.evaluate_mmstar \
  --config "${QWEN_FULL_CONFIG:?未设置配置路径}" \
  --checkpoint "${QWEN_CHECKPOINT:?请设置实际保存的正式adapter目录}" \
  --batch-size 4 \
  --output "/data/${USER}/vlm_learning/qwen_full_v1/rounds/round01/eval/mmstar" \
  --execute
```

各输出目录有 `predictions.jsonl` 和 `metrics.json`，其中保存总体、6类和细类准确率，
以及无效、空回答、长度截断数量。只有完整执行完才写出 COMPLETE metrics。
坏图或推理异常直接报错，不静默丢样本，也不生成正式成功结果。

## 5. 对比与解读

两次正式评估成功后执行，不需要 GPU：

```bash
python -m qwen.src.evaluation.compare_mmstar \
  --base-dir "/data/${USER}/vlm_learning/qwen_full_v1/baseline/mmstar" \
  --after-dir "/data/${USER}/vlm_learning/qwen_full_v1/rounds/round01/eval/mmstar" \
  --output "/data/${USER}/vlm_learning/qwen_full_v1/rounds/round01/eval/mmstar_comparison" \
  --execute
```

结果包含 `comparison.json` 和 `paired_changes.jsonl`：

- `base_accuracy` / `after_accuracy`：原始模型与 adapter 的总体准确率。
- `delta_pp`：after−base，单位为百分点。示例：55%→53%为−2个百分点，非真实结果。
- `correct_to_wrong` / `wrong_to_correct`：同一题上观测到的能力损失/收益数量。
- `by_category`：哪些能力类别下降或改善。
- `base_invalid` / `after_invalid`：解析无效数量，帮助区分格式问题与题目答错。

提示只要求选项字母。解析器接受 A/B/C/D，忽略大小写和外层括号/方括号、结尾句号；
不从解释中抽取或猜测答案，`A or B`、`The answer is A.` 等记为无效并计错，不从
1500题分母中删除。这是本项目的固定解析协议，不冒充官方 judge 的评分。

比较器检查完整1500题、同一数据版本/问题/图片路径/答案/标签，以及 batch、解码、
长度和图片预算一致。协议不一致时不生成“训练导致退化”的对比结果。
负差值表示此次 MMStar 上观测到下降，不能仅凭很小的差值声称统计显著，也不能
据一个 benchmark 推断全部多模态或文字能力。R1 若还会根据这些结果继续调参，
其结果应视为中间诊断，而不是未经使用的最终测试。

尚未回溯检测 MiniMind replay 与 MMStar 的精确/近重复重叠，也不能证明 Qwen 预训练
未见过这些题。结果显式记录 `project_training_overlap_checked=false`，不宣称已去污染。
本次不修改已训练的数据或反复自动重训，也不向配比控制器反馈 MMStar 分数。

## 本地验收与官方参考

```bash
python -m unittest discover -s qwen/tests -v
```

本地验收覆盖代码/文件接口；真实数据准备、模型加载和分数仍要在服务器验证。

- [官方数据文件与大小](https://huggingface.co/datasets/Lin-Chen/MMStar/tree/main)
- [官方数据 schema/config](https://huggingface.co/datasets/Lin-Chen/MMStar/blob/main/README.md)
- [官方评估说明：1500题、视觉输入与MG/ML](https://github.com/MMStar-Benchmark/MMStar/blob/main/eval/README.md)
- [官方能力类别组织](https://github.com/MMStar-Benchmark/MMStar)
