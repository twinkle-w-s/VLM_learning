# 固定 probe 的跨轮学习动态

这部分已经实现：从 base、R1、R2……的真实评估产物中，对齐同一组训练 probe，
计算 NLL 变化、正确性历史、学习/遗忘事件，按来源和桶汇总。
它只负责观察与汇总，不训练、不改变配方、不启动下一轮。

全量训练清单与这部分没有冲突：全量数据进入训练，固定的小 probe 用来低成本地
反复观察。当前全量配置默认 probe 为 CLEVR 192、图文 replay 48、文字 replay 16；
实际数量以已保存的准备报告为准。不对其余训练样本虚构逐样本动态。

## 1. 文件职责

| 文件 | 职责 |
| --- | --- |
| `src/evaluation/evaluate_clevr.py` | 原有 val 生成评估新增 `--subset probe`，只输入问题和图片，记录模型真正生成的答案与严格 EM。默认仍评估 val。 |
| `src/evaluation/evaluate_nll.py` | 沿用已有实现，对固定 probe 的标准回答计算 teacher-forcing NLL，涵盖三种来源。 |
| `src/evaluation/nll_metrics.py` | 沿用 causal shift 和 assistant 监督掩码；组 NLL 为总 NLL/总监督 token 数。这次把 Torch 导入移入数值计算函数，使离线汇总无需 Torch。 |
| `src/learning/__init__.py` | 标记学习动态子包，支持 `python -m` 启动。 |
| `src/learning/artifacts.py` | 读取每次 NLL/EM 产物、核对协议和样本、读取 checkpoint 的实际 `global_step`。 |
| `src/learning/dynamics.py` | 纯 CPU 的指标计算：逐样本历史、最近增益、学习/遗忘次数、来源/桶汇总。不读模型权重。 |
| `src/learning/collect_dynamics.py` | 命令入口：读取历史 YAML，调用校验和计算，保存三个结果文件；默认只预览。 |
| `configs/dynamics_full.yaml` | 声明观察顺序与评估目录。先仅启用 base；真实轮次评估完成后再追加。 |
| `tests/test_dynamics.py` | 用合成产物检查历史对齐、加权 NLL、遗忘恢复、步数口径、预览和拒绝覆盖。 |
| `tests/test_probe_evaluation.py` | 模拟生成接口，检查 probe/val 区分、不输入 assistant 答案、smoke 标志。不是模型效果测试。 |

接口的数据流为：

```text
固定训练 probe + base/真实 adapter
  ├─ evaluate_nll: samples.jsonl + metrics.json
  └─ evaluate_clevr --subset probe: predictions.jsonl + metrics.json
                ↓
       dynamics_full.yaml 中的观察顺序
                ↓
       artifacts 校验 → dynamics 计算
                ↓
sample_dynamics.jsonl + summary.json + history_used.json
```

## 2. 指标具体怎么算

NLL 是标准回答的负对数似然，不是自由生成答案的正确率。两者必须分别计算。
NLL 的监督范围沿用当前 Swift 模板，包含 assistant 正文，以及模板监督的结束标记/
换行；prompt、图片和 pad 不进入损失。`answer_token_count` 因而不一定等于正文
token 数。生成评估只输入 user 问题，不把标准回答提供给模型。

逐样本主要字段：

| 字段 | 含义 |
| --- | --- |
| `nll_history` | 每个已观察 checkpoint 的平均/总 NLL、监督 token 数、累计 optimizer steps、correct。 |
| `correctness_history` | CLEVR 的布尔正确性序列；replay 为 null 序列。 |
| `latest.nll_delta` | 当前 NLL − 上一次 NLL；负数表示下降。 |
| `latest.nll_gain` | 上一次 NLL − 当前 NLL；正数表示改善。 |
| `latest.relative_nll_gain` | gain / 上一次 NLL；上一次为 0 时为 null。 |
| `latest.nll_delta_per_optimizer_step` | delta / 两次观察之间实际 optimizer 更新次数。 |
| `baseline_nll_gain` | base NLL − 当前 NLL。只有 base 时为 null。 |
| `latest_transition` | 最近一次变化：learned、forgotten、stable_correct、stable_wrong；首次为 first_observation，replay 为 not_scored。 |
| `learning_count` | 相邻已观察 checkpoint 中，错误→正确的次数。 |
| `forgetting_count` | 相邻已观察 checkpoint 中，正确→错误的次数。 |

例如 CLEVR 正确性历史 `[true, false, true]` 表示观测到一次遗忘和一次重新学会。
这不是每个训练 step 的事件统计：两次观测之间发生过几次变化无法推断。

`summary.json` 分为 `overall`、`by_source` 和 `by_bucket`。每组保留完整历史、最近
NLL/准确率变化、总学习/遗忘事件和曾遗忘样本数。准确率只使用该组可评分的
CLEVR 样本，`scored_n` 显示分母；replay 不计算 EM，其事件计数为 null，不能解释为
“没有遗忘”。组 NLL 按监督 token 加权，不能直接平均每条样本的 mean_nll。

组 NLL 能用于观察同一固定组随训练的变化。不同来源回答长度和分布差别很大，
不能用 CLEVR 与 replay 的 NLL 大小直接排序能力；overall NLL 也可能被长回答主导。
probe 的小桶统计用于描述，不自动取得 val 的反馈资格。

生成评估支持 `--batch-size` 加速 val，但学习动态要求各轮生成协议中的 batch size
相同。当前 base probe 以 batch=1 建立，后续 probe 继续用默认1；不要直接将
batch=4 的 probe 混入已有历史。NLL 评估仍是原来的逐条计算。

## 3. 历史与步数口径

当前只支持 `step_accounting: round_local_reset`：每轮加载上一轮 adapter 权重，
重新建立 optimizer/scheduler，global_step 从 0 开始。当前训练入口只实现 R1，
R2 continuation 启动仍需后续实现；历史汇总可以接收按此口径训练出的任意连续轮次。

base 的步数为 0；R1 checkpoint 若保存 global_step=300、R2 保存 global_step=200，
观察位置为 `[0, 300, 500]`。读取的是 `trainer_state.json`，不是 YAML 的计划步数。
步数是 optimizer 更新次数，不是 micro-batch 数、样本曝光次数或每个样本被训练
的次数。每 step 平均下降量也不等于“增配这个桶带来的因果收益”。

每轮历史只登记一个观察点。若一轮中登记多个累计计数的 checkpoint，不能把它们
当成独立轮次相加。精确 resume 的跨轮累计 global_step 也不支持这一相加口径。
轮次必须是连续的 0、1、2……，后续 checkpoint 路径不得重复。

汇总器会检查完整 probe、NLL/EM 协议、checkpoint 一致性、样本 ID/来源/桶/难度/
监督 token 数一致性、CLEVR 问题/图片/标准答案不变，以及逐样本结果与 metrics
一致性。val、smoke、部分评估、协议变化会报错。

这些检查不能证明训练谱系：独立从 base 训练的两个 adapter 不会自动被识别为
非 continuation。历史必须由实际运行记录按正确顺序登记。已经准备好的三类 probe
清单和图片须保持不变，尤其 replay 的旧 NLL 产物不包含完整问题/答案，不能只靠
其样本 ID 和 token 数证明文本未被修改。要重做 probe，就另开实验并重评 base。

## 4. 服务器操作：先补齐 baseline

所有命令从服务器项目根目录执行，代码通过 Git 同步后再运行：

```bash
cd /home/wangshuchang/projects/VLM_learning
conda activate qwen-mvp
export QWEN_FULL_CONFIG="/data/${USER}/vlm_learning/qwen_full_v1/full_runtime.yaml"
test -f "$QWEN_FULL_CONFIG" || { echo "找不到 full_runtime.yaml"; exit 1; }
```

模型评估需要一张已获分配的卡。先将 `QWEN_GPU` 设置为你的卡号，再执行：

```bash
export CUDA_VISIBLE_DEVICES="${QWEN_GPU:?请先设置 QWEN_GPU 为获分配的 GPU 卡号}"
```

先检查以下目录有没有完整 `metrics.json`。已有 COMPLETE probe 结果就复用，
不要重复执行到同一输出目录；失败或不完整的旧目录应另行保留并排查，不覆盖它。

```text
/data/${USER}/vlm_learning/qwen_full_v1/baseline/nll_probe/
/data/${USER}/vlm_learning/qwen_full_v1/baseline/clevr_probe/
```

缺少哪一项，运行对应命令，不带 `--limit`：

```bash
python -m qwen.src.evaluation.evaluate_nll \
  --config "${QWEN_FULL_CONFIG:?未设置配置路径}" \
  --subset probe --execute

python -m qwen.src.evaluation.evaluate_clevr \
  --config "${QWEN_FULL_CONFIG:?未设置配置路径}" \
  --subset probe --execute
```

第一项默认写 `baseline/nll_probe`，第二项写 `baseline/clevr_probe`。
两者 COMPLETE 后先离线预览，再保存：

```bash
python -m qwen.src.learning.collect_dynamics \
  --history qwen/configs/dynamics_full.yaml \
  --output "/data/${USER}/vlm_learning/qwen_full_v1/learning/base"

python -m qwen.src.learning.collect_dynamics \
  --history qwen/configs/dynamics_full.yaml \
  --output "/data/${USER}/vlm_learning/qwen_full_v1/learning/base" \
  --execute
```

此时 `BASELINE_ONLY` 是正常状态，差值为 null，并非失败。数据准备的
`FULL_DATA_READY` 不等于这些模型评估产物已经存在。

## 5. R1 训练完成后的操作

先确认真实保存的正式 adapter，不能用 smoke checkpoint 冒充 R1：

```bash
find "/data/${USER}/vlm_learning/qwen_full_v1/rounds/round01/sft" \
  -type f -name adapter_config.json
```

将实际 checkpoint 目录赋给 `QWEN_CHECKPOINT`；不要仅依据计划步数猜名称。
以下命令读取该 adapter，输出目录与历史 YAML 中 round01 示例一致：

```bash
python -m qwen.src.evaluation.evaluate_nll \
  --config "${QWEN_FULL_CONFIG:?未设置配置路径}" \
  --subset probe \
  --checkpoint "${QWEN_CHECKPOINT:?请设置实际保存的正式 checkpoint 目录}" \
  --output "/data/${USER}/vlm_learning/qwen_full_v1/rounds/round01/eval/nll_probe" \
  --execute

python -m qwen.src.evaluation.evaluate_clevr \
  --config "${QWEN_FULL_CONFIG:?未设置配置路径}" \
  --subset probe \
  --checkpoint "${QWEN_CHECKPOINT:?请设置实际保存的正式 checkpoint 目录}" \
  --output "/data/${USER}/vlm_learning/qwen_full_v1/rounds/round01/eval/clevr_probe" \
  --execute
```

两项评估成功后，取消 `configs/dynamics_full.yaml` 中 round01 的注释，保留 base：

```bash
python -m qwen.src.learning.collect_dynamics \
  --history qwen/configs/dynamics_full.yaml \
  --output "/data/${USER}/vlm_learning/qwen_full_v1/learning/through_round01" \
  --execute
```

此时应为 `DYNAMICS_READY`，有 base→R1 的变化。R2 产生后按相同方式评估、追加
round02、另存 `learning/through_round02`。每次重建全部历史，旧目录保留，不追加
修改旧结果。

结果目录包含：

```text
learning/through_round01/
├── sample_dynamics.jsonl  # 每个固定 probe 样本一行，含跨轮历史
├── summary.json          # 整体、来源、桶的指标和历史配置记录
└── history_used.json     # 本次读取的路径、协议、checkpoint 和实际步数
```

val 评估继续独立运行：`evaluate_clevr --subset val` 和 `evaluate_nll --subset val`。
它们用于后续反馈控制，不能替换这里的训练 probe，也不能把 val 错题加入训练。
本次不实现 EMA/配比控制器、下一轮配方物化、R2 训练入口或 MMStar；这些是下一阶段。

## 6. 验收范围

```bash
python -m unittest discover -s qwen/tests -v
python -m compileall -q qwen/src qwen/tests
```

本地测试不依赖 GPU，验证的是文件接口和指标算法；合成观察不代表真实模型效果。
服务器仍需验证真实模型的 probe NLL/生成评估与正式 adapter 加载。
