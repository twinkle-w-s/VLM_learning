# 开源项目数据工程参考：Data-Juicer Sandbox、Open-Instruct 与 data_engineering_book

> 审阅日期：2026-09-13  
> 版本：v0.3；本次更新补充完整数据链路、共享 P0 平台、主线 A/B 最终边界、通用能力保持和 lakehouse 风格组织。
>
> 用途：为 MiniMind-V + CLEVR/真实 VQA 项目建立可复现、可迭代的数据工程方案。
>
> 重要范围说明：本次按用户要求不再下载仓库。当前本地可读源码为 `third_party/data-juicer-sandbox`、`third_party/open-instruct` 和既有的 `third_party/minimind-v`；`data_engineering_book` 不落地源码，仅参考其 GitHub 网页目录和公开章节说明。完整 Data-Juicer 主仓库也未在当前工作区形成可读本地快照，因此关于 Data-Juicer 主仓库的内容只采用 Sandbox 的调用契约和官方公开说明，不把它写成本地源码审阅结论。

> 执行细节：本文记录开源项目的借鉴边界；具体到当前仓库的 job、artifact、状态机、CLI、反馈控制器和验收步骤，统一以 [MINIMIND_V_EXECUTION_PLAYBOOK.md](MINIMIND_V_EXECUTION_PLAYBOOK.md) 为准。

## 1. 参考材料与证据边界

### 1.0 本次更新的真实数据依据

GQA 官方项目页（`cs.stanford.edu/people/dorarad/gqa/`）公开说明了数据的 scene graph、问题/答案文件、balanced 版本和多种诊断评估。对本项目最有用的字段是 `imageId`、`isBalanced`、问题类型、短答案/长答案、语义操作链，以及 scene graph 中的对象属性和关系。这里的结论仅用于选择和设计清洗管线；不把官方字段直接等同于本项目已经完成的清洗结果。

GQA 作为真实数据源的工程判断：它已经提供结构化标注和 balanced 划分，能降低接入成本，但仍需要本项目完成图像完整性检查、答案归一化、同图同问去重、冲突检查、image-level split、质量分层和可追溯 quarantine。因此它比 CLEVR 更适合验证真实数据工程，而不是只验证训练脚本。

主线 A 使用 CLEVR 验证反馈式课程和数据飞轮；主线 B 使用清洗后的 `gqa_aligned_v1` 研究混编和能力退化。二者共享数据契约和评估组件，但不形成“先做 A 才能做 B”的承接关系。

### 1.1 本地源码快照

| 项目 | 本地位置 | 版本标识 | 本次审阅内容 |
| --- | --- | --- | --- |
| Data-Juicer Sandbox | `third_party/data-juicer-sandbox` | `54ed2ce6196fd8782ba71fbaa06ad4f938d2d753` | README、UserGuide、配置、pipeline、hook、factory、data-pool manipulator、context 记录 |
| Open-Instruct | `third_party/open-instruct` | `b0c3b298f2015c11ac6f580716904acd39bd72fd` | Tülu 3 文档、SFT/DPO 配置、数据混合、dataset processor、data loader、rejection sampling、去污染目录 |
| MiniMind-V | `third_party/minimind-v` | `740d467ece78a0b7d2d976fcb424472095d4a688` | 视觉编码器冻结、Projector/LLM 解冻策略、训练脚本、数据格式与本项目已有 CLEVR 脚本 |

### 1.2 网页参考

- Data-Juicer Sandbox：<https://github.com/datajuicer/data-juicer-sandbox>
- Data-Juicer 主项目：<https://github.com/modelscope/data-juicer>
- Open-Instruct：<https://github.com/allenai/open-instruct>
- data_engineering_book：<https://github.com/datascale-ai/data_engineering_book>

### 1.3 不应从本报告推出的结论

- 不把任何一个项目的公开 recipe 当作 MiniMind-V 的最优超参数；它们的模型规模、硬件、数据和目标不同。
- 不把“工业项目常用”写成统一行业标准。数据配比必须由本项目的验证指标和计算预算决定。
- 不把开发阶段的小子集、短训练、低分辨率运行写入最终论文主方法；它们只能用于工程验证和配方筛选。

## 2. Data-Juicer Sandbox：反馈驱动的数据—模型协同开发

### 2.1 项目定位

Sandbox 不是单纯的数据清洗脚本，而是一个把数据分析、数据 recipe、模型训练、模型推理和评测串起来的中间层。其基本闭环是：

```text
Probe 数据/模型
    ↓
根据 probe 结果修改 recipe
    ↓
处理数据并训练模型
    ↓
评估数据与模型
    ↓
进入下一轮迭代或达到目标后停止
```

这一点与当前 MiniMind-V 计划的 Round 1 → 按能力分箱 → Round 2 配方更新高度吻合。

### 2.2 顶层目录结构

```text
third_party/data-juicer-sandbox/
├── configs/
│   ├── demo/
│   ├── data/
│   ├── internvl_coco_caption/
│   ├── easyanimate_text_to_video/
│   └── auto_prompt_optimization/
├── data_juicer_sandbox/
│   ├── cli/sandbox_starter.py
│   ├── pipelines.py
│   ├── hooks.py
│   ├── factories.py
│   ├── context_infos.py
│   ├── data_pool_manipulators.py
│   ├── evaluators.py
│   ├── model_executors.py
│   ├── env_manager.py
│   ├── helper_funcs.py
│   ├── utils.py
│   └── specific_hooks/
│       ├── intervl_coco_captioning/
│       ├── rft/
│       └── text_to_video/
├── thirdparty/
│   ├── mm_eval/
│   └── models/
├── docs/
├── tests/
├── pyproject.toml
└── README.md
```

### 2.3 配置层：把一轮实验表达成可审计 recipe

`configs/demo/sandbox.yaml` 将一轮实验拆成四类 job：

1. `probe_job_configs`：分析原始数据或模型推理结果；
2. `refine_recipe_job_configs`：根据 probe 结果生成新的数据 recipe；
3. `execution_job_configs`：实际处理数据、构建数据池、训练模型；
4. `evaluation_job_configs`：分析处理后数据、评估数据质量和模型性能。

每个 job 具有 `hook`、`meta_name`、`dj_configs`、`extra_configs` 等字段。前一个 job 的输出可通过类似 `-1.refined_recipe_path` 的引用传入后一个 job。这种做法有三个直接经验：

- 数据配方不是散落在 Python 条件分支里的隐式状态，而是显式配置；
- job 输出必须有名字，后续步骤引用“输出名”而不是猜文件名；
- 原始数据、处理后数据、模型和评估结果属于同一条可追踪实验链。

对 MiniMind-V 的对应物应是：`round1.yaml` 产生 `round1/train_manifest.parquet`，训练产生 `round1/val_predictions.parquet`，反馈脚本再产生 `round2.yaml`，而不是覆盖原始 parquet。

### 2.4 `pipelines.py`：四阶段执行器与上下文

`SandboxPipeline.one_trial` 按固定顺序运行四组 hook。`SandBoxExecutor` 还提供：

- `max_iter_num`：最大迭代轮数；
- `iter_targets`：满足目标指标即可停止，例如某个输出 `>= target_value`；
- `iter_targets_mode`：要求全部目标满足或任意一个满足；
- `iter_updater`：把上一轮结果写回下一轮 pipeline 配置；
- `resume` 和 `context_infos.json`：中断后恢复已完成的 pipeline/job。

最值得借鉴的不是“自动循环”本身，而是它把反馈的三个对象分开：

```text
结果记录：ContextInfos
反馈指标：Watcher/W&B 中的命名结果
配置更新：iter_updater 显式映射
```

当前项目不必直接引入 W&B 作为唯一状态库，但应保留这三个概念：`metrics_by_bucket.json`、`predictions.parquet` 和 `recipe_next.yaml` 分开保存。

### 2.5 `hooks.py`：可替换的关键操作

已实现或注册的 hook 包括：

- `ProbeViaAnalyzerHook`：调用数据分析器并记录数据统计；
- `ProbeViaModelInferHook`：抽样数据，运行模型推理，生成模型侧 probe；
- `RefineRecipeViaKSigmaHook`：使用 k-sigma 策略根据统计量修改 recipe；
- `RefineRecipeViaModelFeedbackHook`：预留模型反馈修改 recipe 的接口，但当前实现仍是 TODO/未实现；
- `ProcessDataHook`：按 recipe 处理数据；
- `DataPoolManipulationHook`：构建、组合、复制、排序、下采样、合并数据池；
- `TrainModelHook` / `InferModelHook`：连接外部模型训练和推理；
- `EvaluateDataHook` / `EvaluateModelHook`：分别评估数据质量和模型效果。

对本项目有一个重要的现实提醒：Sandbox 的“模型反馈改 recipe”在当前本地代码中并不是开箱即用的完整算法，而是一个扩展点。因此 MiniMind-V 项目应自行实现一个小而明确的 `feedback_to_recipe.py`，先支持按分箱缺口调权重，不要声称已经复现了完整的模型反馈算法。

### 2.6 Data pool manipulator：把“配比”变成可操作对象

`data_pool_manipulators.py` 提供的操作包括：

- `DataPoolConstruction`：依据统计字段排序并切分数据池；
- `DataPoolCombination`：组合不同排名层级的数据池；
- `DataPoolDuplication`：复制某个数据池若干次，等价于提高曝光量；
- `DataPoolDownsampling`：按固定 seed 下采样到目标规模；
- `DataPoolMerging`：合并多个池；
- `DataPoolRanking`：根据评估指标排序数据池；
- `DataPoolCartesianJoin`：对两个数据池集合做笛卡尔组合。

这里的工程经验是：数据“比例”可以通过复制、下采样、排序后取 top-k 或组合池来实现，而不一定要在训练 DataLoader 中埋一个不可追踪的随机权重。对 MiniMind-V，建议把最终采样物化成 `train_manifest.parquet`，每行记录 `sample_id`、`bucket`、`sampling_weight`、`recipe_version`，从而知道实际训练曝光了什么。

### 2.7 Evaluator、Factory 和环境管理

`factories.py` 将数据执行器、分析器、评估器、模型训练器、模型推理器解耦，通过 `type` 选择实现。`evaluators.py` 提供 Accuracy、MSE、GPT-3 quality、Inception 等适配器。`env_manager.py` 支持 conda/venv 环境创建、依赖安装和命令执行。

对于本项目，建议只采用其中两层：

- 自己定义稳定的 `Evaluator` 接口，输出整体指标和分箱指标；
- 保留 MiniMind-V 与 Qwen-LoRA 的训练脚本为外部 executor。

暂时不要把所有环境安装、模型推理后端都包装进一个大工厂，否则数据工程实验会被部署复杂度拖慢。

## 3. Open-Instruct：后训练 recipe、数据混合和反馈数据

### 3.1 顶层结构

```text
third_party/open-instruct/
├── open_instruct/
│   ├── finetune.py
│   ├── dpo.py
│   ├── grpo.py
│   ├── reward_modeling.py
│   ├── dataset_processor.py
│   ├── dataset_transformation.py
│   ├── data_loader.py
│   ├── mix_data.py
│   ├── mix_data_preferences.py
│   ├── rejection_sampling/
│   ├── rubrics/
│   ├── environments/
│   ├── deepspeed/accelerate/launch utilities
│   └── tests/
├── configs/
│   ├── train_configs/sft/
│   ├── train_configs/dpo/
│   ├── train_configs/tulu3/
│   ├── ds_configs/
│   ├── beaker_configs/
│   └── judge_configs/
├── scripts/
│   ├── data/
│   ├── train/
│   ├── eval/
│   ├── rejection_sampling_tulu.bash
│   └── finetune_*.sh
├── decontamination/
├── docs/
│   ├── tulu3.md
│   ├── data/preference-data.md
│   └── algorithms/
├── tests/
└── pyproject.toml
```

### 3.2 数据 mixture：先物化，再训练

`open_instruct/mix_data.py` 的核心逻辑是读取 `dataset_mixer`，通过 `get_datasets` 合并训练集，并保留 dataset id/source 等字段，再把混合结果保存到 `dataset_mix_dir`。Tülu 3 配置明确记录：

- `dataset_mixer`；
- `dataset_mix_dir`；
- `seed`；
- `max_seq_length`；
- effective batch size；
- learning rate、warmup、epoch；
- checkpointing 方式。

这比只在命令行写一个“数据比例”更可靠，因为实际使用的混合数据可以落盘、复查和复现。MiniMind-V 应照搬这个模式：每个 Round 先生成不可变的 manifest，再启动训练。

### 3.3 dataset processor：训练前的数据契约

`dataset_processor.py` 把 SFT 和偏好数据处理区分开：

- SFT 使用 `messages`，经 chat template 变成 `input_ids`、`attention_mask`、`labels`；
- Preference/DPO 使用 `chosen`、`rejected`，分别生成 prompt、chosen、rejected 的 token 序列；
- 提供最大 token 长度和最大 prompt 长度过滤；
- 支持多进程 map、缓存和 sanity check；
- 记录 `ground_truth`、`dataset`/source 等字段供 verifier 或评估使用。

可借鉴的重点是：原始数据格式、tokenized cache、训练 batch 不应混为一个文件。对 VQA 建议保留：

```text
raw_samples.parquet
tokenized_cache/<model>/<recipe_version>/
train_manifest.parquet
```

### 3.4 DataLoader：sample identity、shuffle 和 checkpoint

`HFDataLoader` 要求数据集带有 `index` 列，并负责分布式 sharding、seed、reshuffle、batch 进度和 checkpoint。这个要求直接对应你前面关心的“Round 1 后如何重新细分箱”：

- `sample_id` 必须跨 round 稳定；
- `index`/行号不能代替 sample_id；
- 每轮的 shuffle seed、采样清单和实际训练步数都要记录；
- 断点恢复不能只恢复模型，还要恢复 data loader 状态。

MiniMind-V 目前的训练脚本具备断点保存，但没有成熟的 val 数据流；数据工程层应先补齐逐样本验证输出，再考虑复杂的自动恢复。

### 3.5 Tülu 3 的 SFT 与 DPO 组织方式

本地 `docs/tulu3.md` 展示的顺序是：

```text
基础模型 → SFT → DPO
```

SFT 示例采用固定 dataset mixture、2 epochs、每 epoch checkpoint；DPO 从 SFT 模型开始，使用 preference mixture、较低学习率和 1 epoch。对于你的项目，这提供了一个清晰的最小后训练路径：

```text
Projector warmup → MiniMind-V SFT → 可选 preference/DPO
```

但 DPO 只有在 chosen/rejected 的质量可控时才值得加入。对于 CLEVR，优先用自动答案验证构造 preference pair；不要仅凭语言模型自评生成“偏好”而不检查图片和标准答案。

### 3.6 Rejection sampling：分片、生成、评分、落盘

本地 `scripts/rejection_sampling_tulu.bash` 和 `open_instruct/rejection_sampling/` 展示了较完整的工程流程：

1. 把 prompt 切成多个 shard；
2. 生成多个 completion；
3. 保存生成结果；
4. 使用 reward model 打分；
5. 保存筛选后的 completion 和 score；
6. 汇总到可供下一阶段训练的数据集。

这里最值得借鉴的是“生成结果”和“评分结果”分开保存，且以 shard 为单位可重跑。MiniMind-V 的版本可以是：

```text
teacher_or_student_generations.parquet
candidate_scores.parquet
accepted_sft.parquet
preference_pairs.parquet
```

### 3.7 去污染和评估

Open-Instruct 包含独立的 `decontamination/` 工具、多个 benchmark 脚本和评估结果收集脚本。对当前项目的对应要求是：

- 按 image_id/scene_id 切分，避免同图多问跨 split；
- 用问题文本和图像身份做泄漏检查；
- 每个 Round 保存 train/val/test manifest；
- test 只在 recipe 和 checkpoint 确定后运行。

## 4. data_engineering_book：从网页参考的工程思想

本次不下载该书源码，只参考其 GitHub 公开页面。它更像一套数据工程课程/书籍，而不是 VLM 专用训练框架。对本项目有用的不是照搬章节代码，而是把数据任务当成一条有契约、有测试、有可观察性的生产管线。

### 4.1 可借鉴的主题层次

根据公开目录信息，内容覆盖数据工程基础、数据采集与存储、批处理/流处理、数据质量、工作流与工程化实践，并配有项目型练习。映射到 VLM 数据工程时，应关注：

```text
原始数据接入
→ 标准化 schema
→ 数据质量检查
→ 变换与特征/标签生成
→ 数据版本与 lineage
→ 可重复的训练输入
→ 监控与回归评估
```

### 4.2 对 MiniMind-V 最有价值的工程化原则

1. **原始层只读**：原始 CLEVR/GQA 样本不被覆盖，所有清洗和标签都生成新版本。
2. **契约优先**：每个中间产物定义必需列、类型、空值规则和唯一键。
3. **幂等处理**：同一输入和同一 recipe version 重跑应得到同样的输出或可解释差异。
4. **数据质量可观测**：记录过滤量、重复率、空值、答案解析失败率、各分箱数量。
5. **任务与数据解耦**：数据处理 recipe 不应该把训练命令和模型结构硬编码在一起。
6. **失败可回溯**：每个被过滤或被拒绝的样本记录 reason，而不是简单丢弃。
7. **小规模先验证**：先用可控子集验证 schema、处理逻辑和训练闭环，再放大规模。

### 4.3 不能直接照搬的地方

通用数据工程书的 ETL、仓库、流处理和编排思想不能直接替代 VLM 的视觉指标。对于你的任务，还必须额外记录：视觉必要性、图像变换、scene complexity、reasoning hops、模型预测和错误类型。

## 5. 三个项目共同体现的工程模式

### 5.1 数据不是一个文件，而是一组有血缘的产物

推荐至少保留：

```text
raw samples
  → normalized samples
  → static tags
  → recipe/manifest
  → tokenized cache
  → model predictions
  → bucket metrics
  → next recipe
```

### 5.2 配方和训练必须解耦

同一个 `round2.yaml` 可以生成 MiniMind-V manifest，也可以生成 Qwen-LoRA manifest；模型训练脚本只消费 manifest，不负责解释“为什么这个样本被选中”。

### 5.3 反馈必须落到可执行的配置更新

反馈不是一句“计数能力较差”，而是：

```text
bucket=count_4_6_objects
val_accuracy=0.62
target=0.75
next_weight=old_weight × 1.4
reason=learnable_gap
```

### 5.4 数据池操作要支持复制、下采样、排名和合并

这比只支持“按比例随机采样”更适合课程学习和失败回流，因为你可以明确表达 replay、hard pool、quality pool 和 real/synthetic pool。

### 5.5 逐样本记录是反馈闭环的最小充分条件

只有总体 accuracy 时，无法在 Round 1 后发现“3-hop front/behind”这一细分缺口。至少保存：

```text
sample_id, image_id, gold, prediction, correct, loss,
confidence, checkpoint, round, error_type
```

## 6. 直接映射到当前项目的模块清单

| 参考项目经验 | 当前项目落地模块 |
| --- | --- |
| Sandbox 四类 job | `prepare → build recipe → train → evaluate` 四类脚本/模块 |
| Sandbox ContextInfos | `runs/<round>/run_manifest.json` 和指标快照 |
| Data pool ranking/duplication | `build_train_manifest.py` 的 bucket weighting/replay |
| Open-Instruct dataset mixer | `recipes/*.yaml` + 物化 Parquet manifest |
| Open-Instruct sample index | 稳定 `sample_id` + `image_id`/`scene_id` |
| Open-Instruct rejection sampling | `generate_candidates → score_candidates → accept` 三步 |
| decontamination | 按图片/场景切分和跨 split 检查 |
| data engineering book 的数据契约 | schema validator、质量报告、血缘和版本目录 |
| MiniMind-V 冻结策略 | Projector warmup → `freeze_llm=1` SFT |

## 7. 最终借鉴结论

最适合当前项目的组合不是完整安装三个框架，而是采用它们的边界清晰的经验：

```text
Data-Juicer Sandbox：借鉴反馈闭环和 data-pool 操作
Open-Instruct：借鉴 mixture 物化、sample identity、SFT/DPO/RFT 分层
data_engineering_book：借鉴数据契约、质量、血缘、幂等和可观察性
MiniMind-V：保留现有轻量模型和冻结策略
```

研究实现上，先完成“可追踪、可复现、可回放”的数据框架，再加入自适应配比；不要一开始就做复杂 HPO、全自动模型反馈或通用数据平台。

## 8. v0.2 对当前项目的落地边界

### 8.1 主线 A 与主线 B 的边界

| 维度 | 主线 A | 主线 B |
| --- | --- | --- |
| 主要问题 | 课程、分箱、数据飞轮是否有效 | 真实数据清洗、混编比例和能力退化 |
| 首选数据 | CLEVR，必要时加入已清洗 bucket | CLEVR + `gqa_aligned_v1` |
| 反馈对象 | bucket 权重、课程阶段、hard/replay 池 | source ratio、quality tier、通用能力保持 |
| 主要对照 | Uniform、static、dynamic curriculum | S100R0、S75R25、S50R50、S0R100 |
| 禁止混淆 | 不把真实数据噪声当作课程收益 | 不把训练预算变化当作比例收益 |

### 8.2 GQA 清洗的项目级验收项

GQA 进入 B 线前必须产出以下文件，而不是只产出一个清洗后的 JSON：

```text
silver_samples.parquet
quality_summary.json
duplicate_clusters.parquet
conflict_cases.parquet
leakage_report.json
quarantine.parquet
gold_manifest.parquet
```

`quality_summary.json` 至少报告每步输入量、输出量、过滤率和原因分布；`quarantine.parquet` 保留原始字段和规则版本；`gold_manifest.parquet` 才是训练/评估可消费的稳定输入。

### 8.3 通用能力退化不是附加日志

Open-Instruct 的去污染、固定数据 mixture 和独立评估目录提示：专项训练后的通用能力必须有独立 replay 集和固定基线。当前项目应把 `general_capability/` 作为每个 B 线 run 的一级产物，至少包括 text-only loss/perplexity、短指令格式遵循和 pre/post delta。它不能从 VQA val 的 accuracy 推断出来。

### 8.4 湖仓语义的最小可行版本

当前不需要部署 Spark、Ray、Iceberg 或 Delta。先用不可变 Parquet 分区实现：

```text
raw → bronze → silver → gold → serving manifest
```

同时记录 schema、recipe、代码 commit、上游 artifact 和质量报告。预测事件采用 append-only JSONL/Parquet，反馈聚合采用批处理；未来扩展到分布式平台时只替换存储和执行器，不改变字段契约。

### 8.5 当前本地代码与目标结构的关系

已有 `split_by_image_three_way.py`、`build_initial_recipe.py`、`annotate_manifest.py`、CLEVR 生成/分析脚本和 Projector/`freeze_llm=1` 训练脚本，属于目标结构中的 split、recipes、feedback/evaluate 和 training 适配层。新增 GQA ingestion/cleaning、exposure report、text replay 和 recipe feedback 时，应保持这些脚本的输入输出兼容，不覆盖用户现有改动。

## 9. v0.3：按完整数据链路补齐的项目映射

用户提供的工程指导把链路明确为：数据设计 → 采集 → 接入 → 解析 → 清洗 → 标注/合成 → 质量评估 → 划分与版本化 → 格式转换 → 训练/评测发布 → 训练反馈 → 数据迭代。对当前项目的最终映射如下：

| 链路阶段 | 当前项目实现/规划 | 关键验收产物 |
| --- | --- | --- |
| 数据设计 | `data_contract/`、数据卡、任务本体、质量门禁 | schema、字段约束、标签规范 |
| 采集/接入 | CLEVR/GQA ingestion、来源登记 | source metadata、license、sha256、样本数 |
| 解析 | JSON/scene graph/image index 解析 | bronze parquet、解析失败清单 |
| 清洗 | 图像完整性、答案归一化、去重、冲突、泄漏 | silver parquet、quarantine、reason 分布 |
| 标注/合成 | CLEVR 程序真值、GQA 题型标签、可选教师候选 | static tags、candidate scores |
| 质量评估 | 规则→模型/执行器→人工抽检 | quality_summary、gold/review/rejected |
| 划分/版本 | image/scene-level split、manifest、lineage | split manifest、dataset card、run manifest |
| 格式转换 | JSONL/Parquet/tokenized cache/serving manifest | schema-compatible train input |
| 发布 | train/val/test gold manifest、独立 text replay | release checklist、exposure report |
| 训练反馈 | bucket metrics、learning dynamics、retention | predictions、metrics、curriculum state |
| 数据迭代 | A 线 recipe feedback；B 线清洗修复和 mixture | next recipe、tags_v2、new gold version |

### 9.1 需要新增而不是遗漏的四类工程能力

1. **接入元数据**：来源、许可证、校验和、文件/样本数、处理状态，不只保存最终 JSON。
2. **质量门禁**：没有通过 schema、完整性、模态匹配、分布和泄漏检查的 gold 不得训练。
3. **幂等与恢复**：按 shard 运行，临时文件原子提交，记录状态，失败后可重跑而不重复写入。
4. **数据生产闭环**：教师生成、自动标注或合成结果必须经过独立规则/执行器/抽检，保存 candidate、score、accepted 和 rejection reason，防止错误闭环。

### 9.2 A/B 的最终边界

```text
Shared P0：数据链路、schema、质量、版本、发布和 lineage
A：CLEVR 上的分箱、难度、课程、学习动态、任务影响度、hard/replay、反馈配方
B：GQA 清洗去重、质量层、CLEVR/GQA 混编比例、静态/阶段混编、通用能力退化
Optional：rejection sampling、蒸馏、DPO、流式事件、Ray/Spark 扩展
```

A/B 是并列研究分支；可以共享平台并行开发，但实验变量和结果表必须分开。

## 10. 扩展开源项目的借鉴边界

本地三个项目仍是主参考：Sandbox 负责反馈闭环和 data-pool，Open-Instruct 负责 mixture/去污染/SFT-DPO-RFT 分层，MiniMind-V 负责轻量 VLM 冻结策略。规划层额外吸收：

- **LLaVA** 的 projector 对齐→视觉指令调优阶段边界；
- **LLaMA-Factory** 的数据集注册、metadata、预处理缓存与训练配置分离；
- **NeMo Curator** 的 filter/dedup/transform 分层和面向分布式执行器的接口思想。

当前不下载或拼接这些项目的全部代码，只将它们的边界映射为本项目的 `ingestion/cleaning/recipes/training/evaluate` 模块。单机阶段用 Python、Parquet、JSONL 和稳定 manifest；未来再替换 Ray/Spark/Iceberg/Delta，不改变 schema、lineage 和质量报告接口。

## 11. 最终确认的实施顺序

```text
P0 Shared：设计/接入/解析/清洗/质量门禁/划分/版本/发布
P1 A：CLEVR 静态分箱 → Projector/SFT → val → 学习动态 → 反馈配方
P1 B：GQA ingestion → 清洗去重 → gold → S100R0/S75R25/S50R50/R100
P2 Shared：text replay、visual stress、长尾/覆盖/曝光/倾斜报告
P2 A：课程和 hard/replay 消融
P2 B：gold/silver 清洗消融与通用能力退化
P3 Optional：教师生成、rejection sampling、DPO、流式/分布式扩展
```
