# MiniMind-V 数据工程与双主线实验框架

> 版本：v0.3 final project design；日期：2026-09-13

> 执行入口：本文负责说明研究框架和边界；需要按命令、状态、产物和门禁落地时，先阅读 [MINIMIND_V_EXECUTION_PLAYBOOK.md](MINIMIND_V_EXECUTION_PLAYBOOK.md)。

## 1. 两条并列主线

主线 A 验证课程学习、数据飞轮、数据分箱和指标反馈：

```text
静态标签 → 配方 → 训练 → 逐样本 val → 学习动态 → 细分箱 → 新配方
```

主线 B 验证真实数据清洗、CLEVR/真实 VQA 混编和通用能力保持：

```text
真实数据 raw → 清洗去重 → 质量分层 → 混编配方 → VQA/通用能力评估
```

A、B 共享 schema、sample_id、lineage 和评估基础设施，但不互相承接，也不把课程收益与真实数据比例收益写成一个结论。

## 2. 真实数据集：GQA Balanced 结构化子集

首选 GQA Balanced，而不是直接使用全量开放式 VQA。GQA 提供 image-level scene graph、对象属性/关系、问题类型、短答案、`isBalanced` 和语义操作链，适合映射到 CLEVR 的 attribute、count、existence、spatial、relation 能力桶。第一版构建 `gqa_aligned_v1`，仅保留有图像、答案可归一化且可稳定映射的题型；外部知识依赖强或开放答案样本进入 `review`，不静默丢弃。

GQA 处理流程：

```text
raw JSON/图片
  → bronze：解析、路径映射、schema、坏图隔离
  → silver：文本/答案标准化、去重、冲突和泄漏检查
  → gold：质量分层、image-level split、能力标签、训练 manifest
  → quarantine：原记录、rejection_reason、rule_version
```

默认去重键为 `image_id + normalized_question`；问题文本全局重复不能直接删除，因为同一问题在不同图片上可能有效。近重复使用图像 hash、文本 MinHash/embedding 生成报告，自动过滤阈值需在 val 上冻结。

## 3. 项目级代码与数据目录

```text
VLM_learning/
├── configs/
│   ├── data_contracts/
│   ├── datasets/clevr.yaml
│   ├── datasets/gqa_aligned_v1.yaml
│   ├── recipes/line_a_static.yaml
│   ├── recipes/line_a_curriculum_r1.yaml
│   ├── recipes/line_a_curriculum_r2.yaml
│   ├── recipes/line_b_s100r0.yaml
│   ├── recipes/line_b_s75r25.yaml
│   ├── recipes/line_b_s50r50.yaml
│   └── recipes/line_b_r100.yaml
├── data/
│   ├── lake/
│   │   ├── raw/clevr/
│   │   ├── raw/gqa/
│   │   ├── bronze/
│   │   ├── silver/
│   │   ├── gold/
│   │   └── quarantine/
│   ├── manifests/
│   ├── tags/
│   ├── predictions/
│   ├── metrics/
│   ├── tokenized_cache/
│   └── generated/
├── recipes/
├── runs/
├── reports/
├── scripts/data/ scripts/line_a/ scripts/line_b/ scripts/eval/
├── src/minimind_v_lab/
│   ├── data_contract/{schemas.py,validate_raw.py,validate_curated.py,quality_report.py}
│   ├── ingestion/{ingest_clevr.py,ingest_gqa.py,normalize_conversations.py,partition_writer.py}
│   ├── cleaning/{image_integrity.py,text_quality.py,answer_normalization.py,deduplicate_questions.py,deduplicate_images.py,conflict_detection.py,filter_policy.py}
│   ├── split/{split_by_image.py,split_by_scene.py,leakage_check.py}
│   ├── tagging/{static_tags.py,clevr_program_tags.py,gqa_question_tags.py,difficulty_tags.py,dynamic_tags.py}
│   ├── recipes/{recipe_schema.py,build_manifest.py,sample_buckets.py,handle_imbalance.py,materialize_recipe.py}
│   ├── curriculum/{stage_scheduler.py,difficulty_estimator.py,learning_dynamics.py,replay_buffer.py,bucket_controller.py,curriculum_state.py}
│   ├── feedback/{collect_predictions.py,aggregate_bucket_metrics.py,classify_errors.py,task_impact.py,visual_dependency.py,update_recipe.py,round_state.py}
│   ├── mixing/{build_source_mixture.py,synthetic_real_ratio.py,source_balance.py,exposure_report.py}
│   ├── evaluate/{vqa_metrics.py,bucket_metrics.py,general_capability.py,text_replay_eval.py,visual_dependency_eval.py,leakage_eval.py,report.py}
│   ├── training/{minimind_runner.py,qwen_lora_runner.py,checkpoint_registry.py,train_manifest_adapter.py}
│   └── lineage/{run_manifest.py,artifact_registry.py,config_snapshot.py,metrics_store.py}
└── third_party/{data-juicer-sandbox/,open-instruct/,minimind-v/}
```

## 4. Lakehouse 风格的最小实现

本地单机阶段用 Parquet/JSONL、目录分区和不可变 manifest 实现核心语义，不引入集群。推荐分区字段：`source=clevr|gqa / split=train|val|test / task_coarse=... / version=...`。分片使用固定 `shard-00000.parquet`，每个产物写入 `schema_version`、`recipe_version`、`git_commit`、`created_at` 和上游 artifact hash。预测事件追加到 `data/predictions/events/`，批处理聚合为下一版 recipe；未来可将同一接口迁移到 Spark/Ray + Iceberg/Delta。每轮报告 source、task、difficulty、shard 的行数和曝光量，检测数据倾斜并显式 rebalance。

## 5. 数据契约

每条样本至少包含：

```text
sample_id, image_id, scene_id, image_path, question, answer,
source, split, conversation, quality_status, quality_tier
```

`static_v1` 只保存无需模型运行即可得出的任务、答案类型、问题长度、对象数、reasoning hops、source 和质量信息。Round 1 后才追加 `difficulty_v2`、`distractor_level`、`visual_dependency_prior` 等细标签。

`dynamic_roundN` 追加：`checkpoint, prediction, normalized_prediction, gold_answer, correct, loss, confidence, error_type, learning_status, visual_dependency`。动态标签只追加版本，不覆盖旧标签；训练 manifest 另存 `recipe_version`、`bucket`、`sampling_weight`、`repeat_count` 和 `selected`。

## 6. 主线 A：复杂课程学习与数据飞轮

独立记录质量 Q、视觉必要性 V、可学习难度 D、信息量 I、独特性 U、代表性 R、任务影响度 T 和噪声 N。排序分数在 recipe 中显式写出：

```text
priority = wq*Q + wv*V + wd*learnable_difficulty
         + wi*I + wu*U + wr*R + wt*T - wn*N
```

课程阶段：

```text
Stage 0：高质量、视觉必要、低难度，建立图像—答案对齐
Stage 1：低/中难度，按 task bucket 平衡
Stage 2：提高 reasoning hops、物体数和干扰物
Stage 3：选择经常错但 loss 正在下降的 learnable-hard
Stage 4：hard + clean anchor replay + 反事实视觉样本
```

阶段转移同时检查 bucket accuracy、最近窗口 loss 斜率、正确率、视觉必要性、text-only replay 退化和预算，不能只按 epoch。保留 `loss_t`、`correct_t`、`first_correct_step`、`forgetting_count` 和 `prediction_stability`，将样本分为 `easy_mastered`、`learnable_hard`、`unlearnable_or_noisy`。

Round 1 只用粗标签。若 `spatial` 低于目标且样本量足够，才追加 `relation_type × reasoning_hops × distractor_level`；通过 `sample_id` join 旧预测，生成 `tags_v2`。配方更新：

```text
deficit = max(0, target_accuracy - bucket_accuracy)
new_weight = clip(old_weight * (1 + alpha * deficit),
                  0.5 * old_weight, 2.0 * old_weight)
```

对照至少包括 Uniform、Static balanced、Easy-to-hard、Learning-dynamics、Learning-dynamics + hard/replay feedback；固定模型、冻结策略、总 steps 和评估集。

## 7. 主线 B：清洗、混编和通用能力保持

GQA 清洗规则依次包括 JSON/schema、图像解码和尺寸、Unicode/模板/长度、答案归一化、题型映射、完全/近重复、同图同问冲突、image-level 泄漏、质量分层和 manifest 物化。保留 `accepted / review / rejected` 及每步计数。训练默认使用 `gold`，必要时以 recipe 显式加入 `silver`。

比例实验固定总 optimizer steps、token budget、batch size、seed 和 checkpoint 频率，只改变曝光比例：`S100R0, S90R10, S75R25, S50R50, S25R75, S0R100`。首轮建议 `S100R0 / S75R25 / S50R50 / S0R100`。另测阶段式混编 `80/20 → 60/40 → 50/50`，但与静态方案保持相同总 steps。

通用能力退化至少评估：固定 text-only replay 的 loss/perplexity 或 exact match；短文本 instruction-following 的格式和任务正确率；基础、Projector-only、SFT、混编 checkpoint 的 pre/post 差值；真图、黑图、错配图的视觉依赖 stress test。

```text
retention_delta = post_metric - base_metric
forgetting_rate = (base_metric - post_metric) / max(base_metric, eps)
```

若 MiniMind-V 接口必须带图像占位符，应单独标记为“多模态接口下文本回放”，不能冒充纯语言能力评估。B 线每组至少报告 CLEVR IID、CLEVR compositional-OOD、GQA aligned test、source×task bucket、visual stress、text replay、过滤率和训练成本。

## 8. 训练、验证和 lineage

保留 MiniMind-V 的 Vision Encoder 冻结策略：Projector-only warmup 后，从该 checkpoint 进行 `freeze_llm=1` SFT。所有 run 保存基础/Projector/SFT checkpoint、manifest、总 steps、seed、recipe hash 和代码 commit。val 用于选阶段、权重和最佳 checkpoint；recipe 冻结后才运行 test，test 结果不得回流训练。

```text
runs/line_a/r1/{run_manifest.json,recipe.yaml,train_manifest.parquet,
  val_predictions.parquet,metrics_overall.json,metrics_by_bucket.json,
  curriculum_state.json,checkpoint/}
runs/line_b/s75r25/{run_manifest.json,source_exposure.json,
  train_manifest.parquet,val_predictions/,general_capability/,metrics/,checkpoint/}
```

## 9. 从八股链路补齐的共享 P0 数据平台

两条研究主线之前，先建立一条不依赖具体模型的共享数据链路：

```text
数据设计
→ 数据采集/接入
→ 解析与登记
→ 清洗与预处理
→ 标注/程序合成
→ 质量评估与门禁
→ 划分与版本化
→ 训练/评测格式转换
→ 发布 manifest
→ 训练与评测
→ 模型反馈
→ 数据迭代
```

### 9.1 数据设计与接入登记

在采集前冻结任务定义、字段 schema、质量标准、采样范围、标注规范和验收指标。每个来源登记：`source_uri`、license、采集时间、文件大小、sha256、预期样本数、实际样本数、解析状态、处理代码版本和责任人/运行 ID。下载成功不等于接入成功，必须再做可解析性、样本数和内容完整性检查。

### 9.2 解析、预处理和模态对齐

解析只负责把输入读成结构化字段；清洗负责判断内容是否有效。图像索引和媒体文件分离保存，样本记录图片路径、尺寸、文件大小、hash、解码状态和来源版本。训练前检查 image-question-answer 是否属于同一个 `image_id`，并检查图片替换、路径失效、尺寸异常、颜色通道和黑图/空图。

### 9.3 标注规范和自动质检

CLEVR 优先使用程序真值；GQA 使用官方答案/scene graph 再做归一化；教师模型只用于表达变体、解释或候选答案，不能成为唯一真值。标注规范要定义边界案例、冲突处理和拒标条件。质量控制采用三层：规则硬过滤 → 模型/执行器质检 → 阈值附近、长尾、冲突和高影响样本人工抽检。

### 9.4 质量门禁与发布验收

每个阶段输出 `quality_summary.json`，至少包含解析率、字段完整率、图片解码率、图文匹配率、重复率、冲突率、答案归一化失败率、各 bucket 覆盖率和过滤原因。发布前必须通过 schema、完整性、分布、模态匹配、泄漏、版本和复现检查；不满足门禁的产物不能进入 gold manifest。

### 9.5 版本、血缘、幂等和容错

每个 artifact 由 `dataset_version + recipe_version + code_commit + environment_lock + seed + upstream_hash` 唯一确定。处理任务按 shard 执行，采用临时文件写入成功后原子改名，记录 `pending/running/succeeded/failed` 状态，失败可重试且重复执行不产生重复样本。保留 `run_manifest.json`、输入输出统计、日志和失败分片，支持断点续跑和回放。

### 9.6 发布格式

原始交换层保留 JSONL/原始图片；silver/gold 分析层使用 Parquet；训练层使用稳定的 serving manifest 和 tokenized cache。不要把图片二进制全部塞进可读 JSONL；索引负责筛选，媒体文件负责读取。大规模扩展时增加 compaction，避免大量小文件。

## 10. A/B 的最终任务划分

| 层次 | 任务 | 数据/输出 | 是否形成研究结论 |
| --- | --- | --- | --- |
| Shared P0 | schema、接入、清洗基础、质量门禁、split、版本、lineage、发布 | raw/bronze/silver/gold、manifest、quality report | 是工程可复现性，不作为 A/B 的效果变量 |
| A | 分箱、难度建模、课程阶段、学习动态、hard/replay、任务影响度、反馈配方 | CLEVR tags、curriculum state、round recipes | 课程/数据飞轮是否有效 |
| B | GQA 清洗去重、质量分层、CLEVR/GQA 比例、静态/阶段混编、通用能力保持 | GQA gold、mixture manifest、retention report | 真实数据和混编是否有效、是否造成退化 |
| Optional | rejection sampling、教师蒸馏、自动标注、DPO、流式事件、分布式执行 | candidates、scores、accepted、preference pairs | 仅在基础链路稳定后单独验证 |

A 和 B 共享平台，但不互相承接。实现上可以先完成 Shared P0，再并行推进 A/B；实验报告中分别维护假设、变量、对照和结论。

## 11. A 线补充后的完整实验包

除原有课程阶段外，A 线还加入：

- **覆盖与长尾**：按 relation、answer type、reasoning hops、object count、distractor 分层，报告 coverage、tail recall 和曝光倾斜；
- **任务影响度**：对 bucket 做小规模 leave-one-bucket-out 或 reweight 对照，估计对目标指标的边际贡献；
- **信息量/独特性/代表性**：用问题长度、答案熵、模型分歧、近重复距离和验证集覆盖作为独立列，不直接把高 loss 当高价值；
- **难例和噪声分离**：高 loss 样本必须结合标签可信度、历史稳定性和人工抽检分为 learnable-hard 或 noisy；
- **动态阶段转移**：由 bucket accuracy、loss slope、replay retention 和预算共同决定，不只按 epoch；
- **消融**：Uniform、静态分层、easy-to-hard、learning-dynamics、learning-dynamics+replay，固定 optimizer steps 和 checkpoint 预算。

## 12. B 线补充后的完整实验包

除 GQA 清洗和比例实验外，B 线还加入：

- **数据卡/许可证/风险记录**：真实数据来源、适用范围、偏差和限制随 gold 版本发布；
- **质量层消融**：gold-only、gold+silver、未去重对照，观察清洗收益和误杀风险；
- **静态比例与动态比例**：固定总 tokens/steps，比较 S100R0、S75R25、S50R50、R100，以及阶段式 80/20→60/40→50/50；
- **分布对齐报告**：比较 CLEVR/GQA 的问题长度、答案类型、难度、图像统计、重复率和 task bucket 覆盖；
- **通用能力保持**：text-only replay、格式遵循、短文本任务、图像错配/黑图 stress test；
- **真实数据错误回流**：bad case 进入 quarantine 或 review，不直接回流到 train；只有修复并重新验收后才能进入下一版 gold。

## 13. 从开源项目抽取的组织形式

除本地已审阅的 Data-Juicer Sandbox、Open-Instruct、MiniMind-V 外，最终方案吸收以下公开项目的组织思想：

| 项目 | 借鉴点 | 本项目落地 |
| --- | --- | --- |
| LLaVA | projector feature alignment → multimodal instruction tuning；数据源和图片目录分离；固定 mixture 和独立评测 | Projector warmup、`freeze_llm=1`、source manifest、评测矩阵 |
| LLaMA-Factory | 数据集注册表、统一 dataset metadata、预处理缓存和训练配置分离 | `datasets/*.yaml`、schema registry、tokenized cache |
| NeMo Curator | 大规模数据的 filter/dedup/transform 分层和 GPU/分布式扩展接口 | 本地规则/Parquet 实现，保留未来 Ray/Spark 执行器接口 |
| Data-Juicer Sandbox | probe → recipe refine → process/train → evaluate 的迭代闭环 | `round_state`、`quality_report`、`feedback_to_recipe` |
| Open-Instruct | mixture 物化、stable index、decontamination、SFT/RFT/DPO 分层 | manifest、泄漏检查、candidate/score/accepted 分层 |
| data_engineering_book | 数据契约、质量门禁、血缘、幂等、批流和故障恢复 | raw/bronze/silver/gold、run manifest、checkpoint/resume |

这些项目不是要被直接拼成一个巨型平台，而是提供模块边界和验收标准。当前规模使用 Python + Parquet + JSONL；当数据量、并行度或媒体规模达到瓶颈时，再替换为 Ray Data、Spark、Iceberg/Delta 或对象存储。

## 14. 最终确认版实施顺序

```text
P0 Shared：数据设计、接入登记、schema、raw/bronze/silver/gold、质量门禁、image-level split、版本/血缘、manifest 发布
P1 A：CLEVR 静态分箱 → Projector/SFT → 逐样本 val → 学习动态 → 细分箱 → feedback recipe
P1 B：GQA ingestion → 清洗/去重/冲突/泄漏 → gold manifest → S100R0/S75R25/S50R50/R100
P2 Shared：text-only replay、visual stress、质量/覆盖/长尾报告、数据倾斜和曝光报告
P2 A：课程消融、任务影响度、hard/replay、阶段式混编独立实验
P2 B：gold/silver 清洗消融、静态/动态混编、通用能力退化分析
P3 Optional：教师生成、rejection sampling、自动偏好对、DPO、流式事件和分布式扩展
```

每个 P 阶段都必须能独立回放；下一阶段不覆盖上一阶段产物。研究结论只来自固定模型、固定训练预算、固定评测集下的数据因素对照，不把工程 smoke test 当作性能结果。

## 15. 实施优先级

优先实现：`validate_dataset_contract.py`、`ingest_gqa.py + clean_gqa.py`、`build_train_manifest.py`、`aggregate_bucket_metrics.py`、`feedback_to_recipe.py`、`text_replay_eval.py`、`source_exposure_report.py`。待清洗、自动答案验证和 chosen/rejected 审计稳定后，再做 rejection sampling 和小规模 DPO。
