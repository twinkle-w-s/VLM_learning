# MiniMind-V 数据链路与数据飞轮可执行手册

> 版本：v0.7 execution specification  
> 日期：2026-09-14  
> 适用范围：VLM_learning、MiniMind-V、CLEVR、GQA Balanced 结构化真实 VQA 子集。

本文把总体规划落到可执行机制：每个阶段说明输入、处理、输出、字段、质量门禁、失败策略和恢复方式。目录中的模块如果尚未实现，统一标记为 planned，不把规划目录当作已完成功能。

建议阅读顺序：先读第 2～5 节理解统一作业和数据契约，再读第 15～17 节理解数据构建与飞轮，最后按第 22～23 节实施。第 24 节以后是面向编码时的接口、配置、增量构建、训练适配和验收附录。

## 0. 执行目标和边界

最终要得到的不是一个“清洗脚本”，而是一套可以反复运行的实验数据系统：

~~~text
原始数据可回溯
中间产物可检查
训练输入可复现
配方变化可解释
训练/评估可恢复
模型反馈可回流
test 不进入反馈
~~~

当前规模采用 Python + JSONL + Parquet + 本地目录实现；未来可替换为 Ray Data、Spark、Iceberg、Delta 或对象存储，但不改变 sample schema、artifact manifest 和状态契约。本文中的 `planned` 只表示接口和执行约定已经确定、代码尚未补齐；不能把 planned 当成已经跑通的实验结果。

两条主线共享 P0 数据平台，但研究变量分开：

~~~text
Shared P0：接入、解析、清洗、质量、版本、发布、lineage
A：CLEVR 课程学习、分箱、学习动态、hard/replay、反馈 recipe
B：GQA 清洗去重、CLEVR/GQA 混编、通用能力退化
Optional：rejection sampling、蒸馏、DPO、分布式扩展
~~~

## 1. 当前代码审计

| 当前入口 | 已有能力 | 必须补的项目机制 |
| --- | --- | --- |
| pipeline/prepare_full_clevr.py | 串联 CLEVR 生成、标注、切分、recipe、Parquet、校验 | state、hash、resume、原子提交、gate report |
| manifest/annotate_manifest.py | task、answer、difficulty 粗标签 | sample_id、tag_version、label_source、独立 tags |
| manifest/split_by_image_three_way.py | image-level split | assignment、overlap、coverage、版本 |
| manifest/build_initial_recipe.py | task quota 和重复采样 | YAML 配方、exposure_id、repeat 语义、曝光统计 |
| convert/convert_manifest_to_parquet_streaming.py | 流式写 Parquet | shard、rejects、resume、lineage 字段 |
| validate/validate_manifest.py | 基础字段/路径/id 检查 | 报告落盘、全量/抽样模式、质量阈值 |
| validate/validate_schema.py | conversation/image 校验 | 发布 gate、全量检查、版本信息 |
| evaluate/generate_clevr.py | 逐样本生成 | checkpoint/manifest/evaluator 元数据、分片、coverage |
| evaluate/analyze_clevr_eval.py | overall/task/answer/difficulty 分组 | JSON report、CI、反馈字段 |
| third_party/minimind-v/trainer/train_sft_vlm.py | freeze_llm=0/1/2、checkpoint | 显式 checkpoint 路径、外置 val、best checkpoint、run card |

已知风险：

1. trainer 的 base/resume 默认路径依赖 trainer 相对目录，和 shell 中 SAVE_DIR 不一定一致；
2. trainer 没有完整 val 参数，不能直接选择 best checkpoint；
3. 当前 Dataset/Collate 主要向模型返回 conversation/image，sample metadata 可能丢失；
4. 训练数据 recipe 的硬编码权重不适合 Round 反馈；
5. 评估脚本没有统一 prediction event 和 JSON 报告。

因此第一阶段不直接重写 third_party，而是在 src/minimind_v_lab 增加 wrapper、orchestrator、artifact registry、gate、feedback 和 evaluator。

## 2. 统一执行契约

### 2.1 单个 job 的固定流程

~~~text
resolve_config
→ validate_inputs
→ execute_to_tmp
→ validate_outputs
→ atomic_commit
→ write_success_marker
~~~

具体规则：

1. resolve_config：读取 YAML 和 CLI，补默认值，解析绝对路径，生成 config_hash；
2. validate_inputs：检查上游 artifact、schema_hash、content_hash、license 和前置 _SUCCESS；
3. execute_to_tmp：所有输出写入临时目录，禁止直接覆盖正式版本；
4. validate_outputs：校验 rows、schema、stats、gate 和输出 hash；
5. atomic_commit：通过检查后原子改名；
6. write_success_marker：最后生成 manifest.json 和 _SUCCESS。

下游只读取同时满足以下条件的 artifact：

~~~text
manifest.complete = true
存在 _SUCCESS
schema_hash 可兼容
parent_artifact_ids 可解析
~~~

### 2.2 统一 Job 状态

~~~text
PENDING
RUNNING
SUCCEEDED
FAILED
SKIPPED
REVIEW
~~~

每次状态变化追加到 events.jsonl，同时更新 pipeline_state.json 快照。事件至少包含：

~~~json
{
  "run_id": "A-r1-seed42",
  "job_id": "materialize-r1",
  "stage": "recipe",
  "status": "SUCCEEDED",
  "attempt": 1,
  "started_at": "<iso-time>",
  "finished_at": "<iso-time>",
  "input_hashes": ["<hash>"],
  "output_hashes": ["<hash>"],
  "rows_in": 10000,
  "rows_out": 9000,
  "rows_quarantined": 1000,
  "error_class": null
}
~~~

### 2.3 幂等规则

同一 job_id、config_hash 和 input fingerprint 已经产生成功 artifact 时，重跑必须复用或 SKIPPED；不同输入或配置不能覆盖旧输出，必须创建新的 dataset_version、recipe_version 或 run_id。

每个 stage 都使用同一套 artifact 三件套：

~~~text
data.parquet / data.jsonl
stats.json
manifest.json
_SUCCESS
~~~

manifest 至少包含：

~~~json
{
  "artifact_id": "gold-gqa-v1-part-00003",
  "artifact_type": "gold",
  "uri": "data/lake/gold/gqa/version=v1/part-00003.parquet",
  "schema_version": "canonical_v1",
  "schema_hash": "<hash>",
  "content_hash": "<hash>",
  "parent_artifact_ids": ["silver-gqa-v1"],
  "sample_count": 12345,
  "token_count": 678901,
  "producer_job_id": "clean-gqa-v1",
  "complete": true
}
~~~

### 2.4 原子提交和恢复

~~~python
def commit_artifact(tmp_dir, final_dir, result):
    write_stats(tmp_dir, result)
    write_manifest(tmp_dir, result)
    validate_gates(result.gates)
    write_text(tmp_dir / "_SUCCESS", "complete=true")
    if final_dir.exists():
        raise ArtifactCollision(final_dir)
    os.replace(tmp_dir, final_dir)
~~~

失败策略：

| error_class | 处理 |
| --- | --- |
| transient_io | 指数退避，最多 3 次 |
| worker_crash | 只重跑当前 shard，最多 3 次 |
| resource_oom | 减小 shard/batch 重试 1 次 |
| schema_error | 不重试，进入 quarantine |
| quality_gate_fail | 不提交 gold，进入 REVIEW |
| license_or_leakage | 立即阻断发布 |
| unknown | 停止且不写 _SUCCESS |

## 3. Run 目录和 CLI

### 3.1 Run 目录契约

~~~text
runs/{track}/{experiment}/{run_id}/
├── config.resolved.yaml
├── run_manifest.json
├── data_card.json
├── inputs.json
├── recipe.yaml
├── command.txt
├── state/
│   ├── pipeline_state.json
│   ├── events.jsonl
│   └── sampler_state.json
├── logs/steps/{job_id}.jsonl
├── manifests/{train,val,test}.parquet
├── checkpoints/step-XXXX/
├── predictions/
├── metrics/
├── reports/
└── _SUCCESS
~~~

run_manifest 必须记录：

~~~text
run_id, track, round, status
dataset_version, schema_version, tag_version, recipe_version
git_commit, environment_hash, seed
base_checkpoint_uri, base_checkpoint_hash
resume_checkpoint_uri, resume_checkpoint_hash
freeze_llm, optimizer, learning_rate
max_steps, total_tokens, effective_batch_size, world_size
train_manifest_hash, val_manifest_hash, test_manifest_hash
created_at, started_at, finished_at
~~~

### 3.2 推荐命令

~~~bash
python -m minimind_v_lab.cli ingest \
  --source gqa --dataset-version gqa_raw_v1 \
  --config configs/datasets/gqa_aligned_v1.yaml \
  --run-id ingest-gqa-v1

python -m minimind_v_lab.cli clean \
  --input-artifact data/lake/bronze/gqa/version=v1 \
  --policy configs/quality/gqa_v1.yaml \
  --run-id clean-gqa-v1

python -m minimind_v_lab.cli materialize \
  --tags data/lake/gold/clevr/tags/static_v1.parquet \
  --recipe configs/recipes/line_a_curriculum_r1.yaml \
  --output runs/A-r1/manifests/train.parquet

python -m minimind_v_lab.cli run \
  --track A --round r1 \
  --config configs/pipelines/line_a_r1.yaml \
  --run-id A-r1-seed42

python -m minimind_v_lab.cli eval \
  --checkpoint runs/A-r1-seed42/checkpoints/step-000500 \
  --manifest data/manifests/eval/val_v1.parquet \
  --suite vqa,bucket,visual_dependency \
  --run-id A-r1-val-000500

python -m minimind_v_lab.cli feedback \
  --run-id A-r1-seed42 --round 1 \
  --output configs/recipes/line_a_curriculum_r2.yaml
~~~

B 线使用同一 CLI，只替换 pipeline 和 mixture recipe：

~~~bash
python -m minimind_v_lab.cli run \
  --track B --round mix-s75r25 \
  --config configs/pipelines/line_b_s75r25.yaml \
  --run-id B-s75r25-seed42
~~~

## 4. Shared P0 数据链路

### 4.1 数据设计

在采集前冻结：

~~~text
task ontology
canonical schema
required fields and enum values
answer normalization rules
quality thresholds
split policy
sampling policy
license/use restrictions
data card template
~~~

输出：

~~~text
configs/data_contracts/canonical_v1.yaml
configs/datasets/clevr.yaml
configs/datasets/gqa_aligned_v1.yaml
configs/quality/gqa_v1.yaml
configs/pipelines/line_a_r1.yaml
configs/pipelines/line_b_s75r25.yaml
data_cards/<dataset_version>.md
~~~

门禁：没有设计版本、字段约束和质量阈值的 raw 数据不能进入 ingest。

### 4.2 Ingest 和来源登记

每个 source card 记录：

~~~text
source_uri
license
acquired_at
file_size
sha256
expected_file_count
expected_sample_count
actual_file_count
actual_sample_count
source_version
parser_version
~~~

处理步骤：

1. 检查 URI、许可证和本地路径；
2. 计算文件 size/sha256；
3. 按 shard 流式读取；
4. 为原始行生成 raw_record_id；
5. 用 source + source_version + raw_record_id 生成稳定 sample_id；
6. 写 bronze；
7. 写 source_manifest 和 ingest_report。

输出：

~~~text
data/lake/bronze/{source}/dataset_version={v}/part-*.parquet
data/lake/bronze/{source}/dataset_version={v}/source_manifest.json
data/lake/quarantine/{source}/stage=ingest/part-*.parquet
~~~

G0 门禁：文件 hash、license、预期/实际数量和解析率齐全；失败行不能 silent drop。

### 4.3 Parse：统一 canonical schema

每条 canonical sample 至少包含：

~~~text
raw_record_id
sample_id
source
source_version
source_split
source_row_id
image_id
scene_id
image_uri
image_sha256
width
height
mime_type
decode_status
question
normalized_question
answer
normalized_answer
conversation
parser_version
schema_version
~~~

CLEVR 保留 program、spatial_relations、program_length、scene_file、scene_hash。GQA 保留 is_balanced、question_type、semantic_operations、scene_graph_id、short_answer、long_answer。

坏行写 parse_errors.parquet，必须包含原始定位、错误类型和 raw hash。G1 门禁：必填字段/类型合格，sample_id 唯一，image_id 与 image_uri 关联一致。

### 4.4 Profile：清洗前体检

先统计，不直接删除：

~~~text
rows_total
null_rate_by_field
question_length min/p50/p95/p99/max
answer_type_distribution
image_decode_rate
image_size/aspect distribution
exact_duplicate_rate
near_duplicate_candidate_rate
conflict_rate
source/task/difficulty distribution
~~~

输出：

~~~text
reports/profile/<dataset_version>/data_profile.json
reports/profile/<dataset_version>/bucket_profile.parquet
~~~

阈值写入 quality YAML，不藏在代码常量中。

### 4.5 Clean：修复、过滤和质量分层

固定顺序：

~~~text
schema/required fields
→ image existence/decode/mime/size
→ Unicode/whitespace/control characters
→ question length
→ answer normalization
→ image-question-answer consistency
→ exact dedup
→ near-duplicate report
→ conflict detection
→ leakage check
→ quality tier
~~~

每行追加：

~~~text
quality_status: accepted|review|rejected
quality_tier: gold|silver|review
rejection_reasons[]
rule_version
dedup_group_id
conflict_group_id
near_dup_score
leakage_status
~~~

输出：

~~~text
data/lake/silver/{source}/dataset_version={v}/part-*.parquet
data/lake/quarantine/{source}/stage=clean/part-*.parquet
reports/quality/{dataset_version}/quality_summary.json
reports/quality/{dataset_version}/rejection_log.parquet
~~~

G2 门禁：gold 中确定性精确重复和互斥冲突为 0；quarantine/review 均有原因；图片解码率、答案合法率和关键 bucket 覆盖达到配置阈值。高 loss 不等于低质量，模型错误只能写入动态标签。

### 4.6 去重和去污染

区分三类问题：

~~~text
冗余：同一数据重复出现
泄漏：训练信息进入 val/test
错误：标签或图像本身不可信
~~~

使用的键和报告：

~~~text
exact_content_hash
image_sha256
image_phash
near_text_hash
semantic_similarity
duplicate_group_id
canonical_sample_id
dedup_action
~~~

默认同图多问题保留；同一 image_id + normalized_question 的互斥答案进入 conflict。评测集至少做 image_id、question exact/normalized、n-gram、embedding 四层污染检查。

G2b 门禁：train/val/test image 或 scene overlap 为 0；确定性 test contamination 阻断发布。

### 4.7 Tag：静态和动态标签

静态 tags 独立保存，append-only：

~~~text
sample_id
tag_version
label_source
label_confidence
task_coarse
task_fine
answer_type
question_length
reasoning_hops
object_count
relation_count
distractor_level
difficulty_static
visual_necessity_prior
information_score
uniqueness_score
representativeness_score
~~~

label_source 区分 program_truth、dataset_annotation、rule、model_estimate、human_review。

动态标签主键为 sample_id + checkpoint_id + round：

~~~text
loss
confidence
prediction
normalized_prediction
correct
error_type
first_correct_step
forgetting_count
prediction_stability
visual_dependency_score
learning_status
~~~

静态 v1 永远保留；Round 1 后新增 v2，不覆盖 v1。

### 4.8 Split：按 group 切分

~~~text
CLEVR：scene_id 或 image_id
GQA：image_id
~~~

输出 split_assignment.parquet 和 split_report.json，包含 group_id、split、seed、split_version、assignment_hash、各 bucket coverage 和 overlap count。

G3 门禁：train/val/test overlap=0，核心 bucket 在 val/test 有最小覆盖；不足则 REVIEW，不为追求分数反复调整 test。

### 4.9 Recipe：物化不可变训练清单

配方 YAML 明确比例语义：

~~~yaml
recipe_id: line_a_r1_v1
selection_mode: stratified_weighted
seed: 42
total_rows: 20000
target_token_budget: null
max_sample_repeat: 3
allowed_quality_tiers:
  - gold
buckets:
  spatial_easy:
    target_fraction: 0.25
    min_exposure: 1000
    max_exposure: 8000
  count_medium:
    target_fraction: 0.20
    min_exposure: 1000
    max_exposure: 6000
replay:
  clean_anchor_ratio: 0.10
  previous_stage_ratio: 0.10
~~~

物化步骤：

1. join tags + split + quality；
2. 仅取 train；
3. 按 source/task/difficulty/bucket 建候选池；
4. 计算 quota；
5. 固定 seed 抽样；
6. 候选不足时记录 shortage；
7. 允许重复时生成 exposure_id；
8. 生成 dense index；
9. 写 rows/tokens/images exposure；
10. 通过 G4 后才转 serving。

每行记录：

~~~text
sample_id
source_row_id
source
bucket
sampling_weight
repeat_count
exposure_id
selected
selection_reason
recipe_version
manifest_version
dense_index
~~~

G4 门禁：source/bucket rows 比例、source token 比例、max repeat、shard skew 和总 budget 均在配置范围内。

### 4.10 Serving 和 token cache

训练 serving schema：

~~~text
sample_id
image_id
source
split
question
answer
conversations
image_uri 或 image_bytes
task
bucket
quality_tier
recipe_version
~~~

训练 batch 可以只消费模型字段，但 sample_id、source、bucket 必须通过 metadata 或稳定 dense_index 映射保留。

token cache 指纹覆盖：

~~~text
dataset_hash
manifest_hash
recipe_hash
tokenizer_hash
chat_template_hash
preprocess_code_hash
cache_version
~~~

目录：

~~~text
data/tokenized_cache/{model}/{cache_hash}/
├── dataset/
├── config.json
├── dataset_statistics.json
├── lock
└── READY
~~~

G5 门禁：conversation、image token、图片、answer、token length 和 tokenization error 均可解释；READY 之前不能训练。

## 5. Checkpoint、训练和评估执行

### 5.1 Checkpoint 完成语义

~~~text
write checkpoints/.tmp/step-000500/
  model
  optimizer
  scheduler
  scaler
  RNG state
  sampler state
  checkpoint_meta.json
write COMPLETED
atomic rename to checkpoints/step-000500/
~~~

只有带 COMPLETED 的 checkpoint 才能 resume/eval。metadata 至少包含 checkpoint_id、global_step、optimizer_step、epoch、manifest_hash、sampler_seed、batches_processed、stage_id、base_checkpoint_hash 和 trainable_parameter_count。

### 5.2 MiniMind-V 训练链

~~~text
base checkpoint
→ projector warmup：freeze_llm=2
→ val，登记 projector checkpoint
→ freeze_llm=1 SFT
→ val，选择 best checkpoint
~~~

项目 wrapper 训练顺序：

1. 解析 base/resume 的绝对路径；
2. 计算 checkpoint hash；
3. 校验 train/val manifest hash；
4. 记录 freeze_llm 和 trainable parameter names/count；
5. 启动 third_party trainer；
6. 收集 checkpoint、optimizer、scaler、RNG、sampler state；
7. 在固定 block/epoch 结束后外置 val；
8. 按配置选择 best checkpoint；
9. 写 train/val metrics 和 run card。

若暂时不能修改 trainer，就用固定 block：

~~~text
训练 N steps
→ 保存 checkpoint
→ 外置 val
→ 记录 metrics
→ 继续或 rollback
~~~

不能自动把最后 checkpoint 当 best。

### 5.3 sample metadata 传递

建议 Dataset 返回：

~~~python
model_inputs = {
    "conversations": row["conversations"],
    "image_bytes": row["image_bytes"],
}
metadata = {
    "sample_id": row["sample_id"],
    "exposure_id": row["exposure_id"],
    "source": row["source"],
    "bucket": row["bucket"],
    "dense_index": row["dense_index"],
}
return model_inputs, metadata
~~~

若第三方 trainer 不接受 metadata，则用 dense_index 到 sample_id 的映射表回写；映射表 hash 写入 checkpoint metadata。只保存 batch loss 不能实现逐样本学习动态，第一版至少保证 eval 阶段逐样本 loss/prediction/correct。

### 5.4 Prediction event

每条预测保存：

~~~text
sample_id
image_id
source
split
checkpoint_id
checkpoint_hash
manifest_hash
recipe_version
gold_answer
prediction
normalized_prediction
correct
confidence
loss
error_type
evaluator_version
normalizer_version
latency_ms
~~~

评估支持 shard_index、num_shards、resume、manifest_id、checkpoint_id。汇总前检查 prediction coverage；缺失样本写 missing_predictions.parquet，coverage 不达标不能生成最终 report。

G6 门禁：prediction coverage 达标，evaluator/normalizer/checkpoint/manifest 版本完整。

## 6. 主线 A：课程学习和数据飞轮

### 6.1 A 线 job DAG

~~~text
probe_profile
→ static_tag
→ initial_bucket
→ materialize_stage
→ projector_warmup
→ sft_block
→ checkpoint_eval
→ learning_dynamics
→ bucket_feedback
→ recipe_diff
→ replay_update
→ next_stage_or_stop
~~~

### 6.2 课程阶段

~~~text
Stage 0：高质量、视觉必要、低难度
Stage 1：低/中难度，按 task bucket 平衡
Stage 2：增加 reasoning hops、object count、distractor
Stage 3：经常错但 loss 正在下降的 learnable-hard
Stage 4：hard + clean anchor replay + 反事实视觉样本
~~~

stage 是 policy artifact，不是硬编码 epoch：

~~~yaml
stage_id: stage_2
replay_ratio: 0.15
clean_anchor_ratio: 0.10
min_bucket_exposure: 1000
max_sample_repeat: 3
exit_conditions:
  min_steps: 2000
  target_bucket_accuracy: 0.75
  max_replay_drop: 0.02
  max_text_replay_drop: 0.02
  loss_slope_window: 5
fallback_stage: stage_1
~~~

调度器只返回 stay、advance、rollback、stop，并保存 curriculum_state.json。

### 6.3 学习动态

至少两个 checkpoint 记录：

~~~text
loss_t
correct_t
confidence_t
first_correct_step
forgetting_count
loss_slope
prediction_stability
~~~

样本状态：

~~~text
easy_mastered：早期正确、后续稳定、loss 低
learnable_hard：当前错误/不稳定、loss 下降、质量通过
noisy_or_unlearnable：反复错误、loss 无改善、存在冲突/低质量/强分歧
~~~

分类先查数据质量，再查 loss/correct/confidence，再查跨 checkpoint 稳定性；不能用高 loss 直接等价 hard。

### 6.4 Feedback → recipe

feedback 输入只来自 train/val：

~~~text
bucket_accuracy
sample_count
target_accuracy
task_impact
forgetting_rate
learnability
noise_risk
coverage_deficit
~~~

推荐控制器：

~~~python
def update_weight(old_weight, stats, policy):
    deficit = max(0, policy.target(stats.bucket) - stats.accuracy)
    score = (
        deficit
        + policy.lambda_impact * stats.task_impact
        + policy.lambda_forgetting * stats.forgetting_rate
        + policy.lambda_learnability * stats.learnability
        + policy.lambda_coverage * stats.coverage_deficit
        - policy.lambda_noise * stats.noise_risk
    )
    return clip(
        old_weight * (1 + policy.eta * score),
        policy.min_weight(stats.bucket),
        policy.max_weight(stats.bucket),
    )
~~~

更新后强制：

~~~text
clean anchor floor
previous-stage replay floor
source floor
max_sample_repeat
rows/token budget normalization
~~~

输出 recipe_proposed.yaml、recipe_diff.json、feedback_report.json、curriculum_state.json，先 REVIEW/APPROVE，再生成下一轮 manifest。

### 6.5 Round 1 后细分箱

只有同时满足 sample_count、置信区间、目标差距和可解释维度条件才细分。

例如 spatial 粗桶低分：

~~~text
spatial × relation_type × reasoning_hops × distractor_level
~~~

实现：

1. 从 dynamic_round1 找低分且样本量充分的 bucket；
2. 用 sample_id join canonical/program/scene；
3. 只计算该 bucket 需要的细标签；
4. 写 tags_v2，不覆盖 tags_v1；
5. 重新物化下一轮 recipe；
6. 保存 v1/v2 coverage diff。

### 6.6 A 线停止和消融

停止条件：

~~~text
目标 bucket 达标
连续两轮提升低于 min_effect
loss 下降但 val 不升
replay/text replay drop 超阈值
coverage 不增加且 noise 上升
预算耗尽
~~~

动作是 stay、advance、rollback、stop。至少比较 Uniform、Static balanced、Easy-to-hard、Learning-dynamics、Learning-dynamics+replay，固定 base checkpoint、freeze strategy、总 steps、seed 和 eval manifests。

## 7. 主线 B：GQA、混编和能力保持

### 7.1 B 线 job DAG

~~~text
ingest_gqa
→ schema/image/text_validate
→ normalize
→ exact_dedup
→ near_dedup_report
→ conflict_check
→ leakage_check
→ quality_tier
→ gqa_gold_manifest
→ mix_ratio_materialize
→ fixed_budget_train
→ vqa_eval
→ text_replay_eval
→ instruction_eval
→ visual_dependency_eval
→ ratio_report
~~~

B 主结果不带入 A 的动态课程反馈；阶段式混编另命名为 B-StageMix。

### 7.2 GQA 质量层

~~~text
gold：图像可读、答案可归一化、题型可映射、无冲突、无泄漏
silver：基本可训练但有低置信/开放答案/轻度近重复，需 recipe 显式允许
review：阈值附近、规则分歧、长尾、高影响、需要人工判断
quarantine：坏图、schema 失败、答案冲突、确定泄漏、无法恢复
~~~

每个版本输出：

~~~text
silver_samples.parquet
quality_summary.json
duplicate_clusters.parquet
conflict_cases.parquet
leakage_report.json
quarantine.parquet
gold_manifest.parquet
data_card.md
~~~

### 7.3 混编 recipe

明确 fraction/count/upsample：

~~~yaml
mixture_id: s75r25
seed: 42
budget:
  optimizer_steps: 5000
sources:
  - name: clevr
    dataset_version: clevr_v1
    split: train
    selection_mode: fraction
    requested_fraction: 0.75
  - name: gqa_aligned_v1
    dataset_version: gqa_aligned_v1
    split: train
    selection_mode: fraction
    requested_fraction: 0.25
~~~

物化时分别采样、生成 dense index、按 hash(sample_id) 重分片，并输出：

~~~text
requested/effective rows
requested/effective tokens
source share by rows/tokens/images
task/bucket/quality share
upsample_factor
max_repeat
seed
recipe_hash
~~~

首轮跑 S100R0、S75R25、S50R50、S0R100；所有比例组固定 base、freeze、optimizer、effective batch、总 steps/token budget、checkpoint schedule、val/test、seed 和 evaluator version。

### 7.4 阶段式混编

单独测试：

~~~text
80% CLEVR + 20% GQA
→ 60% CLEVR + 40% GQA
→ 50% CLEVR + 50% GQA
~~~

必须保持总 optimizer steps、总 token/image exposure 和评估集可比较，不能把“训练更久或看过更多数据”归因于课程策略。

### 7.5 通用能力退化和 release gate

统一评估：

~~~text
text-only replay loss/perplexity
short instruction exact match
format following rate
visual dependency stress
~~~

比较 base、projector-only、freeze_llm=1 SFT 和每个 mixture checkpoint：

~~~text
retention_delta = post_metric - base_metric
forgetting_rate = (base_metric - post_metric) / max(base_metric, eps)
~~~

visual stress 至少使用真实图、黑图、错配图、打乱图，并记录 prediction change、confidence change、latency。

只有以下都通过才更新 serving/latest：

~~~text
GQA gold gate PASS
rows/tokens ratio gate PASS
prediction coverage PASS
VQA 达到最小 effect 或不回归
text replay drop 不超过阈值
visual dependency sanity PASS
run card/data card/license/lineage 完整
~~~

VQA 提升但通用能力退化超阈值时，只保留实验 artifact，状态为 COMMITTED_EXPERIMENT。

## 8. Lakehouse、分片、批流和数据倾斜

### 8.1 数据层

~~~text
data/lake/raw/
data/lake/bronze/
data/lake/silver/
data/lake/gold/
data/lake/quarantine/
data/manifests/
data/predictions/
data/metrics/
data/tokenized_cache/
~~~

raw 只读；silver/gold 只能新版本写入；quarantine 保留原始字段和原因。

### 8.2 分区、分片和 skew gate

推荐分区：

~~~text
dataset/source_split/task_type/quality_tier/version
~~~

每个 shard 统计 rows、tokens、bytes、unique_images、task_count、P50/P95、max/min。默认 max_shard_tokens > 2 × median_shard_tokens 时触发按 hash(sample_id) 重分片。

source 比例同时按 rows 和 tokens 统计，防止长回答源被低估。小文件通过合并/compaction 管理，不让每条样本单独落盘。

### 8.3 批处理和流式事件

当前采用：

~~~text
batch：ingest/parse/clean/tag/split/materialize/tokenize
append-only：train_step/checkpoint/prediction/bad_case/human_review
micro-batch：每 N 个 checkpoint 或每个 round 聚合 feedback
~~~

接口预留 BatchRunner、EventSource、StateStore、ArtifactRegistry，不先引入 Kafka/Flink。

## 9. 验收和 Definition of Done

### 9.1 通用 gate

~~~text
G0：source/license/hash/expected-vs-actual 完整
G1：schema、sample_id、image/question/answer 合法
G2：gold exact duplicate/conflict=0，quarantine reason 完整
G3：split overlap=0，核心 bucket coverage 足够
G4：recipe rows/tokens/source/bucket/max-repeat/skew 通过
G5：serving conversation/image token/answer/tokenization 通过
G6：prediction coverage 和版本字段齐全
G7：config/seed/commit/environment/input-output hash/data card 齐全
~~~

任一 FAIL 都不能写最终 _SUCCESS 或更新 serving/latest；REVIEW 只能提交实验 artifact。

### 9.2 阶段完成条件

P0：job state、artifact manifest、gate report、resume 和幂等通过 fixture smoke。

P1-A：Projector/SFT 链接可追溯，外置 val 能选 best，prediction 能回连 sample_id。

P1-B：GQA 能生成 gold/silver/review/quarantine，清洗前后统计和泄漏报告完整。

P2-A：至少两个 checkpoint 有 dynamic tags，能生成 recipe diff，replay/max-repeat/source-floor 生效。

P2-B：四组 mixture 可重建，rows/tokens/images exposure 齐全，VQA/text replay/instruction/visual suite 都能跑。

P3：candidate、score、accepted、preference pairs 分层保存后，才允许试 DPO。

### 9.3 最小 smoke 命令

~~~bash
python -m minimind_v_lab.cli validate \
  --layer canonical \
  --artifact tests/fixtures/clevr_canonical.jsonl \
  --report-json tests/out/canonical_gate.json

python -m minimind_v_lab.cli materialize \
  --tags tests/fixtures/static_tags.parquet \
  --recipe tests/fixtures/recipe.yaml \
  --output tests/out/train_manifest.parquet

python -m minimind_v_lab.cli exposure-report \
  --manifest tests/out/train_manifest.parquet \
  --output tests/out/exposure.json

python -m minimind_v_lab.cli split-check \
  --assignment tests/fixtures/split_assignment.parquet \
  --report-json tests/out/split_gate.json
~~~

smoke 必须证明：重复运行不追加重复样本、坏图进入 quarantine、exposure_id 可回溯、比例可重算、split overlap 可检测。

## 10. 实施顺序和第一批代码

按依赖关系：

~~~text
P0-1 lineage/artifact_registry.py
P0-2 orchestrator/state_store.py
P0-3 data_contract/gate_report.py
P0-4 recipes/materialize_manifest.py
P0-5 training/train_manifest_adapter.py
P0-6 evaluate/run_eval.py
P1-A feedback/aggregate_bucket_metrics.py
P1-B feedback/update_recipe.py
P1-C ingestion/ingest_gqa.py + cleaning/clean_gqa.py
P2 text_replay_eval.py + visual_dependency_eval.py
P3 rejection sampling/DPO adapters
~~~

第一批不需要实现 Airflow、Spark、Kafka 或完整 NeMo Curator。先把单机可重跑、分片可恢复、输出可验收、数据可回溯、训练可复现和反馈可生成下一轮 recipe 做实。

## 11. 开源项目机制的落地边界

| 参考机制 | 当前项目落地 | 限制 |
| --- | --- | --- |
| Data-Juicer Sandbox 的 probe/refine/process/evaluate | A/B job DAG 和 round state | model-feedback hook 当前不是完整算法 |
| Sandbox ContextInfos/JobInfos | pipeline_state + events + artifact registry | 项目层需要 job 级 resume |
| Sandbox data-pool 操作 | bucket candidate manifest + exposure | 当前 data pool 偏内存式 |
| Open-Instruct mixture | mixture_spec + materialized manifest | 显式区分 fraction/count/upsample |
| Open-Instruct stable index | dense index + sample_id + sampler state | MiniMind-V batch 需保留 metadata |
| Open-Instruct cache/checkpoint | cache hash + READY、checkpoint + COMPLETED | 先由 wrapper 适配 |
| Open-Instruct decontamination | image/scene/question 多层报告 | 阈值需抽检校准 |
| MiniMind-V trainer | Projector/SFT/freeze strategy | val、路径、元数据需项目层补齐 |
| LLaVA/LLaMA-Factory/NeMo Curator | 阶段边界、dataset registry、filter/dedup 分层 | 只借鉴组织方式，不复制整套平台 |

Data-Juicer 的 model-feedback、部分 model probe 和外部 RFT executor 不应写成当前已实现；Open-Instruct 的机制作为字段、缓存和状态设计参考，仍需适配 MiniMind-V。

## 12. 最终目标

每一条数据和每一次训练都要具备：

~~~text
可读：字段和格式明确
可验证：有质量 gate 和报告
可恢复：失败可从 shard/job/checkpoint 继续
可解释：知道为什么选择、过滤或增权
可比较：预算、split、版本和评估固定
可回放：输入、代码、配置、seed 和 lineage 齐全
~~~

## 13. 从“规划”到“可运行代码”的映射

本节把前面的契约映射到当前仓库，避免出现“目录已经设计好，但不知道先写什么”的问题。实现状态使用三种标记：

```text
available：当前仓库已有脚本，可以作为输入/输出兼容的实现基础
adapter-needed：已有脚本能完成核心工作，但需要 wrapper、报告和 lineage 适配
planned：只有接口和设计，尚未声称已经实现
```

| 模块 | 当前状态 | 先实现的最小职责 | 不能由它隐式承担的职责 |
| --- | --- | --- | --- |
| `manifest/build_spatial_manifest.py` | available | 生成 CLEVR canonical 候选 | 质量门禁、版本登记、训练发布 |
| `manifest/annotate_manifest.py` | adapter-needed | 生成静态 task/answer/difficulty 标签 | 动态难度、模型反馈、标签覆盖旧版本 |
| `manifest/split_by_image_three_way.py` | adapter-needed | image-level 切分 | 分层平衡、泄漏报告、不可变 split 版本 |
| `manifest/build_initial_recipe.py` | adapter-needed | 生成首轮训练候选 | Round 反馈配方、token budget、exposure audit |
| `convert/convert_manifest_to_parquet_streaming.py` | adapter-needed | JSONL→Parquet 流式转换 | shard resume、坏图隔离、完整 lineage |
| `validate/validate_manifest.py` | adapter-needed | 字段和路径检查 | 可配置 gate、JSON 报告、发布阻断 |
| `validate/validate_schema.py` | adapter-needed | MiniMind-V serving schema 检查 | 默认抽样不能替代全量发布检查 |
| `pipeline/prepare_full_clevr.py` | adapter-needed | 串联 CLEVR 数据构建步骤 | 状态机、输入输出 hash、原子提交、恢复 |
| `evaluate/generate_clevr.py` | adapter-needed | 生成逐样本预测 | checkpoint/manifest 绑定、分片、coverage gate |
| `third_party/minimind-v/trainer/train_sft_vlm.py` | available | 执行 projector/SFT 训练 | val、best checkpoint、项目级 run card |
| `artifact_registry.py` | planned | 登记 artifact、父子血缘和 `_SUCCESS` | 训练逻辑、模型指标本身 |
| `state_store.py` | planned | 记录 job 状态、事件、重试 | 修改数据内容、替代日志系统 |
| `materialize_manifest.py` | planned | 将 recipe 物化为不可变训练清单 | 直接在 DataLoader 中偷偷改权重 |
| `feedback_to_recipe.py` | planned | 从动态指标生成下一轮 recipe 草案 | 自动判断标签真伪、绕过人工 gate |
| `text_replay_eval.py` / `visual_dependency_eval.py` | planned | 通用能力和视觉依赖评估 | 从 VQA 分数推断通用能力 |

第一条可运行纵向切片不需要先完成全部模块，只要求：

```text
CLEVR canonical → static tags → image split → materialized manifest
→ projector warmup → freeze_llm=1 SFT → 外置 val
→ prediction events → bucket report
```

当这条链路可以从空的 `run_id` 目录重放，并且每个阶段都能通过 gate，才开始加 GQA、复杂课程和反馈控制器。

## 14. 配置展开、运行器和作业生命周期

### 14.1 配置分层和优先级

配置不要散落在 shell 的环境变量和 Python 常量中。建议按以下层次组织：

```text
configs/defaults/*.yaml
  < configs/datasets/*.yaml
  < configs/quality/*.yaml
  < configs/recipes/*.yaml
  < configs/pipelines/*.yaml
  < CLI 显式覆盖
```

右侧优先级更高。运行开始时把所有变量展开为 `config.resolved.yaml`，并计算规范化 YAML 的 hash：

```python
def resolve_config(defaults, dataset, quality, recipe, pipeline, cli):
    merged = deep_merge(defaults, dataset, quality, recipe, pipeline, cli)
    merged["project_root"] = str(Path(merged["project_root"]).resolve())
    merged["train_manifest"] = str(Path(merged["train_manifest"]).resolve())
    merged["val_manifest"] = str(Path(merged["val_manifest"]).resolve())
    merged["test_manifest"] = str(Path(merged["test_manifest"]).resolve())
    merged["config_hash"] = sha256(canonical_yaml(merged))
    return merged
```

`config_hash` 不应该包含运行开始时间、临时目录和日志路径，否则相同配置无法幂等复用。`run_id` 可以包含时间，但 artifact 的内容身份仍由输入 hash、配置 hash、代码版本和 seed 决定。

### 14.2 推荐的 pipeline 配置

下面是一个可以直接作为 `configs/pipelines/line_a_r1.yaml` 起点的结构。数值只是初始实验预算，实际运行前要根据 GPU 显存和数据规模调整。

~~~yaml
pipeline_id: line_a_r1
track: A
round: r1
seed: 42
project_root: E:/DDU/Research/VLM_learning
data_root: E:/DDU/Research/VLM_learning/data
run_root: E:/DDU/Research/VLM_learning/runs

dataset:
  dataset_version: clevr_spatial_v1
  train_manifest: data/lake/gold/clevr/version=v1/split=train/manifest.parquet
  val_manifest: data/lake/gold/clevr/version=v1/split=val/manifest.parquet
  test_manifest: data/lake/gold/clevr/version=v1/split=test/manifest.parquet
  schema_version: canonical_v1
  tag_version: static_v1

recipe:
  recipe_id: line_a_r1_static_v1
  selection_mode: stratified_weighted
  total_rows: 20000
  target_token_budget: null
  max_sample_repeat: 3
  allowed_quality_tiers: [gold]
  replay: {clean_anchor_ratio: 0.10, previous_stage_ratio: 0.00}

training:
  base_checkpoint: checkpoints/minimind-v-base
  projector_warmup:
    enabled: true
    freeze_llm: 2
    epochs: 1
    learning_rate: 1.0e-4
  sft:
    freeze_llm: 1
    max_steps: 2000
    learning_rate: 5.0e-6
    effective_batch_size: 16
    max_seq_len: 512
    checkpoint_interval_steps: 500
  seed: 42

evaluation:
  val_interval_steps: 500
  suites: [vqa, bucket, text_replay, visual_dependency]
  selection_metric: vqa.overall_accuracy
  min_prediction_coverage: 1.0

gates:
  max_image_decode_failure_rate: 0.001
  max_train_val_image_overlap: 0
  max_recipe_fraction_error: 0.01
  max_text_replay_drop: 0.02
~~~

配置展开后必须保存：

```text
config.resolved.yaml
config_hash.txt
command.txt
environment.txt
git_commit.txt
```

### 14.3 runner 的实际执行顺序

统一入口的职责不是实现所有数据算法，而是管理作业生命周期：

~~~python
def run_pipeline(pipeline_config):
    cfg = resolve_all(pipeline_config)
    run = create_or_load_run(cfg.run_id)
    acquire_lock(run)
    write_resolved_config(run, cfg)

    for job in build_dag(cfg):
        if registry.has_success(job.job_id, job.input_fingerprint, job.config_hash):
            state_store.transition(job, "SKIPPED", reason="cache_hit")
            continue
        state_store.transition(job, "RUNNING")
        tmp_dir = make_tmp_dir(run, job.name)
        try:
            validate_inputs(job, registry)
            result = job.execute(tmp_dir, cfg)
            gates = validate_outputs(result, cfg.gates)
            if gates.has_fail:
                raise QualityGateError(gates)
            artifact = registry.commit(tmp_dir, result, gates)
            state_store.transition(job, "SUCCEEDED", artifact_id=artifact.id)
        except RetryableError as exc:
            state_store.record_failure(job, exc)
            retry_or_stop(job, exc)
        except Exception as exc:
            state_store.transition(job, "FAILED", error_class=classify_error(exc))
            raise
    write_run_success(run)
~~~

作业不能在正式 artifact 目录中边读边写。每个 job 的实际输出先进入：

```text
runs/<run_id>/.tmp/<job_id>/
```

只有 `validate_outputs` 和 gate 都通过，才提交到版本目录。作业退出码建议固定：

```text
0：成功或命中可复用缓存
2：配置/输入不合法
3：质量 gate 失败，需要 REVIEW
4：可重试的 I/O、worker 或 OOM
5：不可恢复的数据冲突、许可证或泄漏
10：未分类异常
```

这样 shell、CI 或未来的 Airflow/Ray executor 不需要解析日志文本来判断结果。

### 14.4 lock、resume 和并发边界

一个 `run_id` 同时只允许一个 writer。可以使用本地 lock 文件或 SQLite 状态表：

```text
runs/<run_id>/state/run.lock
runs/<run_id>/state/pipeline_state.json
runs/<run_id>/state/events.jsonl
```

允许并行的粒度是 shard/job，不是同一 artifact 的多次写入。恢复时：

1. 读取 `pipeline_state.json` 和 `events.jsonl`；
2. 找到最后一个没有 `_SUCCESS` 的 job；
3. 检查上游 artifact 是否仍然存在且 hash 不变；
4. 只重跑失败或未提交的 shard；
5. 完成后重新执行 output gate；
6. 不重写已经提交的 artifact。

如果输入 hash 变化，恢复必须停止并要求新建 `run_id`，不能把新数据悄悄接到旧 checkpoint 上。

## 15. 数据链路的可执行实现

### 15.1 统一的 row-level 处理接口

所有数据源适配器都输出 canonical row 和处理事件，不直接输出训练 batch：

~~~python
class DatasetAdapter(Protocol):
    def discover(self, source_config) -> list[SourceFile]: ...
    def parse(self, source_file) -> Iterable[RawRecord]: ...
    def normalize(self, raw_record) -> CanonicalRecord | RejectRecord: ...
    def profile(self, records) -> ProfileReport: ...
    def quality(self, record) -> QualityDecision: ...
    def tags(self, record) -> StaticTags: ...
~~~

每一个 `RejectRecord` 必须保留：

```text
raw_record_id, source, source_version, stage, rule_version,
reason_code, reason_detail, raw_locator, created_at
```

“过滤”实际是把样本写入带原因的 quarantine，而不是在 `if` 分支里直接 `continue`。

### 15.2 Ingest：从来源到 bronze

执行命令：

~~~bash
python -m minimind_v_lab.cli ingest \
  --source clevr \
  --source-config configs/datasets/clevr.yaml \
  --dataset-version clevr_raw_v1 \
  --output data/lake/bronze/clevr/version=v1 \
  --run-id ingest-clevr-v1
~~~

执行顺序：

```text
读取 source card
→ 校验 license/URI/文件存在
→ 对每个文件记录 size、mtime、sha256
→ 生成 raw_record_id
→ 解析最小字段和源定位
→ 按固定 shard_size 写 bronze
→ 写 ingest_report 和 source_manifest
```

`raw_record_id` 推荐由 `source_version + relative_path + row_number` 计算；`sample_id` 由 `source + source_version + raw_record_id` 计算。任何会改变 canonical 内容的步骤都不能重新生成 sample_id。

Ingest 结束后检查：

```text
预期文件数 == 实际文件数（或有明确 missing 列表）
预期样本数与实际样本数差异在配置阈值内
所有文件有 hash
解析失败均进入 ingest quarantine
```

### 15.3 Parse：canonical schema 和模态索引

Parse 只做结构化，不在此阶段根据模型 loss 过滤样本。图像与文本必须分开登记：

```text
image_uri：实际媒体位置
image_sha256：媒体内容身份
image_width/height/mime_type：媒体属性
question/answer：规范化前文本
conversation：供模型消费的格式
source_locator：原始文件和行号
```

图像不能读取时生成 `decode_status=failed`，但仍保存原始定位；后续 clean job 决定是否进入 gold。GQA 的 `image_id`、scene graph 和 `question_id` 必须原样保留，以便回溯官方数据。

### 15.4 Profile：先测量，再制定阈值

Profile job 不能修改数据，只产生报告：

~~~bash
python -m minimind_v_lab.cli profile \
  --input-artifact data/lake/bronze/gqa/version=v1 \
  --output reports/profile/gqa_raw_v1 \
  --sample-rate 1.0
~~~

报告至少输出：

```text
总行数/文件数/字节数
每列 null、空串、异常类型比例
图像 decode/尺寸/长宽比分布
问题/答案长度 p50/p95/p99
source、task、answer type、quality 初始分布
exact duplicate、near duplicate 候选数
潜在冲突数、split group 数
```

阈值必须写到 `configs/quality/*.yaml`。例如，不能把“问题长度 1～256”硬编码在清洗函数里后再用另一套阈值解释报告。

### 15.5 Clean：规则顺序、修复与隔离

Clean job 建议使用“规则链”，每条规则输入一个 row，输出 `accept/review/reject` 三态决策：

~~~python
RULES = [
    schema_required_fields,
    image_exists_and_decode,
    image_size_and_aspect,
    normalize_unicode_and_whitespace,
    normalize_answer,
    validate_image_question_link,
    exact_duplicate_policy,
    conflict_policy,
    leakage_policy,
]

def decide(row, policy):
    decisions = [rule(row, policy) for rule in RULES]
    if any(d.status == "reject" for d in decisions):
        return merge_rejections(decisions)
    if any(d.status == "review" for d in decisions):
        return merge_reviews(decisions)
    return QualityDecision.accept()
~~~

每条规则必须返回 `rule_id`、`rule_version`、`reason_code`、`evidence`。例如 `image_decode_failed` 的 evidence 是路径和异常类型，`answer_conflict` 的 evidence 是冲突组中的答案集合。

GQA 的默认决策建议：

```text
坏图、缺失图、无法解析的结构化字段：reject/quarantine
同图同问同答案：保留 canonical 一条，其他记录 dedup_group_id
同图同问互斥答案：review；确认冲突后 reject
开放答案但无法稳定归一化：review，不直接当作 gold
轻度近重复：先保留 silver，并输出 cluster 报告
```

清洗后的目录必须同时包含：

```text
silver/accepted.parquet
review/review_queue.parquet
quarantine/rejected.parquet
reports/quality_summary.json
reports/rejection_reason_counts.parquet
```

### 15.6 Dedup、conflict 和 leakage 的具体执行

不要用一个“dedup=True”开关掩盖三种不同问题：

| 检查 | key/方法 | 默认动作 |
| --- | --- | --- |
| 完全同记录 | canonical content hash | 保留一条，记录 dedup group |
| 同图同问 | image_id + normalized_question | 同答案合并，互斥答案 review |
| 近文本 | MinHash/normalized n-gram/embedding | 生成 cluster，gold 阈值外置 |
| 同图多问题 | image_id | 通常保留，不当作重复 |
| 图像近重复 | sha256/pHash/视觉 embedding | train 内报告；跨 split 直接阻断或 review |
| 评测泄漏 | image/scene/question/n-gram/embedding | 命中 test 阻断发布 |

泄漏检查必须在 split assignment 生成后执行一次，并在 recipe 物化后再执行一次：前者防数据集层泄漏，后者防 recipe 误把 val/test 引入训练。

### 15.7 Tag：静态标签、细标签和动态标签分开写

静态标签 job 接收 canonical/silver，不读取模型 checkpoint；动态标签 job 只接收固定 checkpoint、固定 manifest 和预测事件。推荐三个物化表：

```text
tags/static_v1.parquet       # 原始程序/规则即可得到
tags/static_v2.parquet       # Round 1 后针对缺口追加的细标签
tags/dynamic/r1/*.parquet    # checkpoint 相关，append-only
```

标签计算必须可局部运行。例如 Round 1 只发现 `spatial` 低分时，命令只对该 bucket 计算：

~~~bash
python -m minimind_v_lab.cli refine-tags \
  --base-tags data/tags/clevr/static_v1.parquet \
  --canonical data/lake/silver/clevr/version=v1 \
  --dynamic reports/A-r1/bucket_metrics.parquet \
  --where 'task_coarse == "spatial" and accuracy < 0.75' \
  --dimensions relation_type,reasoning_hops,distractor_level \
  --output data/tags/clevr/static_v2.parquet
~~~

输出 `tag_diff.json`，明确新增了多少列、多少行、多少样本无法细分，以及 v1/v2 的 coverage 是否变化。

### 15.8 Split：先 group assignment，再写 rows

split job 先对 group（CLEVR scene/image、GQA image）分配 split，再把 row join 回去：

~~~python
groups = sorted(unique(records, key=group_key))
assignment = stratified_group_assign(groups, tags, ratios, seed)
write_parquet(assignment, "split_assignment.parquet")
rows = join(records, assignment, on="group_id")
write_partitioned(rows, by="split")
write_json(report_overlap_and_coverage(rows))
~~~

`split_assignment.parquet` 一旦发布就不可变。若要改变 seed 或比例，创建 `split_version=v2`，不能覆盖 v1，否则旧的 val/test 指标无法复现。

### 15.9 Recipe 物化：把“比例”变成可审计的训练行

物化不是在训练时让 DataLoader 随机抽样，而是先生成一份训练清单：

~~~python
def materialize(recipe, train_rows, tags, seed):
    candidates = join(train_rows, tags, on="sample_id")
    candidates = apply_quality_and_split_filters(candidates, recipe)
    quotas = solve_quotas(candidates, recipe, budget="tokens_or_rows")
    selected = stratified_sample(candidates, quotas, seed=seed)
    selected = add_replay_and_anchor(selected, candidates, recipe)
    selected = add_exposure_id_and_dense_index(selected)
    report = compute_exposure_report(selected, recipe)
    assert_gate_g4(report, recipe.gates)
    return selected, report
~~~

每次物化都生成：

```text
recipe.resolved.yaml
train_manifest.parquet
exposure_report.json
shortage_report.json
recipe_hash.txt
```

`shortage_report` 必须说明某 bucket 是“数据本来不足”还是“质量过滤后不足”，不能静默用其他 bucket 补齐。若确实允许回退，recipe 中显式写 `fallback_bucket` 和 `fallback_reason`。

## 16. 数据构建：CLEVR 合成和 GQA 对齐的工程实现

### 16.1 CLEVR 的构建边界

CLEVR 的程序真值适合验证 A 线，但不能因为它干净就跳过数据工程。构建任务仍然要保留：

```text
scene generation
→ question/program generation
→ answer execution
→ image/scene/question identity
→ static tags
→ group split
→ recipe materialization
→ serving conversion
```

生成器必须保存：

```text
generator_version
scene_seed/question_seed
scene_config_hash
question_template_version
program_hash
scene_hash
```

同一 `scene_seed + question_seed + generator_version + config_hash` 应产生同一 canonical 内容。若只保存最终图片和答案，不保存 scene/program，就无法在 Round 1 后构造 relation、object_count 或 reasoning_hops 细标签。

现有 `prepare_full_clevr.py` 可作为构建逻辑基础，但项目 wrapper 需要把每个步骤拆成独立 job，至少把以下中间产物落盘：

```text
clevr/bronze/scenes.parquet
clevr/bronze/questions.parquet
clevr/silver/canonical.parquet
clevr/tags/static_v1.parquet
clevr/splits/v1/assignment.parquet
clevr/gold/recipe=<recipe_id>/train.parquet
```

### 16.2 GQA 的对齐目标

B 线不是把 GQA 原样塞进 MiniMind-V，而是构造 `gqa_aligned_v1`。对齐规则写进配置：

```yaml
alignment_id: gqa_aligned_v1
allowed_question_types: [obj, attr, relation, count, exist, compare]
require_image: true
require_normalizable_answer: true
require_stable_task_mapping: true
open_answer_policy: review
```

canonical conversation 只在 serving 层生成；原始 question、short_answer、long_answer、semantic_operations 继续保留在 silver/gold，防止后续需要重新映射时只能反解析模型输入。

### 16.3 数据构建的版本关系

构建版本应能用下面的关系回放：

```text
dataset_version
  = source_version + parser_version + quality_policy_version
tag_version
  = dataset_version + tagger_version + tag_policy_version
recipe_version
  = dataset_version + tag_version + recipe_config_hash + seed
serving_version
  = recipe_version + tokenizer/preprocess hash
run_id
  = experiment name + recipe_version + model checkpoint + seed
```

版本不要求真的采用这个字符串格式，但这些依赖必须在 manifest 中可查询。任何覆盖式写文件都会破坏这种关系，应改为新目录或新 artifact_id。

## 17. 数据飞轮：从预测事件到下一轮配方

### 17.1 事件流的最小实现

当前不引入 Kafka。用 append-only Parquet/JSONL 模拟事件流，事件来源包括：

```text
prediction_event：模型对样本的预测、loss、confidence
checkpoint_event：checkpoint 完成、step、stage
bad_case_event：错误类型、人工备注、处置结果
human_review_event：review/accept/reject/修复后的判断
recipe_event：提议、审核、接受、回滚
```

每条 event 具备 `event_id`，并按 `run_id/event_type/date` 分区。聚合器必须是幂等的：以 `event_id` 去重，重复运行不能重复累加曝光或错误次数。

### 17.2 Round 执行时序

一轮 A 线的真实执行顺序建议是：

~~~text
1. freeze static_v1 和 split_v1
2. materialize recipe_r1
3. run projector warmup（freeze_llm=2）
4. 在固定 warmup checkpoint 上运行 val
5. 从 warmup checkpoint 启动 freeze_llm=1 SFT
6. 每个 eval block 写 checkpoint_event 和 prediction_event
7. 聚合 sample/bucket 动态指标
8. 识别 learnable-hard、mastered、疑似噪声
9. 计算 bucket 缺口和任务影响度
10. 只对需要的 bucket 追加 static_v2 细标签
11. 生成 recipe_r2 草案和 diff
12. gate/review 通过后物化 recipe_r2
~~~

对应 CLI 可以先实现成串行命令：

~~~bash
python -m minimind_v_lab.cli materialize --recipe configs/recipes/line_a_r1.yaml
python -m minimind_v_lab.cli train --pipeline configs/pipelines/line_a_r1.yaml
python -m minimind_v_lab.cli eval --run-id A-r1-seed42 --checkpoint step-000500
python -m minimind_v_lab.cli aggregate-dynamics --run-id A-r1-seed42
python -m minimind_v_lab.cli propose-recipe --run-id A-r1-seed42 --next-round r2
python -m minimind_v_lab.cli review-recipe --recipe configs/recipes/line_a_r2.proposed.yaml
python -m minimind_v_lab.cli materialize --recipe configs/recipes/line_a_r2.approved.yaml
~~~

自动化编排可以后置；先确保这些命令的输入输出和状态相同。

### 17.3 动态指标如何计算

对每个 `sample_id`，按 `global_step` 排序得到：

```text
loss_t, correct_t, confidence_t, prediction_t
```

计算：

~~~python
first_correct_step = first(t for t in steps if correct_t)
forgetting_count = count(
    t for t in range(1, len(steps))
    if correct[t-1] and not correct[t]
)
loss_slope = linear_regression_slope(last_k(losses))
prediction_stability = 1 - changes(predictions) / max(len(predictions)-1, 1)
~~~

推荐的初始分类规则：

```text
easy_mastered：质量通过；first_correct_step 早；最后窗口稳定正确
learnable_hard：质量通过；当前正确率不足；loss_slope < 0 或 confidence 上升
unstable：预测在正确/错误间切换；forgetting_count 高
noisy_or_unlearnable：质量低/冲突/泄漏，或多轮 loss 无下降且无可解释难度
```

`noisy_or_unlearnable` 不是最终标签。它只产生 review 候选，必须先回查 canonical、答案、图片和程序真值，再决定进入 quarantine、hard pool 或保留。

### 17.4 bucket 聚合和置信区间

bucket 报告至少包含：

```text
bucket_id, dimension_version, sample_count, exposure_count,
accuracy, mean_loss, p95_loss, mean_confidence,
forgetting_rate, learnable_hard_rate, noise_review_rate
```

样本量较小时不要直接按 accuracy 调权重。先设置最小样本数，并计算 Wilson 区间或 bootstrap 区间：

~~~python
if stats.sample_count < policy.min_bucket_samples:
    action = "keep_weight_and_collect_more"
elif stats.accuracy_ci_low < policy.target_accuracy:
    action = "deficit_candidate"
~~~

最终报告同时保留 `point_estimate` 和 `confidence_interval`，防止把随机波动误认为课程收益。

### 17.5 反馈控制器的可执行版本

反馈控制器先做“受约束的配方更新”，不做黑箱优化：

~~~python
def propose_recipe(previous, bucket_stats, policy):
    proposed = deepcopy(previous)
    for bucket, stats in bucket_stats.items():
        if stats.sample_count < policy.min_bucket_samples:
            continue
        deficit = max(0.0, policy.target(bucket) - stats.accuracy)
        score = (
            deficit
            + policy.lambda_impact * stats.task_impact
            + policy.lambda_forgetting * stats.forgetting_rate
            + policy.lambda_learnability * stats.learnable_hard_rate
            + policy.lambda_coverage * stats.coverage_deficit
            - policy.lambda_noise * stats.noise_risk
        )
        proposed.weight[bucket] = clip(
            previous.weight[bucket] * (1 + policy.eta * score),
            policy.min_weight(bucket),
            policy.max_weight(bucket),
        )

    proposed = enforce_source_floor(proposed, policy)
    proposed = enforce_anchor_floor(proposed, policy)
    proposed = enforce_max_repeat(proposed, policy)
    proposed = normalize_budget(proposed, policy.budget)
    return proposed
~~~

控制器输出不是直接生效的训练文件，而是四个审计产物：

```text
feedback_report.json
recipe_proposed.yaml
recipe_diff.json
review_queue.parquet
```

`recipe_diff.json` 逐 bucket 记录旧权重、新权重、触发指标、样本量、置信区间和更新原因。只有 gate/review 通过后，才把 proposed 改名为 approved，并生成新 manifest。

### 17.6 人工 review 和回滚

review 队列不是人工重新标注全量数据，而是优先看：

```text
高任务影响度 + 高错误率
高 loss 但程序真值/答案可能冲突
同图同问多答案
模型极高置信度但错误
跨 checkpoint 反复遗忘
```

人工结果以 append-only JSONL 保存：

~~~json
{"sample_id":"...","decision":"keep_hard","reason_code":"valid_composition","reviewer":"human_1","tag_version":"review_v1"}
{"sample_id":"...","decision":"quarantine","reason_code":"answer_conflict","reviewer":"human_1","tag_version":"review_v1"}
~~~

如果新配方造成 val 目标桶提升但 text replay 或 clean-anchor 超过退化阈值，状态机必须执行 `rollback`，恢复上一版 approved recipe；不能只删除报告中的坏结果。

## 18. 课程学习、训练预算和验证时机

### 18.1 阶段不是 epoch 数，而是策略状态

课程 stage 由数据资格、配方、训练预算和退出条件共同定义。推荐状态：

```text
READY → RUNNING → EVALUATING → ADVANCE
                         ├→ STAY
                         ├→ ROLLBACK
                         └→ STOP
```

每个 stage 保存：

```text
stage_id, parent_stage_id, recipe_version,
entry_reason, min_steps, max_steps,
entry_metrics, exit_metrics, action, timestamp
```

### 18.2 以 step 为主、以 epoch 为辅

如果 recipe 有重复采样、source 混编或 token 长度差异，epoch 不再代表相同的数据曝光量。因此本项目的主比较单位是：

```text
optimizer_steps + effective_batch_size + token/image exposure
```

建议：

```text
Projector warmup：可按 1 epoch 做初始对齐，但同时记录实际 optimizer steps
A 线 SFT：按固定 step block 训练和验证，例如每 500 steps
B 线混编：必须固定总 steps 或总 token budget，不用“各自 1 epoch”比较
小型 smoke：可以每 epoch 验证，正式实验仍写入 step-based checkpoint
```

验证间隔的选择规则：

1. 先估算完整 val 的成本；
2. 设置 `min_eval_steps`，避免训练几个 batch 就过拟合判断；
3. `eval_interval_steps` 取能在一个 stage 内产生至少 3 个观测点的值；
4. 每个 epoch 末额外记录一次，便于与已有脚本对齐；
5. 只用 val 选 best/stage，不读取 test。

### 18.3 checkpoint 和 best 选择

一个 checkpoint 的可用条件是同时存在：

```text
model weights
optimizer/scheduler/scaler state
RNG state
sampler/dataloader state
checkpoint_meta.json
COMPLETED
```

`best_checkpoint.json` 记录：

```json
{
  "checkpoint_id": "step-000500",
  "selection_metric": "vqa.overall_accuracy",
  "selection_value": 0.812,
  "tie_breakers": ["bucket_macro_accuracy", "text_replay_drop"],
  "val_manifest_hash": "...",
  "evaluator_version": "v1"
}
```

如果 val 指标提升但视觉依赖 sanity 失败，best 不能只按主指标选中；应按 pipeline 中声明的 gate 先筛掉不合格 checkpoint。

## 19. 主线 B 的真实数据构建、混编和退化检验

### 19.1 GQA 真实数据的清洗输出

`gqa_aligned_v1` 不是下载后重命名，而是一个可复查的数据产品：

```text
source_manifest.json
bronze/*.parquet
profile/data_profile.json
silver/*.parquet
duplicate_clusters.parquet
conflict_cases.parquet
leakage_report.json
quality_summary.json
gold_manifest.parquet
quarantine/*.parquet
data_card.md
```

每次清洗策略变更都创建 `quality_policy_version`，并重新生成上述报告。不得只改一个过滤参数后覆盖原来的 gold。

### 19.2 混编物化算法

每个 source 先独立过滤和分层，再按 recipe 取样。不要先把 CLEVR 和 GQA 拼成一个大表再按行随机采样，否则 source、quality 和 task 的控制会互相污染。

~~~python
def materialize_mixture(spec, source_manifests):
    pools = {}
    for source in spec.sources:
        pool = load_manifest(source.manifest)
        pool = filter_quality(pool, source.allowed_quality_tiers)
        pools[source.name] = sample_to_budget(
            pool,
            requested_fraction=source.requested_fraction,
            requested_tokens=source.requested_tokens,
            seed=spec.seed,
        )
    mixed = interleave_by_hash(pools, seed=spec.seed)
    mixed = add_dense_index_and_exposure_id(mixed)
    report = exposure_report(mixed, requested=spec.sources)
    assert_mixture_gate(report, spec.gates)
    return mixed, report
~~~

比例实验至少输出四种分母：

```text
row share：样本行数占比
token share：实际训练 token 占比
image share：图像曝光占比
task share：各 source 内 task bucket 的曝光占比
```

首轮固定预算组：

```text
S100R0、S75R25、S50R50、S0R100
```

只有 source exposure、清洗版本或课程策略是研究变量时，才改变对应字段；base checkpoint、freeze strategy、optimizer、总 steps/token budget、seed、val/test 和 evaluator 版本必须相同。

### 19.3 通用能力退化的执行层

每个 B 线 run 创建独立的 `general_capability/` 目录：

```text
general_capability/
├── text_replay_predictions.parquet
├── instruction_predictions.parquet
├── visual_dependency_predictions.parquet
├── metrics.json
├── retention_delta.json
└── gate_report.json
```

最小评估顺序：

```text
base checkpoint
→ projector-only checkpoint
→ freeze_llm=1 checkpoint
→ 每个 mixture best checkpoint
```

必须报告：

```text
text replay loss/perplexity 或固定 exact match
短指令格式遵循率
VQA 主指标和 bucket macro 指标
真图/黑图/错配图/打乱图的预测变化
post - base 的 retention_delta
```

若 MiniMind-V 的接口强制图像 token，text replay 的限制必须写在 report 中；可采用固定占位图，但不能把该结果称为纯文本模型能力。

## 20. 观测、故障定位和数据倾斜处理

### 20.1 每个 job 的最小可观测性

每个 job 日志必须能回答：处理了什么、过滤了什么、产出了什么、为什么失败。建议统一字段：

```text
run_id, job_id, shard_id, attempt,
input_artifact_id, output_artifact_id,
rows_in, rows_out, rows_review, rows_rejected,
tokens_in, tokens_out, wall_time_sec,
peak_memory_mb, error_class, status
```

每 1000 行或每 30 秒写一次 progress event，避免只在进程退出时才知道卡住。

### 20.2 数据倾斜和小文件

每个 shard 输出：

```text
rows, tokens, bytes, unique_images,
task_count, p50/p95 token length,
min/max bucket exposure
```

默认规则：

```text
max_shard_tokens > 2 × median_shard_tokens：重新按 hash(sample_id) 分片
单 bucket 占比超过 recipe 上限：gate fail
小文件数量超过阈值：compact job，生成新 artifact，不覆盖旧 shard
```

长答案的 source 不能因为 row 数相同而被低估，B 线必须同时看 row 和 token exposure。若按 token 预算采样后 row 比例偏离请求比例，报告中保留两者，不强行把它们改成相同。

### 20.3 典型故障和处理动作

| 现象 | 首先检查 | 动作 |
| --- | --- | --- |
| parquet 行数异常 | `stats.json`、上游 manifest、reject log | 不发布，重跑当前 shard |
| 训练 loss 为 NaN | tokenization report、图片 decode、学习率、batch | 标记 checkpoint invalid，保留日志并回退 |
| val coverage < 100% | evaluator shard 状态、missing predictions | 只重跑缺失 shard，不重算已有预测 |
| Round 2 样本 id 对不上 | tags/manifest join key、schema version | 阻断 feedback，修复 join 后重跑 |
| 混编比例不稳定 | rows/tokens/images exposure report | 检查 upsample、repeat 和 sampler state |
| VQA 提升但 text replay 下降 | retention gate、clean anchor exposure | rollback 或降低 hard/source 权重 |

## 21. 测试与验收：从 fixture 到完整回放

### 21.1 测试分层

```text
单元测试：schema、normalizer、hash、quota、动态指标、配方控制器
fixture 集成测试：ingest→clean→split→materialize→validate
训练 smoke：一个 batch、一个 checkpoint、一次外置 val
回放测试：相同输入/config/seed 产出相同 manifest hash
故障测试：坏图、冲突答案、缺 shard、OOM、重复运行
```

### 21.2 最小 fixture

准备一个不超过几十条的 fixture：

```text
4 个 CLEVR scene
每个 scene 多个问题
1 个坏图路径
1 个同图同问同答案重复
1 个同图同问冲突答案
1 个近重复问题
train/val/test 各至少覆盖一个 task bucket
```

预期结果：

```text
坏图进入 quarantine
同答案重复只保留一条 canonical
冲突进入 review/conflict_cases
scene/image 不跨 split
recipe exposure 可按 sample_id 重算
重复运行不增加行数，不改变 artifact hash
```

### 21.3 推荐验收命令

~~~bash
python -m pytest tests/data tests/recipes tests/lineage

python -m minimind_v_lab.cli smoke \
  --fixture tests/fixtures/clevr_gqa_mini \
  --run-id smoke-$(Get-Date -Format yyyyMMddHHmmss)

python -m minimind_v_lab.cli replay-check \
  --run-id smoke-latest \
  --repeat 2 \
  --assert-same-manifest-hash \
  --assert-no-duplicate-exposure
~~~

如果当前尚未实现 `cli smoke`，先将上述命令拆成现有脚本调用；文档中的命令是目标接口，不代表该入口已经存在。

### 21.4 Definition of Done（可执行版本）

一个数据版本只有在以下条件全部满足后才叫“可训练”：

```text
[ ] source card、license、文件和样本 hash 齐全
[ ] canonical schema 全量通过
[ ] 每条拒绝记录有 reason_code
[ ] train/val/test group overlap=0
[ ] gold/review/quarantine 数量可解释
[ ] recipe rows/tokens/source/bucket exposure 可重算
[ ] serving schema 和 token cache gate 通过
[ ] run_manifest 记录模型、数据、代码、seed、预算
[ ] checkpoint 带 COMPLETED 且能重新加载
[ ] val prediction coverage 达标
[ ] test 尚未被反馈/配方读取
```

一个 Round 只有在以下条件全部满足后才叫“可迭代”：

```text
[ ] 至少两个 checkpoint 有逐样本动态记录
[ ] bucket 报告含样本数和不确定性
[ ] hard 与 noisy 候选有区分和 review 入口
[ ] recipe diff 可解释、受上限/下限约束
[ ] replay/source floor 生效
[ ] 下一版 tags/recipe 不覆盖旧版本
[ ] 失败时可 rollback 到上一版 approved recipe
```

## 22. 实施顺序：先闭环，再扩大规模

### P0：可回放的数据基础设施

先实现以下模块和 fixture：

```text
src/minimind_v_lab/lineage/artifact_registry.py
src/minimind_v_lab/lineage/run_manifest.py
src/minimind_v_lab/orchestrator/state_store.py
src/minimind_v_lab/data_contract/gate_report.py
src/minimind_v_lab/recipes/materialize_manifest.py
```

验收：相同输入/config/seed 可复用；失败不写 `_SUCCESS`；artifact 父子关系可查询；坏行可定位。

### P1：CLEVR 单轮纵向切片

```text
现有 CLEVR 构建脚本适配 artifact/state
→ static_v1 和 split_v1
→ recipe_r1
→ projector warmup
→ freeze_llm=1 SFT
→ 外置 val 和 best checkpoint
```

验收：`sample_id` 能从 prediction 回连到 tags 和 canonical；val 能按 bucket 汇总；checkpoint 能恢复。

### P2：A 线动态课程和反馈

```text
dynamic_tags.py
aggregate_bucket_metrics.py
learning_dynamics.py
feedback_to_recipe.py
curriculum_state.py
review_queue.py
```

验收：Round 1 只使用 v1 粗标签；低分且样本量充分的 bucket 才生成 v2；recipe diff 受上下限、replay 和 max-repeat 约束；能 stay/advance/rollback。

### P3：B 线 GQA 和混编

```text
ingest_gqa.py
clean_gqa.py
deduplicate.py
conflict_check.py
leakage_check.py
build_source_mixture.py
source_exposure_report.py
```

验收：产出 gold/silver/review/quarantine；四组比例可重建；rows/tokens/images exposure 齐全；清洗消融和通用能力 suite 可运行。

### P4：高级后训练和规模化执行

仅在 P0-P3 稳定后添加：

```text
rejection sampling：candidate→score→accepted
偏好对构造和 DPO
teacher distillation
Ray/Spark/Iceberg/Delta
Kafka/Flink 或真正的流式 feedback
```

这些扩展不能替代基础数据契约、质量门禁和固定预算对照；如果基础闭环尚未通过，不应同时引入多个高级变量。

## 23. 当前仓库的直接落地清单

下一次真正写代码时，建议按以下顺序提交小变更，每一步单独验证：

```text
1. 为现有 CLEVR pipeline 增加 run_manifest、step log 和 gate report
2. 给所有 canonical row 补 stable sample_id、source、schema_version
3. 将 split 输出改为 assignment + train/val/test manifest
4. 把初始 recipe 的硬编码权重移入 YAML，并生成 exposure_report
5. 给 projector/SFT wrapper 记录解析后的绝对 checkpoint 路径和 hash
6. 外置 val evaluator 输出 prediction events 和 JSON metrics
7. 从 prediction events 生成 bucket_metrics 和 dynamic_tags
8. 实现受约束的 recipe feedback，并先用 fixture 回放
9. 再接入 GQA 清洗、去重、冲突、泄漏和质量分层
10. 最后跑 CLEVR/GQA 比例实验和通用能力退化 suite
```

以上清单中的第 1～6 项是“能不能可信比较模型”的基础；第 7～8 项才是 A 线数据飞轮；第 9～10 项才是 B 线真实数据与混编研究。这样即使后续暂时没有足够算力，也能先验证数据工程闭环，而不会把尚未实现的自动反馈写成既成事实。

## 24. 编码接口附录：JobRunner、ArtifactRegistry 和 StateStore

前面的目录不是要求一次性开发一个“大平台”。第一版只需要三个小组件，就能把现有脚本包成可回放 job。组件之间通过文件和 JSON 契约通信，不共享隐式全局变量。

### 24.1 最小对象模型

~~~python
@dataclass(frozen=True)
class JobContext:
    run_id: str
    job_id: str
    stage: str
    config_hash: str
    input_artifact_ids: tuple[str, ...]
    input_fingerprint: str
    work_dir: Path
    output_dir: Path
    attempt: int
    seed: int


class JobRunner(Protocol):
    def resolve_config(self) -> ResolvedConfig: ...
    def validate_inputs(self, ctx: JobContext) -> None: ...
    def execute_to_tmp(self, ctx: JobContext) -> JobResult: ...
    def validate_outputs(self, result: JobResult) -> GateReport: ...
    def commit(self, result: JobResult, gates: GateReport) -> Artifact: ...


class ArtifactRegistry(Protocol):
    def register(self, manifest: ArtifactManifest) -> None: ...
    def resolve(self, logical_name: str, version: str | None = None) -> Artifact: ...
    def is_complete(self, artifact_id: str) -> bool: ...
    def find_by_fingerprint(self, job_key: str, input_fingerprint: str,
                            config_hash: str) -> Artifact | None: ...


class StateStore(Protocol):
    def transition(self, job_id: str, old: str | None, new: str,
                   event: dict) -> None: ...
    def get(self, job_id: str) -> dict: ...
    def resume(self, run_id: str) -> list[str]: ...
~~~

第一版可以用：

```text
data/registry/artifacts.jsonl       # append-only artifact manifest
runs/<run_id>/state/events.jsonl    # append-only job event
runs/<run_id>/state/pipeline_state.json
runs/<run_id>/state/*.lock
```

暂时不需要数据库。等查询量或并发量明显增加，再把同一接口换成 SQLite、PostgreSQL 或对象存储上的 catalog。

### 24.2 job key 和 artifact manifest

`job_id` 是人类可读名称，`job_key` 才是一次可缓存执行的稳定身份：

~~~python
job_key = sha256(
    canonical_json({
        "job_name": job_name,
        "config_hash": config_hash,
        "input_fingerprint": input_fingerprint,
        "code_commit": code_commit,
    })
).hexdigest()[:16]
~~~

artifact manifest 至少包含：

~~~json
{
  "artifact_id": "gold-gqa-v1-part-00003",
  "logical_name": "gqa_gold",
  "dataset_version": "gqa_aligned_v1",
  "artifact_type": "gold",
  "schema_version": "canonical_v1",
  "schema_hash": "sha256:...",
  "content_hash": "sha256:...",
  "stats_hash": "sha256:...",
  "parent_artifact_ids": ["silver-gqa-v1-part-00003"],
  "producer_job_id": "clean-gqa-v1",
  "job_key": "a1b2c3d4e5f60718",
  "config_hash": "sha256:...",
  "code_commit": "git-sha",
  "environment_hash": "sha256:...",
  "partition_spec": {"source": "gqa", "split": "train", "shard": 3},
  "row_count": 12345,
  "token_count": 678901,
  "byte_count": 4567890,
  "created_at": "2026-09-14T10:00:00Z",
  "status": "SUCCEEDED",
  "uri": "data/lake/gold/gqa/version=v1/part-00003.parquet",
  "complete": true
}
~~~

`complete=true` 只有在数据文件、`stats.json`、`manifest.json` 和 `_SUCCESS` 都已经提交后才能写入。注册表中永远不覆盖旧行；同一逻辑名称的新结果通过 `dataset_version` 或 `artifact_id` 区分。

### 24.3 状态转移和 stale job

合法状态转移固定为：

```text
PENDING → RUNNING → SUCCEEDED
                 ├→ FAILED
                 ├→ REVIEW
                 └→ STALE → RUNNING
PENDING → SKIPPED（命中完整缓存）
```

`transition` 写事件前先校验旧状态；发现 `RUNNING` 超过 `stale_after_minutes` 且进程心跳已停止时，标记 `STALE`，再由 `resume` 重新取得 lock。不能通过手工编辑 `pipeline_state.json` 把失败 job 伪装成成功。

失败事件必须附带：

```text
error_class, retryable, attempt, next_action,
resume_from_shard, exception_type, exception_message
```

可执行命令：

~~~bash
python -m minimind_v_lab.cli run --config configs/pipelines/line_a_r1.yaml --plan
python -m minimind_v_lab.cli run --config configs/pipelines/line_a_r1.yaml --dry-run
python -m minimind_v_lab.cli status --run-id A-r1-seed42
python -m minimind_v_lab.cli resume --run-id A-r1-seed42
python -m minimind_v_lab.cli rerun-failed --run-id A-r1-seed42
~~~

`--plan` 只解析 DAG 和输入输出；`--dry-run` 还会做输入/schema/gate 预检查但不写数据；`--resume` 只恢复最后一个成功 shard 之后的工作；`--rerun-failed` 只重跑失败 shard。默认不提供覆盖旧 artifact 的 `--force` 行为，必须改版本或新建 run。

## 25. 分片执行、数据守恒和增量构建

### 25.1 transform 的统一执行模板

每个 ingest/parse/clean/tag/convert 都采用同一模板，避免某个脚本自行定义一套失败语义：

```text
read_partition
→ map/normalize
→ filter/decide
→ write_shard(.partial)
→ write rule_counts/stats
→ validate shard gate
→ fsync
→ shard _SUCCESS
→ all-shard gate
→ dataset _SUCCESS
```

伪代码：

~~~python
def execute_partition(input_path, output_dir, policy, partition_id):
    counts = Counter()
    rejects = []
    with open_partial_shard(output_dir, partition_id) as writer:
        for raw in read_partition(input_path):
            counts["rows_in"] += 1
            decision = apply_policy(raw, policy)
            counts[decision.status] += 1
            counts[decision.reason_code] += 1
            if decision.status == "accept":
                writer.write(decision.record)
            else:
                rejects.append(to_reject_record(raw, decision))
    write_reject_shard(rejects, output_dir, partition_id)
    write_json(counts, output_dir / f"rule_counts-{partition_id}.json")
    validate_count_conservation(counts)
    fsync_and_write_success(output_dir, partition_id)
~~~

每个 job 必须满足守恒关系：

```text
rows_in = accepted + repaired + review + quarantine + rejected_drop
```

如果一个规则把行直接 `continue`，但没有增加任何计数或 rejection log，该 job 不能通过 G1/G2。`repaired` 仍要保留 `repair_actions[]`，不能把修复后的样本伪装成原始干净数据。

### 25.2 确定性分片

不要用本次运行时的顺序号作为分片身份。推荐：

~~~python
partition_id = int(sha256(
    f"{source}|{source_version}|{raw_record_id}".encode()
).hexdigest(), 16) % num_partitions
~~~

相同 source/version/partition 数时，重跑会落到同一 shard；只有新增数据会增加该 shard 的内容。若修改 `num_partitions`，生成新的 partition scheme 版本并做一次全量重分片，不在原目录混合两种方案。

### 25.3 增量重建规则

增量不是“只处理文件名更新的目录”，而是根据依赖决定受影响分片：

| 变化 | 是否可局部重建 | 原因 |
| --- | --- | --- |
| 新增 source 文件 | 是 | 只处理新文件对应的 raw_record/shard |
| parser bug 修复 | 否，重建受影响 source/version | canonical 内容可能整体变化 |
| 单个清洗阈值改变 | 通常重建 silver/gold | 过滤结果和统计可能变化 |
| 只增加静态标签列 | 是，tag artifact 新版本 | 不改变 canonical 内容 |
| recipe 权重改变 | 是，只重建 train manifest | 不改变 gold 数据 |
| tokenizer/chat template 改变 | 是，重建 token cache | serving 内容可能不变，token ids 一定变 |
| split seed/比例改变 | 否，生成新的 split_version | 训练/val/test 归属变化 |

每个增量 job 先读取父 artifact 的 `content_hash`，输出 `changed_partitions.json`，并在 manifest 中写 `rebuild_mode=incremental|full`。若上游 hash 不在允许的依赖范围内，宁可停止也不要半增量地混用旧结果。

### 25.4 compaction 和小文件治理

清洗和流式任务可能产生很多小 parquet。`compact` 是独立 job：

```text
选择同一 dataset_version/schema/partition_spec 的小 shard
→ 合并为目标大小
→ 重新计算 stats/content hash
→ 对比 rows/token/image 数量
→ 新 artifact 提交
```

compaction 不能改变样本顺序以外的内容；必须通过 `sample_id` 集合相等、row/token 计数相等和 schema 相等检查。旧 shard 保留，新的 serving manifest 指向 compacted artifact。

## 26. Schema Registry、迁移和确定性身份

### 26.1 分层 schema

不要用一个 `schema_version` 覆盖所有表。至少维护：

```text
raw_v1
bronze_v1
canonical_v1
serving_v1
static_tags_v1
dynamic_tags_v1
prediction_event_v1
recipe_manifest_v1
```

schema registry 的每个版本保存：

```text
required fields, data types, nullable policy, enum values,
unique keys, compatibility, migration entrypoint, owner
```

推荐使用 Pydantic/JSON Schema 做行级检查，再用 PyArrow 做 Parquet 列级检查。迁移只生成新 artifact：

```text
canonical_v1 → migrate_canonical_v2 → canonical_v2
```

旧版本仍能被读取和回放，不能原地改列名后继续使用旧的 `schema_hash`。

### 26.2 稳定 sample_id、group_id 和变体隔离

稳定身份建议：

~~~python
sample_id = sha256(
    f"{source}|{source_version}|{raw_record_id}".encode("utf-8")
).hexdigest()
~~~

`sample_id` 不能由 parquet 行号、DataLoader index 或本次抽样顺序生成。训练曝光另用 `exposure_id`；同一 sample 重复三次时是三个 exposure，不是三个 sample。

split 使用 group identity：

```text
CLEVR：scene_id（若无则 image_id）
GQA：image_id
反事实/增强图：parent_group_id
```

分配可采用稳定 hash，而非依赖 Python 的随机 shuffle：

~~~python
bucket = int(sha256(f"{group_id}|{split_seed}".encode()).hexdigest(), 16) % 10000
split = assign_by_threshold(bucket, train_ratio, val_ratio, test_ratio)
~~~

这样新增样本不会改变已有 group 的 assignment。若必须做分层平衡，则保存显式的 `assignment.parquet`，并把 seed 和算法版本写入 assignment hash。

### 26.3 seed bundle

所有随机源统一记录：

```json
{
  "python": 42,
  "numpy": 43,
  "torch": 44,
  "torch_cuda": 45,
  "dataloader_worker_base": 46,
  "distributed_sampler": 47,
  "recipe_sampling": 48,
  "eval_shard": 49
}
```

同一个 run 需要在 `run_manifest.json` 保存 `seed_bundle`，不能只写一个 `seed=42`。如果使用非确定性 CUDA kernel，也要在 `environment.txt` 和 run card 中说明。

## 27. 训练、评估和 token cache 的适配协议

### 27.1 MiniMind-V wrapper 的真实边界

当前 `third_party/minimind-v/trainer/train_sft_vlm.py` 可以执行 projector/SFT，但不能直接作为项目级 runner。需要 wrapper 解决：

```text
from_weight 的相对路径解析
save_dir 与 resume checkpoint 的语义分离
seed 硬编码
只有 epochs、没有 max_steps
同名 checkpoint 覆盖
没有 COMPLETED/step 子目录
没有完整 val hook
Dataset/Collate 丢失 sample metadata
```

第一版不重写第三方 trainer，而是按 block 调用它：

~~~text
prepare_run
→ inspect_checkpoint（绝对路径、hash、参数结构）
→ count_trainable_params（名称、数量、freeze_llm）
→ launch_block(max_steps=N)
→ collect_checkpoint_to_runs/<run_id>/checkpoints/step-XXXX
→ 写 optimizer/scaler/RNG/sampler/meta/COMPLETED
→ external_val
→ update_best_checkpoint.json
→ next block / advance / rollback / stop
~~~

wrapper 启动前打印并落盘：

```text
resolved_base_checkpoint
resolved_resume_checkpoint
base_checkpoint_hash
train_manifest_hash
val_manifest_hash
freeze_llm
trainable_parameter_names/count
effective_batch_size
requested_steps / actual_optimizer_steps
```

`SAVE_DIR` 不能推断模型从哪里加载；必须显式传 `--base-checkpoint` 和 `--resume-checkpoint`，并记录解析后的绝对路径。warmup 的输出 checkpoint 必须成为 SFT 的 `base_checkpoint`，而不是再次从默认 LLM 权重启动。

### 27.2 固定 block 的外置验证

如果暂时不能改 trainer 的 `max_steps`，wrapper 可以把每个 block 映射成短 epoch 或重新启动：

```text
启动/恢复训练 N steps
→ 等待带 COMPLETED 的 checkpoint
→ 运行 val evaluator
→ 追加 metrics.jsonl
→ 根据 gate 更新 best_checkpoint.json
→ 决定继续/回滚/进入下一 stage
```

不能因为最后一次训练进程退出就默认最后 checkpoint 是 best。若一个 block 没有生成完整 checkpoint，状态为 `TRAINING_INCOMPLETE`，不得参与比较。

### 27.3 serving/tokenized cache

cache key 除数据和 tokenizer 外，还必须包括：

```text
dataset_hash
manifest_hash
recipe_hash
tokenizer_hash
chat_template_hash
image_processor_hash
preprocess_code_hash
max_seq_len
truncation_policy
dtype
cache_version
```

缓存提交流程：

```text
acquire cache lock
→ write data to .tmp
→ 写 statistics/config
→ 校验 rows/token/image/error
→ 原子 rename
→ 最后写 READY
```

超长样本策略必须显式配置：

```text
keep：允许完整保留
terminate：只在 assistant answer 完整结束处截断
drop：进入 tokenization quarantine
```

不能把 assistant answer 截断一半后仍当成完整监督。tokenization report 至少记录 cache hit/miss、rows_drop、图片失败、token p50/p95/p99、truncated_rows 和 answer_truncated_rows。

## 28. A 线 Round contract 和稳定反馈控制器

### 28.1 每轮冻结的输入

Round 不是“重新跑一次训练脚本”，而是一个可审计的数据/模型快照。每轮开始先写：

```yaml
round_id: A-r1
data_snapshot_id: clevr_gold_v1
train_manifest_hash: sha256:...
val_manifest_hash: sha256:...
base_checkpoint_hash: sha256:...
eval_suite_version: v1
tag_version: static_v1
seed_bundle: {...}
feedback_policy_version: controller_v1
```

Round 产物固定为：

```text
round_spec.yaml
feedback_input.parquet
bucket_stats.parquet
candidate_replay.parquet
recipe_proposed.yaml
recipe_diff.json
approval.json
curriculum_state.json
coverage_delta.json
```

feedback 只能读取 train/val 及其预测事件；`test` 的 artifact reader 在代码层直接拒绝被 `aggregate-dynamics` 或 `propose-recipe` 打开。

### 28.2 统计稳定性和小桶保护

不要用单次 point accuracy 直接增权。先计算 Wilson 下界或经验贝叶斯收缩值：

~~~python
def wilson_lower(correct, n, z=1.96):
    if n == 0:
        return 0.0
    p = correct / n
    denominator = 1 + z*z/n
    centre = p + z*z/(2*n)
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (centre - margin) / denominator


def feedback_score(stats, policy):
    if stats.sample_count < policy.min_bucket_samples:
        return 0.0, "explore_only"
    p_lb = wilson_lower(stats.correct, stats.sample_count)
    deficit = max(0.0, policy.target(stats.bucket) - p_lb)
    score = (
        deficit
        + policy.lambda_impact * stats.task_impact
        + policy.lambda_forgetting * stats.forgetting_rate
        + policy.lambda_learnability * stats.learnable_hard_rate
        + policy.lambda_coverage * stats.coverage_deficit
        - policy.lambda_noise * stats.noise_risk
    )
    return score, "deficit_candidate"
~~~

控制器更新时同时使用：

```text
EMA 平滑历史 bucket 指标
单轮权重倍率 clip（例如 0.5～2.0）
random exploration floor
clean-anchor floor
source floor
max_sample_repeat
总 rows/token budget 归一化
```

如果同一 bucket 的权重方向连续两轮反转，标记 `feedback_oscillation=true`，自动降低 `eta` 或 rollback；不能让控制器在两个配方之间来回放大。

### 28.3 hard、replay、noise 的分层处置

反馈输出不是一个 `hard=true` 布尔列，而是三个可执行池：

```text
candidate：满足缺口/影响度/样本量条件，等待复核
accepted_hard：质量通过、loss 下降或组合难度可解释，允许增权/replay
review_or_quarantine：冲突、泄漏、低质量、长期无学习信号，禁止直接增权
```

每个池的样本可进入不同 recipe：

```yaml
replay_sources:
  clean_anchor: {ratio: 0.10, max_repeat: 2}
  previous_stage: {ratio: 0.10, max_repeat: 2}
  accepted_hard: {ratio: 0.20, max_repeat: 3}
  review_or_quarantine: {ratio: 0.00, max_repeat: 0}
```

人工 review 具有状态机：

```text
open → accepted
     ├→ rejected
     ├→ superseded
     └→ needs_more_evidence
```

旧 review 事件永不覆盖；修复后使用新 `rule_version/tag_version` 重跑质量链路。

## 29. B 线混编实验的归因协议

### 29.1 训练前先对齐非研究变量

CLEVR 和 GQA 混编前必须固定或对齐：

```text
conversation 模板
image processor 和 resize/normalize
answer normalizer
<image> token 位置
最大序列长度和截断策略
quality tier 允许集合
```

`source`、`dataset_version` 和 `quality_tier` 只作为 metadata，不放进 prompt，否则模型可能学到来源标记，比例效果无法解释。

### 29.2 四种预算和四种比例

每个 ratio run 都记录：

```text
row share：行数占比
token share：token 曝光占比
image share：图像曝光占比
task share：各 source 内 task bucket 曝光占比
```

公平比较至少固定：

```text
base checkpoint
freeze strategy
optimizer/learning rate
effective batch size
total optimizer steps
total token budget
total image exposure
checkpoint/eval schedule
seed bundle
val/test manifest
```

首轮矩阵：

```text
S100R0、S75R25、S50R50、S0R100
```

阶段式 `80/20 → 60/40 → 50/50` 单独命名为 `B-StageMix`，固定相同总 steps 和 exposure；不能把“阶段式训练更久”误报成混编策略收益。

### 29.3 能力保持的基线矩阵和统计

同一套 `eval_spec.yaml` 至少评估：

```text
base
projector-only
freeze_llm=1 SFT
每个 static ratio best checkpoint
每个 StageMix best checkpoint
```

报告同时给 point estimate、bootstrap/Wilson 95% CI、配对样本差值和 `insufficient_n` 标记。VQA 指标提升但 text replay 或 clean-anchor 超过退化阈值时，run 只能标记 `COMMITTED_EXPERIMENT`，不能更新 `serving/latest`。

### 29.4 eval_spec 和 prediction coverage

评测规格独立于训练配置：

~~~yaml
eval_spec_id: vqa_retention_v1
manifest_hash: sha256:...
checkpoint_hash: sha256:...
evaluator_version: v1
normalizer_version: answer_norm_v1
decode:
  temperature: 0.0
  max_new_tokens: 32
  stop: ["</s>"]
num_shards: 4
min_prediction_coverage: 1.0
~~~

每条 prediction event 有 `status=ok|error|timeout`。汇总前必须检查：

```text
expected sample ids == ok + error + timeout
missing sample ids 已落盘
coverage 达标
checkpoint/manifest/evaluator hash 一致
```

`test` 评估由单独的 `release_eval` 命令触发；普通 `eval`、`aggregate-dynamics` 和 `propose-recipe` 不允许读取 test manifest。

## 30. 质量守恒、抽检和运行可观测性

### 30.1 规则计数和审计样本

每条质量规则都要输出：

```text
rule_id, rule_version,
rows_in, accepted, repaired, review, quarantine, rejected_drop,
reason_counts, execution_time
```

阈值附近、长尾、冲突、模型/规则分歧和高影响样本进入固定 seed 的 `audit_sample.parquet`。抽检结果保存：

```text
sample_id, audit_reason, reviewer, decision,
evidence, reviewed_at, policy_version
```

抽检不是修改原始标签的快捷通道；人工结论必须通过新版本的修复/标签 job 回写。

### 30.2 job 日志和 run card

每条 JSON 日志统一字段：

```text
run_id, job_id, stage, shard, attempt, status,
duration_ms, rows_in, rows_out, rows_review, rows_rejected,
tokens_in, tokens_out, throughput, peak_memory_mb,
input_artifact_id, output_artifact_id, error_class
```

每轮 run card 至少包含：

```text
Python/Torch/CUDA/GPU 信息
git commit 和 git diff 摘要
完整 CLI 与 resolved config
输入/输出 artifact hash
cache hit/miss
quarantine/review 数量
训练 steps、tokens、images exposure
val/test evaluator 版本
```

本地阶段用 JSONL/Parquet 足够；W&B、Prometheus 或 OpenTelemetry 只能作为观察出口，不能替代本地可回放 artifact。

### 30.3 故障处理矩阵

| error_class | retryable | 处理 | 是否允许发布 |
| --- | --- | --- | --- |
| `validation_fail` | 否 | 修复输入或 schema，生成新版本 | 否 |
| `quality_gate_fail` | 否 | 进入 REVIEW，保留报告 | 否 |
| `transient_io` | 是 | 指数退避，最多 3 次 | 成功后是 |
| `worker_crash` | 是 | 从失败 shard 重跑 | 成功后是 |
| `oom` | 一次 | 减小 shard/batch 后重试 | 成功后是 |
| `license_or_leakage` | 否 | 立即阻断并隔离 | 否 |
| `unknown` | 否 | 停止、保留现场、人工判断 | 否 |

失败 job 不写 `_SUCCESS`；部分输出留在 `.partial/`，由 `resume` 或清理 job 处理，不能被下游当作完整 artifact。

## 31. 测试、CI 和 vertical slice 验收

### 31.1 测试目录

当前仓库尚未形成完整 `tests/` 时，先按下面的规格创建 fixture；不要为了“有测试目录”而写空测试：

```text
tests/
├── fixtures/
│   ├── clevr_gqa_mini/{raw,images,canonical,tags,recipes}
│   └── expected/{gate,exposure,metrics}
├── unit/
│   ├── test_schema.py
│   ├── test_hash_and_split.py
│   ├── test_recipe.py
│   ├── test_gates.py
│   ├── test_feedback.py
│   └── test_cache.py
├── integration/
│   └── test_pipeline_smoke.py
└── regression/
    └── golden_stats/
```

最少测试用例：

```text
schema roundtrip 和类型错误
sample_id/deterministic split 稳定性
rows_in 守恒和 rejection log 完整性
exact/near dedup 与冲突分离
split leakage
quota 的 rows/tokens/source 误差
exposure_id/max_repeat
artifact 幂等和 collision
crash/resume
cache lock/READY
checkpoint COMPLETED
prediction coverage
feedback 拒绝读取 test
controller clip/anchor/noise/oscillation
```

CI 最小命令：

~~~bash
python -m pytest tests -q
python -m minimind_v_lab.cli run --config configs/pipelines/line_a_r1.yaml --plan
python -m minimind_v_lab.cli smoke --fixture tests/fixtures/clevr_gqa_mini --run-id smoke-ci
~~~

golden report 只比较 schema、行数、关键分布和 gate 状态；时间戳、机器路径和随机日志不纳入相等判断。

### 31.2 垂直切片而非一次性大开发

把实现拆成可独立验收的切片：

| 切片 | 输入 | 预期产物 | PASS 条件 |
| --- | --- | --- | --- |
| P0-S0 | 5～20 条 fixture | schema/hash/gate report | 错误行可定位，计数守恒 |
| P0-S1 | fixture artifact | registry/state/_SUCCESS | 重跑命中缓存，失败不发布 |
| P0-S2 | CLEVR 小子集 | canonical→serving manifest | split 无泄漏，exposure 可重算 |
| P1-A-S1 | serving manifest + base ckpt | projector/SFT checkpoint + val | warmup→SFT 路径正确，best 可选 |
| P1-A-S2 | 两个以上 checkpoint | dynamic tags + recipe diff | hard/noise 分离，能 rollback |
| P1-B-S1 | GQA raw 小子集 | gold/silver/review/quarantine | 清洗前后守恒，冲突/坏图进入隔离 |
| P1-B-S2 | CLEVR/GQA gold | 四组 ratio run card | rows/tokens/images exposure 可比 |
| P2 | 固定 eval suite | retention/visual reports | 通用能力退化可量化，test 不回流 |
| P3 | accepted candidates | preference pairs/DPO artifact | chosen/rejected 可追溯后才训练 |

每个切片完成后再扩大数据量或并行度。即使 GPU 暂时不足，也可以先完成 P0-S0～P0-S2 的数据工程回放，不把“还没有模型结果”误认为工程阻塞。

## 32. 最终执行检查表

开始一次正式实验前，运行负责人逐项确认：

```text
[ ] config.resolved.yaml、config_hash、command.txt 已落盘
[ ] source/license/输入 artifact hash 已核验
[ ] schema、quality、split、leakage gate 已 PASS
[ ] train/val/test manifest hash 已冻结
[ ] recipe exposure（rows/tokens/images/task）已生成
[ ] base/resume checkpoint 绝对路径和 hash 已记录
[ ] freeze_llm 与 trainable parameter 清单已记录
[ ] seed_bundle、effective batch、总 steps/token budget 已记录
[ ] checkpoint 使用 step 目录并带 COMPLETED
[ ] val evaluator 的 coverage 和版本字段齐全
[ ] test reader 在 feedback 任务中被拒绝
[ ] A 线 recipe diff 有上下限、replay/source floor 和 review
[ ] B 线 ratio 只改变声明的 source exposure 变量
[ ] run card、data card、gate report、failure report 齐全
```

只有这些检查完成，结果才进入正式比较表；否则只能作为开发 smoke 或 `COMMITTED_EXPERIMENT` 保存。
