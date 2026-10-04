# Qwen CLEVR SFT MVP

日期：2026-10-04。版本：v0.2。状态：方案已更新；下面的 Python、配置和 shell 文件尚未实现。

目标：用 Qwen3-VL-4B-Instruct + ms-swift LoRA 快速跑通两轮 SFT 数据飞轮。CLEVR / MiniMind-V 图文 replay / MiniMind-V 文字 replay 从 80/15/5 起步；外层调来源比例，内层调 CLEVR 桶比例；计算基础学习动态，用 MMStar 检查通用视觉能力变化。服务器上仍按逐文件讲解、用户创建并粘贴的方式实现。

## 选型与范围

- 模型：`Qwen/Qwen3-VL-4B-Instruct`，不使用 Thinking 版。直接从公开多模态 Instruct 权重开始，已包含图文对齐；不需要随机初始化 Projector 或另做 Projector warmup。
- 训练：ms-swift 的 SFT/LoRA，BF16；冻结视觉编码器与视觉对齐模块，在语言模型实际检查出的线性层加 LoRA，初始 rank=16、alpha=32。保存可训练参数清单，核实视觉模块未被 `all-linear` 意外纳入。正式参数名以选定 ms-swift 版本为准。
- 环境：服务器独立 Python 3.10/3.11 环境；固定 ms-swift、Transformers、Torch 和 processor 版本。Qwen3-VL 官方要求 Transformers >=4.57.0；不沿用旧 Python 3.8 环境直接升级。
- 数据：复用 `/data/${USER}/vlm_learning/CLEVR_v1.0` 图片和现有 CLEVR 未重复采样的 train/val/test JSONL。旧 recipe 中重复的 question_index 不适合作为基础候选池。
- 首轮规模建议：CLEVR 候选池 5,000～10,000；图文 replay 2,000；文字 replay 1,000；CLEVR 控制 val 500；真实图文代理控制 val 200；训练 probe 256（CLEVR 192、图文 48、文字 16）。数字是初始预算而非已完成产物。
- 只有 CLEVR 专项数据；不接 GQA、DPO、复杂课程或完整湖仓编排。第一版只写 1 个配置、4 个 Python 文件、2 个 shell 文件。
- 实现时间目标是一晚上，不包括训练；以模型/数据可用为前提。环境冲突、大文件获取或真实 QA 验证集不足是实际可能的时间瓶颈，不通过增加框架解决。

## 七步实现

1. 环境/模型 smoke：读取一张 CLEVR 图片，完成一次生成、一次 LoRA 更新和 adapter 重载；检查单卡显存、训练参数和多卡拓扑。
2. 数据准备：复用 CLEVR image-level split；从 MiniMind-V 的同一 SFT 数据中抽图文和文字 replay，先划分 train/val；生成 sample_id、图片路径、来源和桶标签。训练 JSONL 使用 ms-swift 的 messages+images 格式，分析标签保存 sidecar。
3. Base 评估：固定解码、图片预算和评估样本，记录 CLEVR/真实图文代理控制 val 的准确率与 answer-token NLL、训练 probe；另保存 MMStar baseline。
4. Round 1：初始 80/15/5 配方物化输入，训练 300 optimizer steps；训练前后的同一 probe 足以计算第一版动态，首版不要求中间 checkpoint 评估。保存 adapter 和实际曝光。
5. 更新：聚合 NLL、正确性、遗忘次数、来源/桶统计；更新 EMA 和门控状态，自动产生外层及内层的新 recipe。
6. Round 2：使用 Round 1 adapter、新 recipe 再训练 300 steps；再次评估并生成下一版 recipe，但默认不自动启动 Round 3。
7. 汇总：锁定配方和最终 checkpoint 后，再执行 CLEVR test 与 MMStar 最终评估。保存 base→R1→R2 控制集变化、MMStar pre/post 差值、曝光和成本。另跑相同总预算的静态配方对照才可判断飞轮是否有收益。

## 数据来源、筛选与 80/15/5 配比

初始配比按实际抽样曝光行数定义：CLEVR 80%、图文 replay 15%、文字 replay 5%。额外报告 assistant token、总 token、图像数，避免长答案 replay 的实际 loss 占比被行数掩盖。首版不再额外设置 CLEVR anchor 池。

- CLEVR：仅来自项目 train，沿用 task_type×difficulty。空间关系 left/right/front/behind 保留多标签统计，不做关系笛卡尔积；候选样本少于 50 的桶合并到 task_type，同 task 仍过少则进入 other。标签、候选池和 split 在两轮中固定。
- 图文 replay：统一使用 `jingyaogong/minimind-v_dataset` 中服务器已有版本的 SFT 图文数据，优先自然图像 QA，兼容实际文件格式，不另接 GQA。不能称为 Qwen 原始训练数据。
- 文字 replay：从同一 MiniMind-V SFT 数据里识别真正的 t2t 样本；不把图文对话删掉图片冒充文字 replay。若服务器旧版没有 t2t，则从 MiniMind 配套公开文本 SFT 数据补齐并单独记录来源；不能静默省略这 5%。

本地 `third_party/minimind-v/README.md` 当前记录：`sft_i2t.parquet` 含图文 instruct、caption 和用 8×8 黑图占位的纯文本；列为 conversations、image_bytes。这是代码文档证据，不代表服务器实际文件已经核验。准备数据时先打印 schema、来源/角色/占位符和抽样记录，再适配；不假设所有版本相同。

最小处理链在 `prepare_data.py` 中完成，不拆成一堆模块：

1. 流式读已有 Parquet/JSONL，固定 seed 抽取小候选池，不全量导出 290 万张图片。Parquet 用 record batch 扫描；仅写入入选图片，已有图片直接引用。
2. 检查合法 user/assistant、非空答案、图片解码、图像占位符数量和长度；首版只接可独立使用的单图、单轮 QA，复杂多图/工具样本跳过并计数。保留原始答案，不做 LLM 改写或 judge。
3. 图文按图片内容 SHA256 组级划分，同图所有问答只进同一 split；文字按规范化对话 SHA256 去重并划分。训练/控制 val/probe 的身份明确：probe 是 train 子集，控制 val 不是。
4. t2t 根据来源标记和对话无视觉输入共同确认；8×8 黑图只作辅助证据，不能仅凭黑图误判。输出 Qwen 数据时删除占位图，文字样本不带 images/<image>。不调用 MiniMind 随机注入身份 system prompt 的处理函数。
5. 图文恢复标准 `<image>` 与图片路径，使用 Qwen processor/template；不复制 `<|image_pad|>` 展开、256×256 resize 或 MiniMind tokenizer。已有 256×256 图像无法恢复高分辨率细节，记入限制。
6. 保存筛选计数、唯一图片数、去重/划分检查和来源文件版本。首版只做精确去重，不增加 embedding、近重复检索或全量质量模型。

“真实”在首版指 MiniMind 图文 replay 的真实图像 QA 代理，而不是独立 GQA 任务。ALLaVA 来源还包括文档/图表/合成场景；有来源标记时优先自然图像子源，没有时标为 `replay_vl_proxy`，抽检并披露，不能宣称整个 replay 都是真实照片。

控制准确率的真实 QA 子集须有可自动判定的短答案：只保留单一 yes/no、数字或规范化后 1～3 个英文词的答案，人工抽查约 30 条的问题是否确实支持这种评分；排除解释、caption 和含多答案的问法。先扫一个有上限的候选窗口，凑不足 200 条就报告实际 n；少于 100 条时禁用真实来源的准确率驱动增权，保持外层 80/15/5，仍跑内层和 NLL 报告，不无期限扫库或伪造指标。

长答案图文仍可用于训练和 NLL probe，但不计入上述 accuracy。真实 QA 与 CLEVR 难度、答案风格不同，直接比较准确率只是 MVP 控制启发式，不是可比较的能力排名；后续再做任务匹配或基线归一化缺口。

## 最小可计算动态指标

- loss：teacher forcing 下每条样本 assistant answer 有效 token 的平均交叉熵；prompt/image/pad 不参与，采用正确的 causal shift。保留 sum_nll 与 answer_token_count；组 NLL 为总 sum_nll / 总有效 answer tokens，跨轮使用完全相同样本和截断规则。
- accuracy：固定生成参数的 normalized exact match，记录失败/缺失预测数量。
- learning：同一 train probe 的跨 checkpoint loss_delta、delta/实际 optimizer steps、correctness history、forgetting_count（观测正确→错误次数）；仅给可评分 QA 记录正确性，长答案/文字不强行算 EM。首版不加置信度、任务影响度等额外指标。
- val：CLEVR 控制 val 和独立的真实 QA 控制 val 决定来源/桶缺口及边际收益门控；train probe 记录学习动态，暂不据此做逐样本重采样。不能把 val 错题或答案加入 train。
- 外推：probe 没覆盖的训练样本只继承桶级权重，不能虚构逐样本 loss 或 learning status。

## 两层 group-DRO-inspired 飞轮（planned）

只借鉴 group-DRO 的“表现较差的组指数增权”机制：在 round 边界更新采样配比，不修改 ms-swift 的 batch loss；不是严格 group-DRO 优化器，也不保证 worst-group 风险下降。

### 外层：CLEVR ↔ 图文 replay

- 初始 p=(0.80, 0.15, 0.05)；文字固定 0.05，另外两个来源共占 0.95。
- CLEVR 的份额范围 [0.60, 0.85]，图文 replay 对应 [0.10, 0.35]；每次任一来源最多变动 5 个百分点。这些只是可修改的 MVP 初值。
- 有效且未被门控的两个来源中，EMA accuracy 更低的一组优先增权；若低分组被门控，保持配方，不转而给高分组无理由增权。差距小于 1 个百分点先不动，避免抖动。
- EMA 初始化为 baseline，随后 `ema = beta*old + (1-beta)*current`，beta=0.6；NLL 同样保存 EMA。所有 accuracy 在 [0,1]，百分点换算为 0.01。
- 定义缺口 `d=max(0, target_acc-ema_acc)`，默认 target_acc=0.8。已有得分达到目标不因另一组更高继续追涨。

### 内层：CLEVR 的 task×difficulty 桶

初始桶份额是候选池的原始频率。EMA accuracy、NLL 和门控与外层复用；控制 val 桶 n<20 不调整。图文和文字 replay 内部第一版固定均匀抽样，不再嵌套更多控制器。

更新目标权重后混入 20% 原始桶分布，保证覆盖；单轮桶权重倍率限制在 [0.8,1.5]。门控禁止上调的桶不能因归一化间接变大：固定其当前份额，只对允许调整的剩余桶分配剩余质量。没有可增权桶则保持原配方。

### EMA、指数更新与退火

设第一次 R1→R2 更新编号 r=1：`T_r=0.7**(r-1)`，`eta_r=eta0*T_r`，eta0=1.0。T 是退火系数，不是 softmax 温度；不使用 score/T，否则降低温度反而可能让配比更激进。

对允许增权的组：`q'_g ∝ q_g * exp(eta_r*d_g)`；其他组的指数分数为 0。外层只给最低准确率组非零分数，内层给有效低分桶非零分数；q 在各自层内归一化。外层重新乘 0.95，文字恢复为 0.05，然后施加范围和单轮步长约束；内层混合原始分布并施加倍率/固定份额约束。约束冲突时保持旧配方并报告原因，不启动通用优化器。

外层最大变动也随 T 缩小：`max_delta_r=0.05*T_r`；内层倍率上限随 T 收紧为 `1+0.5*T_r`。这样不只名义温度下降，实际调整上限也下降。

### 边际收益门控

使用同一控制 val 的 EMA 变化定义观测收益：

- `acc_gain=ema_acc_now-ema_acc_prev`；
- `loss_gain=(ema_nll_prev-ema_nll_now)/max(ema_nll_prev,1e-8)`；
- 初值 `eps_acc=0.005`（0.5 个百分点）、`eps_loss=0.01`（1% 相对 NLL 下降），只作工程启发式，不作显著性检验。
- 缺口大且 loss_gain≥eps_loss：允许继续增权；没有明显 NLL 下降则先 hold，不靠高 loss 无限制加数据。baseline/历史缺失时给最多一次受步长限制的探索，不填造收益。
- 仅当本轮该来源/桶相对上一轮确实增配并增加实际曝光，且 acc_gain<eps_acc **同时** loss_gain<eps_loss，才把 low_gain_streak 加 1；有明显收益则清零。没有增配的轮次不提供“增配无效”证据，计数不增加。
- 连续两次实际增配后收益都低：标记 `PAUSED_LOW_GAIN`，停止继续提高，输出待检查原因：答案/评分格式、数据质量、已掌握、模型容量或冻结策略。首版只报告并保持，不自动调用清洗模型或解冻视觉模块。
- 门控状态跨轮写入 controller_state.json；不允许归一化绕过禁增权。一次 loss 上升先 hold，首版不自动回滚/降权；所有变动有 reason。

这是“增配后观察到的收益”，不是增加该组数据的因果收益；各组共享模型且相互影响。第一版无额外探测训练，因果判断以后靠静态对照补足。

两轮闭环只有一次实际使用新配方的训练，不能真实验证全部两次低收益分支；R2 结束可产生 R3 recipe，但默认不继续训练。用模拟指标覆盖门控状态和退火的分支，后续增加 R3/R4 才积累真实证据。

控制流程示意：

```python
metrics = load_fixed_val_and_probe_results()
state = update_ema_and_observed_gain(metrics, old_state, actual_exposure)
gates = check_min_n_target_loss_gain_and_low_gain_streak(state)
source_ratio = update_two_vl_sources(old_recipe, state, gates)  # text=5%
clevr_ratio = update_clevr_buckets(old_recipe, state, gates)
validate_sum_bounds_step_caps_and_paused_group_nonincrease()
save_next_recipe_state_and_diff()  # 不在这里启动训练
```

recipe/state 至少保存：round、来源/桶份额、seed、EMA acc/NLL、目标、有效 n、上轮份额及实际曝光、low_gain_streak、paused、eta/T、原因；旧状态输入新状态输出，保留每轮文件，不只覆盖 latest。

物化规则：global batch=32、steps=300 时，每轮计划 9,600 次样本曝光。首轮整数配额 7,680 / 1,440 / 480；下一轮按最大余数法分配来源及桶配额，固定 seed 抽样后全局打乱。总步数与有效 batch 不变，比较实际图片/有效 token 成本；不声称同时固定 token 预算。

每条样本每轮最多重复 4 次；候选池不足以满足某桶配额则先在同源未满上限的桶重新分配、报告请求/实际份额，再检查总量。来源配额仍无法满足时在训练前报错，请补候选数据或降低 steps；不能无限重复或把缺少的文字回放悄悄补成 CLEVR。controller 下一轮用实际份额而非只看请求份额。

## MMStar：通用视觉能力退化检查

- 数据：官方 `Lin-Chen/MMStar`，HF 配置/划分叫 val，共 1,500 条、6 个核心能力，每类 250 条。这里视为外部保留评估集，名字 val 不代表可拿来训练或调配方。
- 第一版只跑 base 和锁定后的最终 R2 全量；不把 MMStar 得分送入控制器、不用它决定 checkpoint、配比或阈值。若以后需要据此选模型，另划 development 和 untouched test 并明确报告。
- 复用 `evaluate.py` 的 Qwen+PEFT 加载和推理；固定选择题 prompt，要求只输出选项字母，temperature=0/do_sample=False、短输出；按选项答案 exact match 计分。参考 VLMEvalKit 的 MMStar 数据/提示和解析，但首版不增加整套框架或外部 LLM judge。
- 保存总体、6 类准确率和逐样本 base/final 对齐结果，报告 `delta=final-base`（百分点）、正确→错误/错误→正确计数、无法解析/推理失败数。无法解析计错误、不从分母删除；多选项或歧义不让 judge 猜。
- MMStar 只覆盖部分通用视觉能力，不能证明所有能力或文字能力都保持；文字 replay 固定留出约 100 条控制集，只记录 answer NLL 变化，不扩展新 benchmark。
- 对入选 replay 与 MMStar 能得到的图像/文本做精确重叠检查，有重叠从训练候选移除、保留 benchmark。只声明本项目 exact 去污染，不保证 Qwen 预训练或近重复零污染。
- 调参完成后看见 MMStar 下降，只报告退化和下一版研究方向；不能反复据此重训又把同一分数称为无污染最终结果。

## 多卡策略

单卡先通过 model smoke，再短测 1/2/4 卡。建议训练使用 DDP；每卡加载完整冻结权重，只同步 LoRA 梯度，不进行 ZeRO-3/TP 全模型通信。4B 的 BF16 权重约 8GB 只是参数内存量级，不包含视觉参数细节、激活和其他开销；单卡 24GB 是否装得下仍须实测，DDP 不会把权重分摊成四份。

RTX 4090 没有 NVLink；PCIe 链路、NUMA、P2P 和 CPU 图像处理会影响吞吐。不能事先宣称四卡一定快或一定慢。

- 拓扑：`nvidia-smi topo -m`、`nvidia-smi -L`。
- 比较固定 global batch=32、每卡 micro-batch=2：1卡 accum=16；2卡 accum=8；4卡 accum=4。若 4B+图像 micro-batch=2 OOM，所有卡数统一改 micro-batch=1、accum=32/16/8，以保持公平。
- 各测至少 60 steps，排除前 10 steps 初始化/预热，固定同一数据、图片预算和长度；报告 samples/s、tokens/s、optimizer-step 时间和 peak GPU memory。
- DDP 的 all-reduce 仅针对可训练梯度；理论环形流量每卡约 `2*(N-1)/N*S`，S 为实际梯度总字节数。参数/梯度 dtype 以实际运行检查为准。
- 采用实际训练加固定 val 的总耗时最短方案。4卡若收益有限，可用1/2卡训练；验证时按样本分片并行推理，保持预测集不变。
- A 飞轮前后 round 有依赖，不能并行训练 R1/R2；静态对照可独立运行。

默认 BF16、gradient checkpointing、图片像素上限和短答案输出；初始 sequence length=2048、learning rate=1e-4，检查 replay 答案截断并排除完全无监督 token 的样本。像素/token 上限用安装版本支持的参数明确记录，不猜 CLI 名称。FlashAttention 只有现成可用时开启，避免第一版耗在编译。推理/训练遵守 Qwen processor 的图片处理方式，不复制 MiniMind-V 图片预处理。

## 待实现目录

```text
qwen/
├── README.md                    # 当前文件
├── configs/mvp.yaml             # 4B、80/15/5、路径、预算、控制器初值
├── src/prepare_data.py          # CLEVR/MiniMind 图文与文字→统一格式/split
├── src/materialize_recipe.py    # 两层配额抽样、repeat cap、曝光统计
├── src/evaluate.py              # 控制 val/probe NLL+EM、MMStar pre/post
├── src/update_recipe.py         # 动态、EMA、退火、收益门控、recipe/state/diff
├── scripts/train.sh             # ms-swift LoRA，单卡/DDP
└── scripts/run_round.sh         # 物化→训练→评估→更新
```

大数据、adapter 和运行结果放服务器 `/data/${USER}/vlm_learning/qwen_mvp/`，在其下组织 `manifests/`、`replay/images/`、`eval/mmstar/`、`runs/base/`、`runs/r1/`、`runs/r2/`。不拷贝 CLEVR 图片，manifest 引用原位置；MiniMind 内嵌图片只导出入选部分。JSONL+小 JSON 足够，不引入数据库/服务。

每轮最低产物：train.jsonl、exposure.json、adapter/、predictions.jsonl（含 sample_id/checkpoint/source/bucket/split/prediction/answer/nll_sum/answer_tokens）、metrics.json、next_recipe.json、controller_state.json、recipe_diff.json。只有实际实现后才给对应 CLI，不把示意文件名当成已可运行命令。

Round 1→2 先采用 continuation：加载上一轮 adapter 权重，新的 recipe 启动新 optimizer/scheduler，明确记录 optimizer reset。真正的精确 resume 包括 optimizer/scheduler/sampler 状态；换数据配方不能冒充精确 resume。第一版用固定两轮揭示接口问题，最终比较补足静态对照。

## 一晚上实现的顺序与验收

1. **配置+数据（约 1～2 小时）**：确认服务器环境和 MiniMind 文件；创建 mvp.yaml、prepare_data.py，验收三个池、图片/纯文本格式、无 train-val 精确重叠。数据不可用则先报告，不下载全部新项目。
2. **模型+训练（约 1 小时）**：创建 train.sh，单卡跑 10 steps、重载 adapter；检查 trainable 参数只在 LLM LoRA。训练 smoke 时间不计入编码估时；随后用现成 DDP 路径测多卡。
3. **评估（约 1～2 小时）**：创建 evaluate.py；各模态几条验证生成/NLL，检查 mask、有效 token 数、base/adapter 加载、MMStar 字母解析。真实短答案不足走显式降级，不编造 accuracy。
4. **飞轮（约 1～2 小时）**：创建 materialize_recipe.py、update_recipe.py、run_round.sh；用模拟 metrics 验证更新和门控，再以每轮 10 steps 做完整 smoke，正式预算恢复 300+300。

时间是预估，不保证依赖安装、数据读取和 API 适配都无阻碍。首版优先通过以下五个验收点：

- 首轮 9,600 曝光的来源配比为 80/15/5；图片路径可读，文字无占位图。
- 模型训练与 adapter 重载成功；prompt/image/pad 不进入 answer loss。
- 模拟低分且可学习组会增权；高分组、样本不足组、低收益暂停组不会误增权；所有份额和为 1，文字始终 0.05。
- 模拟连续两次增配低收益触发 PAUSED_LOW_GAIN；退火后更新幅度更小；指标缺失或非有限时保持/报错而非写 NaN recipe。
- 一次 smoke 完成 prepare→materialize→train→eval→update→下一轮；MMStar pre/post 输出存在，但不进入 feedback。

后续再补静态同预算对照、真实近重复清洗、复杂课程/逐样本重采样和更完整能力评估；第一晚不做。

## 参考

- Qwen3-VL 官方型号与使用：https://github.com/QwenLM/Qwen3-VL
- ms-swift 模型、LoRA、DDP 与混合模态训练支持：https://github.com/modelscope/ms-swift
- MiniMind-V 本地数据组织证据：`third_party/minimind-v/README.md` 的数据集章节及 `dataset/lm_dataset.py`；公开数据：https://huggingface.co/datasets/jingyaogong/minimind-v_dataset
- group-DRO 原实现及论述：https://github.com/kohpangwei/group_DRO （借鉴组增权，不移植旧训练环境）
- MMStar 官方数据卡/配置/规模：https://huggingface.co/datasets/Lin-Chen/MMStar ；官方代码：https://github.com/MMStar-Benchmark/MMStar
- VLMEvalKit 评估组织参考：https://github.com/open-compass/VLMEvalKit （首版不强制安装）
- 本仓库现有 `annotate_manifest.py`、`split_by_image_three_way.py`；保留它们的标签/split，数据适配代码独立讲解。

2026-10-04 已核对 Qwen 官方 4B-Instruct 型号、MiniMind-V 本地数据文档、MMStar 官方 HF 数据卡和 group-DRO README。训练 CLI 及冻结参数的精确写法，需要在服务器固定安装版本并查看对应 help/示例后再给出。此页是 Qwen 快速 MVP 的当前方案，不覆盖已有 MiniMind A/B 完整规划。
