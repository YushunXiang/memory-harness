# Memory Harness

面向 E-MAC 的最小、可插拔 memory program runtime。它不包含 benchmark、π0.5 backbone、数据切分或 controller；这些边界保持独立。

## 组件与执行顺序

每条 memory path 由 `encoder → writer → store → retriever → lifecycle` 组成，`controller` 选择本步激活的 path，输出统一交给 `utilizer`。每个 policy step 的顺序固定为：

```text
controller SELECT → lifecycle reset → RETRIEVE(previous history) → USE → WRITE_DECISION → optional WRITE(current moment)
```

因此当前 observation 不会在同一步作为“历史”回流。每次运行可输出 `RESET / RETRIEVE / USE / WRITE_DECISION / WRITE` JSONL trace，主动写入模块的跳过理由也可审计。

`WriteDecision.write_step` 支持 causal delayed write：writer 可以在当前步确认一个先前候选，并把当时的 token、robot state、phase 和 metadata 作为原始 payload 写入；未来步、跨 episode payload 和 privileged payload 会被 runtime 拒绝。`WRITE` trace 同时记录确认步、`source_step_index` 与延迟，writer-local buffer 在 episode/phase lifecycle reset 时同步清空。

`controller` 只决定本步哪些 path 执行 retrieve/write；所有 path 的 lifecycle 仍随每个 environment step 推进。这样某条 path 即使在整个中间 phase 被关闭，之后重新启用时也不会带回跨 phase 的陈旧内容。

`mem0_context` 不再根据 path 名称猜测 token 角色；每个 program 必须显式配置 `anchor_path` 与 `history_path_quotas`。多个 history path 共享固定窗口时，各自预算必须显式声明且总和等于窗口大小；trace 只把真正进入 policy context 的 item 记为 `USE`。因此 Agent 可以自由命名、替换或组合 store，而不会发生静默覆盖。
构建 program 时会检查这些角色必须恰好覆盖所有声明 path；漏映射或拼错名称的候选会在 rollout 前失败，而不会静默运行成“写了但没有用”的假架构。

已生成的 context manifest 会绑定 program config 哈希。若之后只把旧的隐式路径角色迁移为显式 `anchor_path/history_path_quotas`，必须先用 `memory_harness.audit_program_migration` 做旧/新配置的逐 token 等价回放，并把审计文件交给训练 finalizer；其他哈希漂移不允许绕过。这样 runtime 可以继续演进，而已冻结实验不会被静默改义。

## 固定程序

- `fixed_none.json`：不写、不读、不复制 observation；用于 action parity。
- `fixed_anchor.json`：当前 subtask 的第一个 moment 固定保留；phase 变化时 reset。
- `fixed_sliding.json`：Mem-0 的 30-step ring，每步保存一个最终层 contextual image latent。
- `fixed_anchor_sliding.json`：anchor 与 sliding 分库存储，按固定 `1 + 30` slot 布局输出。
- `fixed_consolidating.json`：MemoryVLA-style bounded store；超容量时合并余弦相似度最高的相邻 latent，而不是 FIFO 丢弃最旧项。
- `fixed_novelty_sliding.json`：保持相同 ring/retrieve/utilize，只在当前 latent 相对最后保留项足够新颖或达到最大写入间隔时写入；这是 learned write controller 的低成本基线。
- `fixed_kinematic_event.json`：`anchor + causal_kinematic_peak + ring`；用 joint-motion slowdown 延迟确认并回写候选时刻。其 `motion-window / peak-window / refractory = 30/20/8` 对齐 KEMO 论文的 Cover Blocks 表值，但论文未定义 online peak window 的端点与 tie 规则，项目页代码仍标为 coming soon，因此本实现只称算法完全显式的 inspired baseline；冻结 DINOv2、共享阈值 `0.05` 的 visual dedup，以及 16-patch SigLIP cross-attention/gated fusion仍是后续独立层。
- `fixed_dhem_event.json`：DiM-WAM DHEM 的单 bank、无学习维护算子；固定最初与最新事件，在容量满时比较“新事件—latest”与中间历史最冗余相邻对，选择丢弃新事件或累计质量加权合并历史。
- `fixed_content_recency.json`：episode 级 contextual-latent bank；用当前 latent 的余弦相似度减 frame-gap penalty 做 top-30 检索，再按时间顺序填入相同 Mem-0 slots。它受 TempoFit 的 content+recency 原则启发，但不是其 layer-wise K/V 复现。
- `fixed_semantic_recent_union.json`：OnEvoMemory 启发的无训练 retrieve 消融；固定保留最新 10 项，再从 semantic ranking 选取 20 个不重复项，按时间顺序填入相同 30 个 slots。它把论文的 hard-union 改成预算匹配的 disjoint quota，并记录去重前重合量；不冒充 learned value writer 或 online evolution。
- `fixed_temporal_multiscale.json`：CycleManip 启发的无训练 retrieve lower bound；至多 15 个预算按指数时间间隔覆盖历史，其余预算用全局均匀采样补齐，仍按时间顺序填入相同 30 个 raw-token slots。它只隔离多尺度采样，不冒充论文中的全历史 joint-state encoder、progress objective 或联合训练。
- `fixed_uniform_global.json`：RoboMME FrameSamp 启发的无训练 retrieve lower bound；在完整 causal history 上均匀选取 30 个 raw contextual tokens。它是 `temporal_multiscale` 全局分支的必要消融，不冒充 MME-VLA 的 16-patch/frame representation 或 learned AdaLN modulator。
- `fixed_recent_global.json`：第一个真正由两个 history path 组成的预算匹配 program；15 个 slot 保存连续 recent tail，另 15 个 slot 在排除该 tail 后均匀覆盖旧历史，避免两个模块静默覆盖或重复消费同一时刻。
- `fixed_tiered_chunk_mean.json`：MemoAct 启发的两级 store lower bound；最近 6 条无损保存，最老 3 条均值压缩后迁入容量 8 的长期区，长期区满时相邻合并。
- `fixed_completed_phase_handoff.json`：WeaveLA 启发的 M(n) 透明下界；按连续 phase 分段，只把最近一个已完成 segment 的逐步 contextual latent 均值压成 1 个 token，交给下一 phase。当前 architecture 与 `planner_no_key` 共用 Mem-0 planner 和 RMBench oracle prompt-change 边界，以隔离 executor handoff；另有 `key_completed_phase_handoff` 检验它与 planner-side key 的组合。重复出现的同名 phase 不会跨段合并；它不冒充论文的 8-query learned pooling、action-expert AdaRMS injection 或可部署边界检测器。
- `key_planner.py`：忠实 planner-side key memory，保存 episode 初始 RGB，以及按顺序排列的“已完成 subtask 文本 + 结束 RGB”；它不走 action latent 注入。
- `planner_no_key`：忠实 Mem-0 消融，每个阶段只读取当前 RGB；它不是“清空 key history 后继续看初始图”，并使用单独训练的 planner checkpoint。
- `configs/architectures/`：统一组合入口；`none / anchor / sliding / anchor+sliding / consolidating / novelty_sliding / dhem_event / kinematic_event / content_recency / semantic_recent_union / temporal_multiscale / uniform_global / recent_global / tiered_chunk_mean / completed_phase_handoff / planner_no_key / key / key+completed-phase / key+anchor+sliding` 都由同一个 `MemoryArchitecture` facade 构建。

planner 与 executor 共享 architecture/controller facade，但保持不同 typed target；RGB+text key history 不会被伪装成 action latent。

每个 `USE` 事件记录写入前总 store item 数，每个 `WRITE` 事件记录该 path 与全 program 写入后的 item 数；validated run manifest 汇总 `max_stored_memory_items` 与 `max_stored_items_by_path`。因此固定 30-token utilization budget 不会掩盖 full-history store 的额外容量，Agent 排名可以同时使用 success、latency、写入次数、注入 token 和实际 peak storage。

固定模块的 paired comparison 后必须再经过 `memory_harness.utility_gate`。该工具把证据明确分成 `<20` 的 screen、`20–49` 的 pilot 和 `>=50` 的 confirmation；success 是唯一 primary endpoint，`max_reward/total_reward` 只能作为追加实验的阶段信号。单一 candidate 即使通过 confirmation，也只满足 Gate 1 的 utility 子条件；oracle、内容干预和 `M(0)-control` 仍须单独验证。示例：

```bash
python -m memory_harness.utility_gate \
  --comparison /tmp/none_vs_sliding.json \
  --evidence-kind fixed_ablation \
  --output /tmp/none_vs_sliding.utility.json
```

研究 Agent 不需要解析 Python 源码来猜可用组件。以下命令导出当前真实 registry、path/program contract、已有组件的语义说明，以及每个构造器的必填和可选参数；该清单直接由 executable registry 与实现 docstring 生成，不维护第二份手写能力表：

```bash
python -m memory_harness.catalog --output /tmp/memory_component_catalog.json
```

Agent 生成候选配置后，必须先通过任意-config smoke。该入口会执行真实 registry/build/runtime，覆盖足以触发 delayed writer 的时域、phase change 和第二 episode reset，并把 config hash、event counts、最大 token/store 用量与 reset 隔离结果写入机器可读 summary：

```bash
python -m memory_harness.smoke \
  --config /path/to/candidate.json \
  --output-dir /tmp/candidate-smoke
```

单候选的 Agent 提交使用更严格的原子入口。submission 目录只能包含
`submission.json / architecture.json / executor_program.json`，architecture 必须引用同目录
`executor_program.json`；任何 Python、额外文件、symlink、路径逃逸、非 deployable program、未知
搜索轴或重复内容都会在 rollout 前拒绝。artifact 冻结 hypothesis、parent content hash、typed search
axes、完整 runtime snapshot、真实 smoke trace 和 path-independent behavior SHA-256；改候选名字或文字标签
不会逃过内容去重。Gate 1 前所有 artifact 明确保持 inactive：

```bash
python -m memory_harness.search_candidate create \
  --submission /path/to/submission \
  --output-dir /path/to/immutable-candidate
python -m memory_harness.search_candidate validate \
  --candidate /path/to/immutable-candidate
```

在花费 rollout 预算前，再用真实 validation context 对所有同 payload-family 的固定 program 做最终输入指纹去重。该审计比较实际送入 policy 的 `memory_tokens + memory_mask`，而不是配置名字；v3 从 context manifest 恢复与 source latent 对齐的真实 `phase_label`，因此 phase-handoff 候选不会被单一假 phase 静默测成空模块。缺少 robot state 或 ordered episode outcome 的候选会显式列入 exclusion：

```bash
python -m memory_harness.profile_candidate_distinctness \
  --manifest ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_manifest.json \
  --context-bank ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_bank.npz \
  --config-dir configs \
  --output artifacts/2026-08-15-candidate-distinctness/summary.json
```

一轮候选确定后，用单命令自动发现全部 `fixed_*.json` architecture，构建并 smoke 每个 executor，再把 runtime、完整 config tree、catalog、逐候选 summary 和全部 SHA-256 冻结为不可变 suite：

```bash
python -m memory_harness.candidate_suite create \
  --output-dir artifacts/2026-08-16-candidate-suite-v9
python -m memory_harness.candidate_suite validate \
  --suite artifacts/2026-08-16-candidate-suite-v9
python -m memory_harness.candidate_suite validate-checkpoint \
  --suite artifacts/2026-08-16-candidate-suite-v9 \
  --checkpoint /path/to/mem0/checkpoint
```

同一轮 paired rollout 必须共用该 suite；runner 会先完整校验 suite，再复制其 runtime/config/manifest 到 run artifact。已有 checkpoint 只提供权重，不决定本轮候选集合：

```bash
CHECKPOINT_DIR=/path/to/mem0/checkpoint \
MEMORY_CANDIDATE_SUITE=$PWD/artifacts/2026-08-16-candidate-suite-v9 \
bash scripts/run_fixed_pi05_rmbench.sh put_back_block recent_global
```

runner 首先用 checkpoint 自带的训练 manifest/config snapshot 对全部候选做 CPU preflight，随后 policy wrapper 在第一次构造时再次比较 program 的 `(token_budget, embed_dim)` 与实际加载的 Mem-0 fusion shape；不匹配直接失败，不允许等到 rollout 中途才由 JAX shape error 暴露。当前 v9 的 21 个候选全部完成 build/smoke 并通过 Put Back full-memory checkpoint 的 `[31, 2048]` preflight，且冻结了 phase-aware distinctness v3；manifest SHA-256 为 `00b7400c12a7e5f663417aa5896725e8085ce1e7871efc112e0714e4b5323bf7`。其中 `completed_phase_handoff` 与 `planner_no_key` 共用 no-key planner，`key_completed_phase_handoff` 与 `key` 共用 key planner，保证 executor handoff 的比较不混入 planner 类型变化；两者当前都依赖 RMBench oracle prompt-change，仅作为诊断性 M(n) 下界。v6 的 14 个 latent-only、episode-local executor 在 10 条 Put Back validation episode、323 个 sampled queries 上不存在最终 memory context 完全等价的候选对；该 handoff 在单一全局 phase 的 Put Back 上按定义为空，因此只在有真实 phase transitions 的 M(n) trace 上做后续去重和 rollout，不能用 M(1) 的空输出把它误判为 `none`。Jaccard `>0.95` 的唯一既有 review pair 是 `anchor_sliding / sliding`，因其用于隔离 anchor 的必要嵌套消融而保留，但不计作两种独立新架构。`kinematic_event` 与 `verified_success_latent` 因分别需要 robot state 和 ordered outcome protocol 而不做伪造比较。旧 suite 保持不可变，只用于复核已绑定的 run；当前运行中的实验继续使用其启动时冻结的 snapshot，不被 v9 静默替换。

Put Back readiness 的 u1200 与 u3000 分支现在都能无人值守接到 `none / anchor / sliding / anchor+sliding` Gate-20。只有至少一个 fixed pilot 的 success 或两个阶段指标呈正向，才会调用 `continue_put_back_candidate_screen.sh`。候选、参照、任务、共享 seeds 和 evidence kind 全部由 `configs/screens/put_back_fixed_v9_screen3.json` 声明，并在执行前对 v9 suite 的可用 architecture alias 做校验；runner 不再硬编码候选集合。当前 plan 在同一 full-memory checkpoint、3 个新 seeds 下，以 sliding（kinematic-event 使用 anchor+sliding）为参照筛选 novelty、DHEM、kinematic-event、content/recency、semantic+recent、uniform/recent-global、multiscale、boundary-chunk 和 tiered-chunk。该轮标记为 zero-shot compatibility screen；正向结果仍需自己的 matched context training，脚本不会直接训练 controller。

截至 2026-08-16，u1200 `full-memory / empty-mask / native-none` 已完成**总 optimizer exposure 匹配**的共享 Gate-3：三路均为 `0/3` complete success、`max_reward=0`。其中 full-memory 的条件 schedule 是 `none 200 → anchor+sliding 1000`，empty/native 则从相同 base 分别以自己的条件训练 1200；因此该结果可作 executor readiness screen 和同一 full checkpoint 的 memory mask 消融，但不能称为三路都从初始参数按各自条件训练。comparison v3 现逐级校验并报告 chronological program schedule、每种 program exposure、precondition updates 与 evidence scope，readiness 也拒绝共享 run provenance 不一致。源码审计另发现 Put Back 只在完整放回后把 reward 置为 1，移到中心和按按钮没有 dense reward；因此旧 readiness 只能证明没有完整成功，不能证明零部分进展。runner 现直接记录 `0/1/2/3 = 无进展/到中心/按按钮/放回` 的 task progress，并把它作为 screening endpoint；success 仍是主指标，progress 不能单独放行 controller。修正审计见 `artifacts/2026-08-16-put-back-sparse-reward-progress-audit.json`。累计 u3000 学习曲线继续按 `native → full → empty` 运行；full 分支保持 `none 200 → anchor+sliding 2800`，完成后先按新指标决定 fixed Gate-20，若需确认三路训练条件效应，再从同一 base 启动 clean schedule，而不改写本轮证据范围。

旧 u1200 rollout 已保留每一步实际送入 RMBench 的 action。`memory_harness.replay_put_back_progress` 会校验原始 `config / episodes / action_stats` 的完整性与 SHA-256，在同一 simulator seed 下只回放动作、不加载 VLA，并用同一 `0/1/2/3` contract 恢复阶段进度；回放 success 与原记录不一致时拒绝产出。该回放已排在 u3000 链之后、Cover Blocks 之前运行，避免与训练争抢 GPU。它只补齐旧轨迹的观测证据，不替代 u3000 的前瞻评测。

## 本地验证

从本目录运行：

```bash
../../openpi-libero/.venv/bin/python -m pytest -q
../../openpi-libero/.venv/bin/python -m memory_harness.smoke \
  --output-dir artifacts/smoke
```

真实 RMBench baseline 使用 `CHECKPOINT_DIR=/path/to/audited/checkpoint scripts/run_clean_pi05_rmbench.sh <task>`；fixed pilot 使用 `CHECKPOINT_DIR=/path/to/audited/checkpoint scripts/run_fixed_pi05_rmbench.sh <task> <architecture>`，architecture 名直接对应所选 snapshot 中的 `configs/architectures/fixed_<architecture>.json`，不维护另一份硬编码列表。task spec 不保存 baseline/memory checkpoint 默认值，两个 runner 都要求显式绑定本次实验 checkpoint；这样过时 checkpoint 不会在新 lineage 中被静默复用。fixed runner 默认加载 checkpoint 随训练冻结的 suite；Agent 新候选通过 `MEMORY_CANDIDATE_SUITE` 显式选择独立 immutable suite，不能分别覆盖 runtime/config。task、数据、prompt、最大步数和任务专属 layout protocol 均来自 suite；runner 显式固定 checkpoint、`policy_adapt_to_pi=false` 和 policy seed。比较器要求 candidate-suite、runtime、task config 以及每一侧 architecture/executor config 的哈希在 successive shards 中一致。planner 当前使用 oracle phase transition 作为诊断边界，manifest 会标记为不可部署。

baseline 分为两种，不能混称：`fixed_none` 是 memory-capable checkpoint 的全空 mask，用于同 checkpoint 模块消融；真正的原生无 memory π0.5 用 `scripts/run_pi05_baseline_train.sh <task>` 训练、再用 `scripts/run_clean_pi05_rmbench.sh <task>` 评测。前者控制 memory 内容，后者回答完整 memory architecture 是否优于普通 π0.5。

M(n) executor 在进入 planner/memory 对照前，可用显式 oracle-subtask diagnostic 检查六个
低层技能是否已经形成：

```bash
CHECKPOINT_DIR=/path/to/corrected/cover-blocks/native/checkpoint \
ORACLE_SUBTASK_DIAGNOSTIC=1 \
scripts/run_clean_pi05_rmbench.sh cover_blocks
```

该模式把 `prompt_protocol` 固定为 `diagnostic_spatial`，manifest 标记
`condition=oracle_subtask_pi05_none / deployable=false / evidence_scope=executor_skill_diagnostic_only`。
默认模式仍要求 `phase_aware_subtask_prompt=false`，不能把 oracle 结果混入 deployment baseline。
修正后的 Cover Blocks native u1200 训练、Gate-3 rollout 和 signal decision 已封装为单一
fail-fast 入口；它先校验冻结 template SHA，再从 π0.5 base 训练，不读取历史 `9999`：

```bash
bash scripts/run_cover_blocks_corrected_executor_gate.sh
```

若 diagnostic 出现 success 或阶段 reward，decision 仅放行
`train_budget_matched_memory_executor`；若仍全零，则选择
`increase_native_executor_training_budget`。executor floor 建立后，continuation 从同一 base 训练
预算匹配的 empty-memory executor，并在相同 checkpoint 上比较 `key / no_key`：先跑共享
Gate-3，再以不重叠 seeds 自动补到 `3+17=20`；只有 pilot 呈正向 success 或阶段信号时才继续
`+30=50` confirmation。所有 shard 由比较器拒绝重复 seed/layout evidence；任何分支都不会直接
放行 controller。

两个 run 完成后，用严格配对比较器检查 checkpoint、seed、layout、policy RNG 和其余受保护变量，再计算逐 episode delta：

```bash
../../openpi-libero/.venv/bin/python -m memory_harness.compare_fixed_runs \
  --reference-run ../rmbench_runs/reference \
  --candidate-run ../rmbench_runs/candidate \
  --output ../rmbench_runs/comparison.json
```

不同训练方案的 checkpoint 不能使用上面的同-checkpoint消融比较器。纯 none 与 full-memory 的方法级比较使用独立入口；它会沿 `initial_weight_params` 递归审计训练链，并要求总 `optimizer_updates × effective_batch`、初始 base params、task config 和 rollout seeds 全部匹配：

```bash
PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.compare_training_runs \
  --reference-run ../rmbench_runs/none_budget_matched \
  --candidate-run ../rmbench_runs/full_anchor_sliding \
  --output ../rmbench_runs/training_variant_comparison.json
```

2026-08-14 早期 3-episode pilot 使用的是已废止的 shared adapter，只保留作历史诊断；它不再代表当前 Mem-0 port 的项目状态。

状态语义只有两层：项目主线保持 `ACTIVE / reproduction in progress`；实验门槛只控制 `controller eligibility`。executor 或固定模块尚未通过门槛时，继续训练、复现和诊断，不把整个项目标记为 HOLD。

当前“可插拔”声明有明确边界：21 个 frozen candidate 的 executor 路径已经全部通过同一
`encode / write / store / retrieve / lifecycle / utilize / controller` registry 与配置 facade；
planner 路径目前只能在配置中选择 `none/key`，内部仍直接构造 `KeyPlannerMemory`、固定
`Mem0PlannerPolicy`，且 boundary 只有非部署的 oracle diagnostic。因此现在不能声称 Agent 已能
自由创造或重组 planner-side memory。机器可读证据和无兼容层的下一 schema 替换条件见
[`artifacts/2026-08-16-architecture-composability-audit.json`](artifacts/2026-08-16-architecture-composability-audit.json)。
活动 Put Back/Cover Blocks 链的最终 validator 仍优先导入 live harness；为避免中途破坏其冻结的
schema v3，planner registry 替换必须等该链完成并通过校验后一次性进行。

ALMA 是当前与最终目标最直接重叠的公开基线：Meta Agent 已能根据 parent code、成功/失败
trajectory 和 score 生成、调试并归档新的 text-memory Python。其论文 GPT-5-nano/mini 总体
success 为 `12.3/53.9%`，分别比 no-memory 高 `6.2/12.8 pp`；ALFWorld 的 greedy/ALMA
为 `11.9/12.4%` 与 `77.1/87.1%`。本项目因此不再声称“首次让 Agent 设计 memory”，而把
贡献收紧为 fixed-VLA 上的 typed multimodal memory-only search、预算匹配、locked transfer 与
memory-content intervention。`memory_harness.search_archive` 已实现 content-addressed 的
ALMA-compatible visit-penalized sampler，供 Gate 1 后做强基线；它当前不激活，也不是 policy
memory plugin。官方 source/released archive 的完整审计见
[`artifacts/2026-08-16-alma-source-audit.json`](artifacts/2026-08-16-alma-source-audit.json)：原版候选
是任意 task-specialized Python，Docker bind mount 可写且未禁网，候选 ID 不是 content hash，
LLM/DB/context 成本也未作搜索时预算匹配，故只复用搜索思想，不复制其执行边界。

MemEvolve 是第二个必须保留的直接搜索基线，但不新增 policy memory。论文在 Flash-Searcher 的
GAIA/xBench/WebWalkerQA 上相对 no-memory 为 `+4.24/+5.00/+3.53 pp`，核心新价值是按
success、cost、latency 做 Pareto survival，并把选中架构冻结后跨任务迁移。`search_archive`
现已实现 content-addressed、确定性的 `ParetoRecord / pareto_ranks / select_pareto_survivors`；
Gate 1 前同样 inactive。官方 commit `6035d56…` 虽可 compile，但 release 默认关闭 Pareto，
survivor/task-count/tie-break 与论文不一致，后续轮仍以初始 provider 作代码模板；generated code
会先修改 live tree，再在带 secrets/network 的同一进程动态 import，且没有 core tests、paper runs
或独立 search seeds。因此不复制其 runtime，只采用经过收紧的 Pareto 与 frozen-transfer contract。
完整证据见
[`artifacts/2026-08-16-memevolve-source-audit.json`](artifacts/2026-08-16-memevolve-source-audit.json)。

EvoMem 补齐的是搜索侧“哪些 mutation 经验值得长期保存”的 writer gate，而不是另一种机器人
memory payload。`memory_harness.research_memory.LineageEvidencePromoter` 现以不可变 candidate/run
hash 接收单次 search lineage，分别记录机制首次引入相对最强 parent 的增量、同 parent siblings、
继承该机制的 descendants、downside rate 与独立重现，再输出可审计的 promote/reject decision。
所有阈值必须显式提供，因为论文没有发布 exact values；它只报告观察性 lineage evidence，不能将
分数变化归因给单个机制。论文称代码随 submission 提供，但公开 arXiv source 只有 manuscript/
figures，官方 GigaEvo commit `9b8687e…` 也没有 EvoMem runtime，所以其 LLM extraction、dedup 和
retrieval 不作假复现。该 Research-Agent 插件在 Gate 1 前 inactive，不进入 policy registry、
17-family source catalog 或 21-candidate suite。完整证据见
[`artifacts/2026-08-16-evomem-paper-release-audit.json`](artifacts/2026-08-16-evomem-paper-release-audit.json)。

EvolveMem 作为 Gate 1 后的 structured-config search 强基线保留，但不再新增同义插件。论文报告
LoCoMo `0.305→0.543`，相对 strongest baseline 高 `25.7%`，MemBench 为 `67.9%`，说明
failure-log-driven search 本身值得比较；然而官方 commit `db80b6a…` 的 action space、LoCoMo
surface flags、expected-lift recipes 和三项“新维度”均已写入源码，默认 LoCoMo runner 还会在第 5 轮
直接注入作者预写的 `evolved_config()`。meta new-dimension 只写 JSONL，不会实现或注册 operator；默认
elitist 路径也跳过 meta 参数建议，所谓随机探索实际确定性地选择首个未试 fusion mode。故本项目只用
现有 `search_candidate / search_archive / research_memory / utility_gate` 表达其有效模式，并强制独立
search/confirmation split、禁止 terminal-config 注入和 task/evaluator patch；不改变 17-family catalog、
21-candidate suite 或当前 Put Back 链。完整证据见
[`artifacts/2026-08-16-evolvemem-source-audit.json`](artifacts/2026-08-16-evolvemem-source-audit.json)。

论文源码已经审计、但尚未满足当前 runtime payload/training contract 的候选单独保存在 `configs/source_audited_candidates.json`。当前有 17 个互不重复的 typed family，包括 Chronos selective-SSM state、μVLA recurrent tokens、TFP continuous-time belief、NativeMEM action-supervised native visual tokens、OptimusVLA cross-episode action prior、VLA-Pro parameter memory、HALO observation/action K/V sparse retrieval、GMP error-calibrated read gate、RoboMME perceptual patch memory、G0.5/MEM bounded raw-frame video context、TRACE trajectory-addressed slots、AHA-WAM asynchronous planner K/V context、KC-VLA task/phase event keyframes、MemoryVLA dual-stream PCMB、BridgeVLA++ viewpoint-aligned canonical point-cloud anchor，SERF articulated robot–environment relational neural-point state，以及 PhysMem evidence-linked verified physical principles；每个 proposed operator 都声明 `inputs → output` typed edge。validator 同时拒绝重复 payload family，不能靠换名字登记同一种 memory。它们不会混入可执行 candidate suite；Research Agent 必须先满足各自 `requirements` 与 `entry_gate`，实现真实 typed operator 并通过 smoke/preflight 后，才能生成新的不可变 rollout suite。μVLA 额外要求 episode-ordered training、memory/action attention guard 和每环境步更新；TFP 把 visual/proprio encoder、elapsed-time LTC update、episode reset、action-head AdaLN 和 ordered TBPTT 拆成独立 operator，并要求 real/fixed `Δt` 与 concat/AdaLN 配对消融；NativeMEM 则把 action-supervised tokenizer、dense frame-view writer、bounded token queue、native prefix utilizer和 Stage1/cache/Stage2 recipe 分开，不能用 mean-pooled token 冒充；OptimusVLA GPM 被拆成 task embedding、跨 episode action bank、semantic top-k、可部署 progress alignment、flow initializer、adaptive NFE 与 episode session，必须排除 evaluation episode 并与 random/wrong-task/shuffled-progress prior 做相同 checkpoint 对照；其 LCM 与现有 Motion Tail 重叠，不登记第二个 family；GMP 要求先有 all-on/all-off 固定 policy、held-out error calibration、冻结 gate 和最终 policy 重训；RoboMME family 保留 `FrameSamp / prefix-causal RGB TokenDrop` 两个 patch retriever 和 `Context / action AdaLN / separate Expert` 三个 utilizer，形成可组合的 2×3 因子，而不是把整篇论文或现有 `uniform_global` 包装成一个总开关。G0.5/MEM family 则把 bounded raw-frame window、same-patch causal mixer、current-token compressor、native-prefix utilization、reset/start clamp 与 whole-history dropout 分开；公开 server 只在 action chunk recompute 写 history，因此未来 port 必须比较 policy-query cadence 与每控制步 cadence。BridgeVLA++ family 只保留不重复的初始 colored point cloud、当前 coarse waypoint 对齐重渲染、fine-stage per-view cross-attention 与 reset/joint-training contract；temporal anchor/recent/subgoal 继续复用既有 operator，不能重复计数。SERF family 不重复登记动态环境地图，只保留把 URDF 机器人表面与环境放入共享 neural-point latent state 的新 payload，以及 base 多尺度、双末端、robot-only、environment-only、global 五组可独立选择的 retriever；官方实现将这些分支硬编码为固定顺序，未来 port 必须将其拆成 typed branch 并分别做 matched training。KC-VLA 只把 progress-aware **写入**登记为新轴：官方读取端没有 dynamic top-k，而是最多五个历史关键帧加当前帧；移植必须显式区分 pending-immediate-use 与 confirmed-write，并把官方 oracle-phase 阈值校准改成 recursive-prefix calibration。真实 Put Back profile 显示 RoboMME 两个 selector 的 frame-set Jaccard 仅 `0.098`，同时发现 released full-episode TokenDrop 与 online heap 只有 `83.23%` query 完全一致；所以新训练强制 prefix-causal selector parity，原预计算只作 checkpoint compatibility ablation。可用以下命令验证审计文件、哈希、typed edges 和非执行状态：

PhysMem 只新增“带支持/反证和验证状态的跨 episode 物理原则”，复用已有 episode、transition、procedure 与 semantic retrieval；action-level attribution、targeted experiment、promotion/refutation 和 evidence folding 分别保留为可组合 operator。官方 synthetic quickstart 可运行，但源码的 bounded eviction、decay pruning、resume binding、主动验证 promotion 与 trigger-aware retrieval 和论文不一致，且没有 paper checkpoint/data/run。因此它保持 non-executable typed contract，修复 parity 并建立 deploy-visible action-level outcome 与 safe experiment API 前不进入 π0.5。

TRACE 现也以第十个非执行 family `trajectory_addressed_evidence_slots` 登记：执行 state 的真 path signature 地址、visual/state evidence、gated slot update、selective readout、policy adapter、episode reset 与 ordered causal training 分开声明。官方 `simple` fallback 不保序，deploy runtime 又使用无界全前缀重算；所以未来 port 强制 signatory fail-closed 与 bounded streaming address，不能把近似 moments 冒充 TRACE。

AHA-WAM 作为第十一个非执行 family `asynchronous_layerwise_planner_kv_context` 登记：慢 planner K/V FIFO、当前观测 OVCR、action-side joint attention、version/age/stale guard、episode reset 与 phase-offset training 分开声明。其 source-level 核心 router 和 lifecycle smoke 已通过，但 plain released checkpoint config 未启用 offset，且完整模块约 1.22B 参数；所以先做官方 checkpoint/config 复现与澄清，再决定 π0.5 最小 adapter，不加入当前 fixed suite。

MemoryVLA 作为第十二个非执行 family `dual_stream_perceptual_cognitive_memory` 登记：认知/感知 encoder、独立双 bank、timestep cross-retrieval、gate、相邻合并、两种 action utilization、reset 与 ordered joint training 均是独立 typed operator。官方源码与 33.5 GB checkpoint 的不可变配置已核验，核心 CPU smoke 通过；公开配置实际使用 raw-token write，而非论文文字中的 fused write。完整 memory path 约增加 `575.8M` 参数，且论文没有 matched all-memory-off 消融，因此先复现官方 checkpoint，再在 π0.5 上做 `none/cognitive/perceptual/dual` 同预算训练，不加入当前 fixed suite。

G0.5/MEM 作为第十四个非执行 family `bounded_multiview_raw_frame_window` 登记：官方 commit `89f2322…` 的 factorized temporal-spatial ViT、全历史 dropout、current-frame token drop、wall-clock dataset stride、episode frame buffer 与 reset 已逐项审计，CPU source smoke 证明历史输入会改变当前输出。它与 MEM/HyVLA 是同一 short-video family，不重复登记；论文也没有 matched memory-off，普通 release configs 仍为单帧，gated checkpoint sidecar 未能核验。因此它提供可实现源码契约，不提供当前 π0.5 上的免训练开关或独立收益结论。

只有论文、没有可审计源码的候选不会塞进上述 source-audited catalog。LaMem-VLA、VQ-Memory、HyMeS、RTCF、AtlasVLA、OnEvoMemory、RB-VLA 与 StemVLA 分别冻结为独立 `artifacts/*-audit.json`：前两者分别保留 action-hidden dual vault/native-prefix 与 deployment robot-state→离散 phase token→vocabulary prefix 的新增轴；HyMeS 只保留 symbolic task-state、multi-evidence verified update、rollout-driven program refinement 与 frozen-program selection 契约，并把 flow-gradient steering 标成单独扩展边界；RTCF 不新增重复的 action-trajectory payload，只把 causal monotonic alignment frontier、aligned future chunk 与 clipped low-frequency residual utilization 合并进既有 `trajectory_action_prior` family；AtlasVLA 只新增 calibrated streaming-depth voxel world-state family，ego bank 复用 MemoryVLA；OnEvoMemory 不新增 payload family，只保留 action-conditioned value encoder、elite/value-delta writers、三库 lifecycle 与 outcome-driven module update；RB-VLA 复用 continuous/recurrent belief payload，只新增 stochastic action-conditioned update、EMA multi-horizon predictive pretraining 与 inverse-dynamics grounding；StemVLA 复用 bounded video-history payload，只增加 VGGT geometry encoder、VideoFormer temporal aggregator 与 future-geometry distillation。八者都声明可复用 operator、真正新增 operator、matched factorial 和 release blocker，直到 source/训练契约可审计前保持 `paper_contract_only`；HyMeS 只作为 task-specific code 强参照，不与 memory-only typed search 混排。OnEvoMemory 源码中的容量、阈值、Top-K、merge/eviction 与 online loss 细节均位于未渲染 TeX 注释块，不能被当作已发布复现规范。

另有源码可审计、但不构成新 payload family 的训练方法单独冻结。`artifacts/2026-08-16-streaming-grpo-memory-source-audit.json` 审计了公开的 Streaming GRPO memory planner：它复用 MemER nomination/vote-cluster graph，只新增把最终 episode progress 延迟回传给早期离散 memory writes 的 group-relative training operator。核心 48 项 CPU tests 通过；但课程项目没有 checkpoint/raw logs/per-seed episodes，且 oracle 执行非 pick 阶段，因此只作为 Gate 1 后 learned-controller training reference，不计入 source-audited payload catalog，也不借用其 headline 作为全自主 π0.5 结果。

`artifacts/2026-08-16-hime-source-audit.json` 进一步冻结 HiMe 官方 commit `261766e…`：recent window、图文 key record 与 procedural plan 均复用现有 payload，故不增加重复 family；只保留 `Sentry completion → Planner wakeup` 和 Planner 生成 CRUD 两个独立 controller/lifecycle 轴。源码 CRUD、save/resume 和 FIFO smoke 可运行，但没有顶层 license/tests/executor assets，且存在 paper/source FIFO `8/20` 不一致、exact-tag Top-K 失效、默认无界 store、unknown policy fail-open、无 namespace/version/evidence/transaction 等风险。因此未来实现采用本项目自己的 typed、bounded、versioned、atomic contract，不复制源码或增加 `hime=true` 总开关。

`artifacts/2026-08-16-vermem-source-audit.json` 将 VerMem 去重为六个 controller/training contract：统一的 LTM/STM 原子操作策略、versioned update/soft-delete transaction、realized-transition local verifier、terminal-memory global verifier、operation-normalized hierarchical credit 和 `LTM→STM→joint` curriculum。generic CRUD/retrieve/filter/summary/episode recall 已被现有 operator 覆盖，故 source-audited family 仍为 17 个。论文 Qwen2.5-7B 平均由 no-verifier `41.76%` 到 full `48.01%`；但官方 source 没有有状态 memory runtime，发布 verifier 是 Python rule heuristic 而非论文的 frozen DeepSeek-V3.2，bundled reward loader 又让 local/global/credit 路径全部归零，且没有 checkpoint、训练数据或运行结果。因此不制造 executable 空壳；fixed π0.5 memory 通过 utility gate 后，再在本项目 typed transaction runtime 上把它实现为 operation-controller 强基线。

`artifacts/2026-08-16-mmpo-paper-release-audit.json` 为 recursive textual summary 保留 `progress+information-gap anchor → response-token entropy → outcome-anchored dense reward → future-aware turn advantage` 四个 evaluator/training contract。它不同于 VerMem 的 transition/terminal correctness verifier，也不同于只回传最终机器人结果的 Streaming GRPO：RULER-HQA 56K 上 outcome-only/direct-answer/gap-only/progress+gap 为 `80.47/78.17/82.02/82.98%`，直接奖励答案置信度反而退化，说明必须探测 progress 与 unresolved information 并保留 verified outcome anchor。该方法只在文本 QA/WebShop 验证，依赖 autoregressive token logits 和任务专属 probe，论文无 official code/checkpoint/run ledger；因此不新增 payload/source family或 executable，也不能拿它给 π0.5 latent memory 打分。只有文本 summary 候选通过 fixed utility gate 且 anchor 在 calibration split 校准后，Research Agent 才能选择该训练轴。

`artifacts/2026-08-16-signals-to-structure-paper-release-audit.json` 不新增 payload 或 executable，而是给 architecture search 增加三个评测约束：主表保持统一 token budget，Gate 1 后对入围架构做 capacity×history sensitivity；组合 memory 记录重复、过期、冲突与 revision；task utility 与 coverage/compression/structure 指标分报。论文的 rolling window、shared board、private scratchpad、slot codebook 和 slot+meta 均可由现有 operator 表达，但其 cap=25→64 的 scratchpad/memory-only 排序差异，以及 codebook+meta 的负干扰，说明不能把单一容量下的结果外推为架构固有优劣。该证据来自 symbolic signaling game、多数条件单 seed且无代码/数据，因此只改变后续评测协议，不扰动当前 Put Back/Cover Blocks 链。

`artifacts/2026-08-16-hierarchical-memory-theory-audit.json` 不增加第 18 个 payload family；它指出当前 hierarchy lower bound 的接口缺口：`AdjacentMergeStore` 把 adjacent partition 与 mean representative 融在一起，`TieredChunkMeanStore` 再绑定 temporal chunk 和 long-tier consolidation，`BoundaryChunkRetriever` 则把 query-time segmentation 与 traversal 融在一起。保留的设计契约是让 Agent 独立选择 `partition / representative / traversal`，并记录 parent-child/source provenance、self-sufficiency–traversal 校准以及 relevant-child 被低分 parent 错剪的诊断。论文只映射 11 个已有系统，没有 matched `representative × traversal` 实验或机器人结果，且静态 hierarchy 理论不覆盖在线 restructuring；因此当前不造新插件。活动训练链结束后再用 exact trace replay 拆除 fused hierarchy 配置，不保留兼容层。

`artifacts/2026-08-16-himem-wam-source-audit.json` 冻结 HiMem-WAM 官方 commit `cca217e…`、MIT license 与 12.0 GB LIBERO checkpoint metadata。论文的技能边界事件 token、boundary-supervised sparse writer 和 teacher-forced warmup 值得作为既有 `task_phase_keyframe_history / observation_action_kv` 上的训练轴，但 RMBench `10.8→26.3%` 是整套 WAM 对 π0.5，未提供同 backbone memory-off/Stage-III-off。源码也不能闭合该归因：简化 policy 没有 Qwen/三阶段 trainer，false gate 仍写零 slot；完整 Wan wrapper 用九个 task rule 写 memory，而公开 checkpoint 的文档 eval 直接实例化 base Wan、没有构造 wrapper。因此不增加重复 family 或 `himem_wam=true` 开关，只在 fixed π0.5 memory 通过后做 matched learned-writer training。

`artifacts/2026-08-16-worldscape-policy2-paper-release-audit.json` 新增一个真正不重复、但目前只能 paper-only 的 payload：`planner_reasoning_trace_event_history`。它按 action chunk 保存 VLM 感知 hidden states 与自回归 subgoal hidden states，并构造 global/latest/semantic-boundary/compact-full 四种视图；这不同于文本 key、RGB keyframe、raw video window 与只缓存当前 planner K/V。matched progressive ablation 从短时视觉 `44.67%` 加长期事件到 `46.25%`，再加 latent reasoning 到 `47.89%`。官方 GitHub 与 HF 尚无源码或权重，且关键 slot、window、boundary 和 mask 参数缺失，所以当前只冻结 typed encoder/write/store/retrieve/utilize/training/reset 契约；既有 source-audited family 与 21 个 executable candidates 均不变。

`artifacts/2026-08-16-mem-world-paper-release-audit.json` 只给既有 spatial/raw-frame family 增加 future-action-conditioned retrieval 轴：计划 action chunk 经 FK 形成未来腕部视角，timestamp/task-relevance surfel 逐时刻渲染并按可见性、目标相关性与新近性打分，temporal NMS 后返回历史腕部帧。matched selector ablation 的 wrist object consistency 为 recent/stride/W-VMem `0.401/0.463/0.502`，说明它不是普通 recency/stride 的换名；但证据对象是 world-model 视频，policy `58→72%` 来自人工筛选 synthetic data 后重训，不是部署 memory utility。论文没有 code/checkpoint/supplement，当前 Put Back 也缺少移动 wrist calibration，所以不新增 payload family、source candidate 或 executable 近似。

`artifacts/2026-08-16-memora-paper-release-audit.json` 把 MEMORA 去重为四个 planner-side lifecycle/retrieve 算子：evidence-linked entity state-history editor、query-time snapshot rollback、cross-episode regularity consolidation 与 procedure/grounding typed router。environment/entity/activity/skill 四类 payload 已被现有 object、scene、transition 与 executable-skill family 覆盖，不再建立 `memora=true` 总开关。论文 matched EAM-QA 从 episodic `60.1%` 到 full `74.5%`，支持 consolidation；但 Qwen planning Replay 上 full `0.338` 低于 episodic `0.345`，所以它应是 controller 可选择的 operator，不是默认常开模块。项目页当前 404，公开 code/checkpoint/benchmark 尚不可用；当前 Put Back 也没有 participant-level cross-episode target，因此不加入 source catalog 或 21-candidate executable suite。

`artifacts/2026-08-16-mimir-paper-release-audit.json` 进一步把 planner-side dynamic grounding 拆成 feedback-supported observation/action world writer、postcondition-verified task agenda update、active-goal bounded hypothesis retrieval 和 fail-closed evidence binding。它不增加新的 task/world payload；关键价值是让 scene belief 与 execution progress 独立更新，再在执行前形成明确的 `(goal, object, source, target, evidence)` binding，无法支持时拒绝动作。论文 world/task removal 都有大幅退化，但没有隔离 grounder 本身，且 source/checkpoint/raw runs 尚未发布，所以它只作为 MEMORA typed router 的更强 action-bearing contract，等待 key utility 与 deploy-visible feedback 后实现。

`artifacts/2026-08-16-gesto-paper-release-audit.json` 保留一个新的 paper-only payload：`grounded_hierarchical_human_activity_event_memory`。它不是另一份 scene graph，而是把 timestamped human-object atomic interactions 完整、有序地组成 goal-driven events，再用显式 evidence link 接到可替换的 persistent object/place graph；因此可双向回答“这个物体/地点发生过什么”和“这个活动用了什么、在哪里”。matched ablation 显示 hierarchy 和 context refinement 都有贡献，但证据仅来自回溯 QA，代码、prompt、query 与 graph outputs 尚未公开。当前 Put Back/Cover Blocks 没有被观察的人类活动与 RGB-D persistent tracking，所以不把它加入 source catalog 或 21-candidate executable suite；未来只在 shared-space/human-activity benchmark 上激活。

`artifacts/2026-08-16-spatial-memory-agent-paper-release-audit.json` 不新增另一套 procedure store，而是保留 card-level 的 `visit_evidence_shrunk_transfer_reliability_updater` 与 `semantic_then_transfer_reliability_reranker`：memory 初值一致，只有后续被检索后的可验证结果才逐步改变 transfer reliability，deployment 再冻结 payload/value。这与项目最终让 Agent 管理 memory 经验直接相关，但只能作为透明 lower bound：论文把同一终局 reward 同时赋给 top-3 cards，没有局部 credit，且从 10 个 calibration pass 选主表最佳 deployment checkpoint。官方页面仍为 Code Coming Soon；因此它不进入 source catalog 或 21-candidate executor suite，也不替代 paired success utility gate。

`artifacts/2026-08-16-r4dsg-paper-release-audit.json` 把 R4DSG 去重到既有 grounded-object、scene-graph、action-transition 与 episodic-segment payload 上，只保留六个不重复的 paper-level operator：保守相对对象身份关联、static-anchor/dynamic-object 角色推断、anchor-relative state 编码、持续 anchor change 写入、segment-window relational document 写入和 answer-option-blind retrieval。最有用的 matched control 是 option-blind 下 No-Transition→完整 no-why 的 overall/when `34.9/34.7→37.3/43.1%`，支持 identity+transition 的联合价值，但尚未拆开两者。官方 MIT GitHub 的完整 21-file tree 只有静态项目页与媒体，论文所称 released scene-graph outputs、代码、memory JSON 和 runs 均未出现；因此不新增重复 source family 或 executable `r4dsg=true` 开关，等待 RGB-only pipeline 与 identity/transition factorial。

`artifacts/2026-08-16-dreamfly-paper-release-audit.json` 保留新的 paper-only `instruction_conditioned_evidence_promoted_visual_landmark_slots`：冻结 dense/region routers 产生 instruction-conditioned local candidates，跨观察积累 visual/spatial support，经 persistent 或严格 single-observation promotion 写入 16 个 `best-instance anchor + optional stable prototype + age` slots。它不同于 Mem-0 initial-frame anchor、recent sliding 或 phase keyframe；同时抽出通用 `decision_read_before_write_memory_boundary`，这里的 causal 只表示历史前缀无未来泄漏，不是因果推断。memory-only progressive ablation 的 SR/SPL 为 `21.55/16.09→24.11/19.85%`，完整系统去 memory 则 `31.46/27.17→19.60/18.76%`。但 adapter 需与 policy 联合训练，论文未公开代码、checkpoint、FIFO/threshold/write utility/retention 等关键规则，故不制造硬插入 π0.5 的假复现，也不加入 source/executable suite。

`artifacts/2026-08-16-streamflow-paper-release-audit.json` 不新增 raw-frame、patch 或 compressed-latent payload，而是保留五个替代 operator：ViT 前 I-frame raw-residual patch selector、reference-anchored adjacent GOP sparse consolidator、generation-prefix max-frame GOP retriever、`visual_attention_deficit_memory_read_controller` 与 generation-time post-prefix latent injector。full/去 mid-term/去 long-term 在 RTVU 为 `81.55/76.86/80.18%`，VideoMME-Long 为 `62.11/60.33/51.67%`；两时域确有互补价值。但 VAS timing 只比预算匹配 delimiter/random 高 `0.23/0.49 pp`，主要收益是允许 long-term access，不是精确 trigger。论文无代码、compressor checkpoint 或 labels，且该控制点依赖 autoregressive answer tokens，不能冒充 π0.5 read gate；故仅冻结 paper contract。

`artifacts/2026-08-16-drivevla-m0-source-audit.json` 新增 paper-only `retrieved_supervision_adaptation_case_memory`：跨 episode 记录不只保存检索 key，还保存与 policy checkpoint 绑定的 decoder inputs、expert trajectory 与 oracle proposal scores；命中后用这些标签对当前场景的 static/dynamic LoRA 做临时梯度更新，并在下一场景重置。它不同于 Mem-0 token injection、OptimusVLA action prior、VLA-Pro 已训练 adapter retrieval 和 RoboTTT 持续 fast weights。同一 NAVSIMv1 base 的 no-memory/map/map+agent 为 `91.0/91.7/92.3` PDMS，offline LoRA/full TTT/LoRA TTT 为 `91.2/92.4/92.3`。但 Apache-2.0 官方仓库明确只发布 Base 与 Retrieve Model，HF 只有两个 gated checkpoint；memory bank、dedup/hierarchical search、trigger、TTT、场景 reset、target stats 与 runs 均缺失。因此只登记 typed failure writer、dual-key encoder、labeled store、retriever、trigger、temporary adapter utilizer、reset 与 score fusion，不加入 source/executable catalog，也不改变正在运行的 Put Back gate。

`artifacts/2026-08-16-consolidator-source-audit.json` 保留 Consolidator 的非重复轴：显式边界处把 routed STM 经共享 slot-local transform 累积到 LTM，清空 KV/STM 后让 LTM 既参与读取，又直接改变下一段同层 slot routing。公开 30/30 run package 复算得到 frozen-PMNet same-checkpoint learned/identity `87.02/18.32%`，direct routing on/off `87.02/44.38%`，而 immediate STM 同为 `89.90%`，说明 retained state 作为 future-addressing signal 有独立价值。它复用现有 learned latent/recurrent payload，只登记 consolidation、LTM-conditioned router、跨段 lifecycle、identity intervention 与 dual-objective training 五个 typed contract；由于只验证两段 synthetic task、依赖 PMNet phase slots、repo 无 tests 且需要跨 boundary 训练，不加入 executable/source family，也不硬塞当前 π0.5 checkpoint。

`artifacts/2026-08-16-streamttt-paper-release-audit.json` 将 StreamTTT 去重到既有 RoboTTT fast-weight family：parallel sliding KV、KVB fast weights、near-zero tanh residual 与 ordered sequence training 都不另计，只保留 input-dependent momentum/decay updater 和跨 forward 续接 `fast weights + momentum + conv prefix + partial chunk` 的 lifecycle。same-backbone joint sliding/hybrid 的 RT/episodic-recall 为 `68.19/49.58→78.85/59.55%`，说明完整 hybrid 有信号；但仅一 seed、没有 wrong/reset/shuffled-state intervention，heuristic store 又是无 retrain 对 jointly-trained fast weights。v1 明示 work in progress/code will be released，当前无 official code/checkpoint/raw runs或完整超参数，因此不新增 payload/source/executable 数量。

`artifacts/2026-08-16-qcr-paper-release-audit.json` 保留一个与 store/retrieval 分离的 post-retrieval utilization：给定同一 selected trajectory、当前 query 和初始 observation，生成 `workflow invariant / bindings to re-obtain / applicability / verification` 四字段 support，要求 actor 从当前证据重建 binding，而不是复制 source 值。2391 targets 上 QCR 相对完整旧轨迹把 mean success 从 `51.6→62.3%`、online tokens 从 `18.4k→9.4k`，large binding shift stale-binding 从 `46.9→10.9%`。它不新增 procedure payload，只登记 target-bound reuse adapter 与 fixed-retrieval intervention；由于任务由 source 受控改写、仅用成功单 memory、四字段未拆分且无 code/bank/raw ledger，当前只服务后期 architecture-search Agent，不进入 π0.5 executor。

`artifacts/2026-08-16-echovla-paper-release-audit.json` 将 EchoVLA 去重到既有 spatial-map、sliding 与 MemoryVLA dual-stream 设计，只保留 `reconstruction_discrepancy_voxel_write_gate`、`coordinate_keyed_ema_voxel_revision_store` 和 `local_frustum_spatial_feature_retriever`。同模型 PnP Counter-to-Stove 消融中，去 scene memory 后 mobile/static success 从 `17/21%` 降到 `9/16%`；但 Open Refrigerator 上完整方法低于 π0.5 的 `40/50%`，论文归因为动态几何 ghosting。标题匹配的 GitHub 仓库虽声称提供完整实现，固定 commit 实际只有 21 个网页模板/静态文件，Code/arXiv 链接仍为占位符，HF 也无权重或数据。因此不增加 payload/source/executable 数量；只有具备部署可见 calibrated RGB-D/pose、out-of-view task 与 map namespace 防泄漏后，才把这三个空间算子接入 Agent 搜索。

`artifacts/2026-08-16-dream-spatial-source-audit.json` 为已有 spatial-map family 补上 pose correction 生命周期，而不是重复登记 voxel payload。`memory_harness.spatial_reintegration` 已实现表示无关的 `PoseGraphReintegrationPlanner` 与 `KeyframeAwareObservationPruner`：前者从历史 integration pose 和优化后 pose graph 产生 `none/local/regional/global` 重建计划及 cache-invalidation 信号，后者产生 observation eviction 与 keyframe/recent feature-retention 计划；`SpatialLifecycleProgram` 将二者组合为一个 typed decision，并要求 `calibrated_rgbd_observation_archive / optimized_pose_graph / spatial_evidence_reintegrator` 三项 capability，缺一项即 preflight 拒绝。DREAM 整套系统相对 DynaMem 的真实机器人 long-term success 为 `62.5%` 对 `48.8%`，shared replay memory 为 `0.37–0.63 GB` 对 `1.32–10.43 GB`；但论文没有单独消融 RMP，官方配置还默认关闭 global rebuild，因此这些数值不能当作两个 lifecycle 插件的独立收益。当前 Put Back/Cover Blocks 缺少上述空间输入，模块保持不可选，也不改变 17-family source catalog 或 21-candidate suite。

```bash
python -m memory_harness.research_candidates
```

## 官方 Mem-0 M1 sanity baseline

三种 memory 的论文数值、released checkpoint 诊断与 π0.5 port 状态统一见
[`docs/2026-08-15-mem0-three-memory-reproduction.md`](docs/2026-08-15-mem0-three-memory-reproduction.md)。

论文数值与 π0.5 port 必须分开。RMBench 在 2026-07 重新发布了无需 planner 的 `m1_mix` executor；其 model card 在 `put_back_block / demo_clean / unseen / action_horizon=30` 上报告 `100/100`。下载得到的是 byte-split checkpoint，先用带 SHA-256 校验的流式入口重组，不能直接加载单个 part：

```bash
PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.reassemble_checkpoint \
  --parts-dir /path/to/m1mix/checkpoint \
  --checksum-file /path/to/m1mix/checkpoint/m1_mix_final_step50000.pt.sha256 \
  --output /path/to/m1mix/checkpoint/m1_mix_final_step50000.pt \
  --manifest /path/to/m1mix/checkpoint/reassembly_manifest.json
```

重组后可用同一个参数化 runner 评测任意官方 M1 任务；task prompt 从 release 的 `task_instructions.json` 读取，不在脚本中复制：

```bash
memory-harness/scripts/run_official_mem0_m1mix.sh \
  --asset-root /path/to/m1mix \
  --checkpoint /path/to/m1mix/checkpoint/m1_mix_final_step50000.inference.pt \
  --task put_back_block \
  --num-episodes 10 \
  --seed-start 100000 \
  --gpu-id 1 \
  --output-dir /tmp/mem0-m1mix-put-back-gate10
```

要在相同 checkpoint、相同 seeds 上做执行时 memory 依赖诊断，可运行：

```bash
memory-harness/scripts/run_official_mem0_m1mix_ablation.sh \
  --asset-root /path/to/m1mix \
  --checkpoint /path/to/m1mix/checkpoint/m1_mix_final_step50000.inference.pt \
  --task put_back_block \
  --num-episodes 10 \
  --seed-start 100000 \
  --gpu-id 1 \
  --output-dir /tmp/mem0-m1mix-put-back-ablation
```

该脚本依次运行 `full / without_anchor / without_sliding`。后两项是在同一已训练
checkpoint 上把相应 memory 输出替换为当前视觉 token，同时保留 write/lifecycle；它们是
推理 intervention，用于快速检查模型是否实际使用 memory，不等同于论文中重新训练各消融
架构的严格复现。

该官方 checkpoint 只用于验证 released Mem-0 executor、RMBench 环境和 anchor/sliding 的原生性能上界；π0.5 port 仍使用自己的 backbone、训练预算和同 checkpoint mask 消融，不能借用官方 `100/100` 作为移植结果。

## Mem-0 M(n) key / no-key matched reproduction

Cover Blocks 的 released executor 已在本地通过发布方 SHA-256 校验。planner 侧使用相同
50 demonstrations、相同 75 optimizer steps 分别训练的 key 与 no-key Qwen3-VL-8B；key
读取初始观测和有序 `(completed subtask, end image)`，no-key 严格只读取当前阶段边界图。
两种输入策略由同一 hook 选择，executor、SEC、环境和 seeds 不变：

```bash
bash memory-harness/scripts/run_reproduced_mem0_cover_blocks_planner_condition.sh \
  --condition no_key \
  --num-episodes 10 \
  --seed-start 100000 \
  --gpu-id 1 \
  --output-dir /tmp/mem0-cover-blocks-no-key-gate10
```

发布方没有提供论文 Table 2 的 planner ablation 权重，因此这里是 matched local
reproduction，不是官方 ablation checkpoint 的精确 replay。runner 会拒绝占用中的 planner
端口，结束时清理完整 server process group，并把实际 planner 输入图、输出、executor
checkpoint SHA、planner weight-index SHA 和与既有 key `2/10` 结果的 paired delta 写入 artifact。

10-seed 配对已完成：key 为 `2/10`，no-key 为 `0/10`，两个成功 seed 均为
key-only。planner 输出的 diagnostic exactness 从 no-key 的 `25/46 (54.35%)` 提升到
key 的 `48/48 (100%)`；差值为 `+45.65 pp`。两侧使用相同 simulator seeds
`100000–100009`、policy seeds `120000–120009` 与 released executor。固化 summary 位于
`artifacts/2026-08-15-mem0-cover-blocks-key-no-key-summary.json`，其 SHA-256 为
`6f8a8d451b58dd797711d901aa1d80a8228ae14b08fd17468cdfc17f7997346c`。

## Mem-0 π0.5 port 训练数据

固定 store/retrieve program 本身无需训练；Mem-0 的 fusion 和 π0.5 action path 需要联合训练。每个任务先建立确定性的 train/validation split 和空 memory baseline context：

```bash
PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.build_task_template \
  --task-config configs/tasks/put_back_block.json \
  --hf-lerobot-home ../rmbench_lerobot_data \
  --output ../rmbench_runs/emac_put_back_block_v1/task_template.json

PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.build_training_data \
  --task-config configs/tasks/put_back_block.json \
  --template-manifest ../rmbench_runs/emac_put_back_block_v1/task_template.json \
  --program-config configs/training_empty_mem0.json \
  --output-manifest ../rmbench_runs/emac_put_back_block_v1/none_context_manifest.json \
  --output-bank ../rmbench_runs/emac_put_back_block_v1/none_context_bank.npz \
  --output-audit ../rmbench_runs/emac_put_back_block_v1/none_context_audit.json

scripts/run_pi05_memory_train.sh put_back_block
```

M(n) 任务不能复用上面的“每个 episode 一个全局 prompt”假设。Cover Blocks 的模板生成器
读取每帧 `task_index`，将轨迹 run-length encode 为六个连续 subtask segment，并从
`meta/tasks.jsonl` 恢复真实 executor prompt。当前修正版从可用的 79 条 episode 中按冻结
seed 选择任务配置要求的 50 条，得到 300 个 segment（每条轨迹恰好 6 个）；manifest 为
`../rmbench_runs/emac_cover_blocks_v2_subtask_prompt/task_template.json`，SHA-256 为
`7e4a611764bc3e2323a54858c28b68430a84fb40e42c942e86537b7e2ebfcb74`。后续 context
必须显式使用这个 manifest 和新的输出目录，不能覆盖旧的 global-prompt artifact：

```bash
PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.build_task_template \
  --task-config configs/tasks/cover_blocks.json \
  --hf-lerobot-home ../rmbench_lerobot_data \
  --output ../rmbench_runs/emac_cover_blocks_v2_subtask_prompt/task_template.json

CONTEXT_ROOT=$PWD/../rmbench_runs/emac_cover_blocks_v2_subtask_prompt \
TEMPLATE_MANIFEST=$PWD/../rmbench_runs/emac_cover_blocks_v2_subtask_prompt/task_template.json \
scripts/build_pi05_memory_context.sh cover_blocks anchor_sliding /path/to/corrected/none/checkpoint
```

context generator 会检查所选 OpenPI data factory 同时启用 `prompt_from_task` 并在 repack
中保留 `prompt`；任一条件缺失都会在编码前失败，禁止再静默退回 global instruction。

若复用的是迁移前生成、但已通过上述等价审计的 context，训练入口通过 `PROGRAM_MIGRATION_AUDIT=/absolute/path/to/audit.json` 显式传入审计文件。正常的新 context/config 哈希一致时不需要该变量。预算匹配的另一个 memory-training 分支应通过 `RUNTIME_SNAPSHOT_SOURCE=/absolute/path/to/reference-checkpoint/runtime` 复用同一冻结 runtime；方法级比较器会拒绝两条 memory 分支的 runtime 哈希不一致。

随后从 no-memory checkpoint 生成 `fixed_anchor_sliding.json` 的部署一致 contextual latent，再训练 full-memory checkpoint：

```bash
scripts/build_pi05_memory_context.sh put_back_block anchor_sliding \
  ../rmbench_checkpoints/pi05_aloha_pen_uncap_mem0/<none-run>/<step>
```

标准 sliding 每个 environment frame 写一次 memory；即使 π0.5 连续执行 10 个 cached actions，runner 也只跳过 action sampling，不跳过 `observe → write decision`，因此 always-write 的 30-slot sliding 始终表示最近 30 个环境步。anchor/sliding 固定消融在这一个 full-memory checkpoint 上 mask；改变 latent 分布的 program（如 consolidating、novelty_sliding、dhem_event）先做 zero-shot compatibility screen，只有出现正向信号才使用独立、预算匹配的训练表。生成器直接执行所选 runtime program，因此新增 write/store/lifecycle 无需修改训练数据管线。旧 8-token pooled adapter 实验是已终止的诊断分支，结果见 `docs/2026-08-14-shared-utilizer-experiment.md`。候选模块路线见 `docs/2026-08-14-memory-module-catalog.md`。

Successive evaluation 使用不重叠 seed shards：screen 3 个、pilot 再加 17 个、confirmation 再加 30 个。`compare_fixed_runs` 可组合 run sets，并拒绝重复的 `(layout seed, policy seed, layout fingerprint)`，防止重复 episode 被错误计入有效样本量。固定消融至少进入 20-episode pilot；zero-shot 探索模块只有 success 正向或 max/total reward 同时正向才晋级，负向或无方向 screen 不自动消耗更多 rollout。

novelty writer 的阈值用已有 causal context bank 离线校准；工具直接复用正式 `NoveltyWrite` 决策实现，并报告每个阈值的写入率、触发原因、写入间隔和距离分位数：

```bash
PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.calibrate_novelty \
  --manifest ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_manifest.json \
  --context-bank ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_bank.npz \
  --output /tmp/put_back_block_novelty_calibration.json
```

DHEM store 也可在同一 causal source stream 上先做低成本行为画像，报告 append/discard/merge 比例、最终 token 的代表年龄和累计质量：

```bash
PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.profile_dhem \
  --manifest ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_manifest.json \
  --context-bank ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_bank.npz \
  --output /tmp/put_back_block_dhem_profile.json
```

两级 store 可直接复用下一时刻 context 的 latest slot，因果重建 source moment；profiler 不生成新的大型 bank，只报告其与去除 anchor 后 sliding history 的差异、token 数、原始历史覆盖范围和维护事件：

```bash
PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.profile_tiered_contexts \
  --manifest ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_manifest.json \
  --context-bank ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_bank.npz \
  --program-config configs/fixed_tiered_chunk_mean.json \
  --output /tmp/put_back_block_tiered_chunk_mean_profile.json
```

多时间尺度 retrieve profiler 复用相同 causal source stream，同时比较 `temporal_multiscale`、`uniform_global` 和显式 `recent:15 + global:15` 组合，报告固定 30-slot 预算中有多少项超出 latest-30、选中帧的 lag 分布、集合重合度，以及是否退化为 sliding：

```bash
PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.profile_temporal_multiscale \
  --manifest ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_manifest.json \
  --context-bank ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_bank.npz \
  --output /tmp/put_back_block_temporal_multiscale_profile.json
```

`boundary_chunk` 先只用 train split 冻结相邻 contextual-token 的分段阈值，再在 validation split 报告 coherent-chunk retrieval；两个步骤都不读取 rollout outcome：

```bash
PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.calibrate_boundary_chunk \
  --manifest ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_manifest.json \
  --context-bank ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_bank.npz \
  --template-manifest ../rmbench_runs/emac_put_back_block_v1/task_template.json \
  --output configs/calibrations/put_back_boundary_chunk.json \
  --threshold 0.9993 --threshold 0.99935 --threshold 0.9994 \
  --threshold 0.9995 --threshold 0.9996 --threshold 0.9997 \
  --max-items 30 --min-chunk-items 30 --target-median-chunk-count 3

PYTHONPATH=. ../../openpi-libero/.venv/bin/python \
  -m memory_harness.profile_boundary_chunk \
  --manifest ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_manifest.json \
  --context-bank ../rmbench_runs/emac_put_back_block_v1/anchor_sliding_context_bank.npz \
  --template-manifest ../rmbench_runs/emac_put_back_block_v1/task_template.json \
  --calibration-report configs/calibrations/put_back_boundary_chunk.json \
  --output artifacts/2026-08-15-put-back-boundary-chunk-profile.json \
  --max-items 30 --min-chunk-items 30
```

该插件只复现“先分段、再按整段检索”的结构下界；RoboMME-Interference 原方法使用 SigLIP 并跨 session，二者结果不能混报。

当前 v4 训练校准、直接 action cross-attention 修正和梯度累积协议见 `docs/2026-08-14-mem0-pi05-v4-calibration.md`。单卡 batch 2 时 runner 默认累积 28 个 micro-batch，形成 effective batch 56；正式复现按 checkpoint 学习曲线报告，不再用 50-update calibration 判断 utility。

## Paper-only train/deploy availability contracts

BPP v2 的 semantic keyframe payload 已由 `task_phase_keyframe_history` 覆盖，不再增加同义
family。它新增的有效接口是 rising-edge event writer 与 detector-latency-aligned
availability mask：训练 context 只能包含在 `t-Δ` 前发生的事件，rollout 只能包含异步
detector 已经返回的事件，并锁定 detector revision、prompt/criterion、图像配对、query
cadence 与共享时基。该约束必须同时接入 `build_training_data` 和异步 rollout；只在 runtime
加 minimum-age filter 会形成不完整实现。论文没有官方代码或 latency-mask 单独消融，因此
当前只保留 [release audit](artifacts/2026-08-16-bpp-paper-release-audit.json)，不加入
source-audited catalog 或 executable suite，也不改变正在运行的 Put Back 链。
