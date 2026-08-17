# E-MAC：让研究 Agent 自主探索具身记忆架构

> **一句话概括**：先在 RMBench 上用同一套 π0.5 backbone 建立可靠的无记忆与固定记忆基线，确认不同任务确实需要不同历史信息处理方式；再让研究 Agent 在统一接口、固定预算和锁定评测下，自主提出、实现和迭代新的 memory architecture。

## 1. 项目定位

- **论文类型**：Method / System Paper，同时提出 Autonomous Embodied Memory Architecture Search 这一研究设定。
- **核心问题**：研究 Agent 能否根据机器人实验反馈，自主发现比人工固定方案更有效、更省资源、且能跨任务迁移的 memory architecture？
- **核心贡献不是** anchor、sliding 或 key memory 本身，也不是因果推断方法；这些固定模块和内容干预只用于建立可信起点、分析有效区间并约束后续搜索。
- **主 benchmark**：RMBench；策略主干固定为 π0.5。Gate 1 后用 RoboMME 的 locked tasks 检验跨 temporal/spatial/object/procedural memory 类型的迁移，不替代 RMBench 主线。

## 2. 可行性判断

### 2.1 总体结论

该方向**有条件可行，值得推进**，但不能直接从开放式 Agent search 开始。当前最合理的路线是：

1. 先解决 π0.5 executor 的能力地板和评测稳定性；
2. 在同一 π0.5 上实现真正可比较的 fixed memory modules；
3. 先训练只能在少量固定模块间选择的 controller；
4. controller 确实优于固定方案后，再开放写入规则、表示、检索、融合和生命周期的 architecture evolution。

这条路线的科学价值在于：ALMA 已经直接证明 Meta Agent 可以从零生成、调试、评测并归档 text-agent memory code；MemEvolve 已证明通用语言 Agent 的 memory program 可以被演化，ENPIRE 已证明 coding agent 可以根据真实机器人反馈自主改进策略，AgentCanvas/KDLoop 已在导航、EQA 和操作中验证“coding agent 编辑 typed embodied graph + simulator rollout”，HyMeS 又在相同 π0.5 权重上搜索 task-specific symbolic memory / verification / steering program。因此本项目不能再声称“首次让 Agent 设计 memory”“首次做具身架构搜索”或“尚无 Agent 搜索具身 memory code”；它要解决的更窄问题是：**Agent 能否在任务无关的 typed VLA-memory 接口中搜索 encode、write、store、retrieve、utilize 与 lifecycle 架构，并在预算匹配和 locked-task 评测下获得可迁移的 memory 结构，而不是搜索任意文本 memory code、整个 agent graph，或为每个任务编写 stage plan 与 steering reward。**

### 2.2 仓库已经具备的基础

- π0.5 fine-tune、RMBench rollout、固定 seed 和 full-episode 评测入口已经存在：`run_rmbench_finetune_pi05_base.sh`、`scripts/run_rmbench_cover_blocks.py`。
- 已有 memory payload、provider、selector、injector 和 lifetime 抽象：`scripts/memory_injection/`。
- OpenPI 分支已有 `memory_tokens`、mask、memory router 和 cross-attention 接口，可作为 latent memory 的模型侧入口：`../openpi-libero/src/openpi/memory/` 与 `../openpi-libero/src/openpi/models/`。
- 已有 snapshot intervention、full-episode ablation、Mem-0 planner/executor 和配对布局审计代码，可复用实验基础设施。
- RMBench Cover Blocks 的 Mem-0 本地复现链路已经跑通；当前最严格的 10-seed 版本为 `2/10`，可作为工程 sanity check，但不能等同于论文报告的 `68/100`。
- 原论文消融已经证明三类 memory 在其适用任务族上有明显平均价值：M(1) 完整版 `52.8%`，去 anchor/sliding 分别为 `26.8%/40.4%`；Put Back 单任务为 `90%→35%/78%`；M(n) 完整版平均 `28.5%`，去 key 为 `4.8%`。但模块不是逐任务单调有效，例如 Cover Blocks 完整版 `68%`，去 anchor/sliding 反而为 `92%/84%`。因此本项目不是重新质疑 memory 是否总体有效，而是先忠实复现，再把这种任务依赖性变成 Agent 可搜索的模块选择问题。
- 官方在 2026-07 重新发布了无需 planner 的 `m1_mix` executor，并在 Put Back Block 上报告 `100/100`（论文原表为 `90/100`）。其 15.3 GB split checkpoint 已按发布方 SHA-256 完整重组，并生成不含 optimizer 的 inference 副本。本地 seeds `100000–100009` 的三条件已完成：`full=10/10`、`without_anchor=2/10`、`without_sliding=0/10`；配对下降分别为 `80/100 pp`，证明 released model 对 anchor 和 sliding 都存在强执行时依赖。三组真实 seeds、逐 episode 日志、aggregate 与 checkpoint hash 已通过机器审计。`full` 是 released Mem-0 原生 sanity baseline；后两项只诊断执行时依赖，不冒充论文 Table 2 的精确重放。论文正文与当前 release 均未说明或提供 anchor/sliding 消融的独立训练产物；所有结果均与 π0.5 port 分栏报告。
- π0.5 上的 Mem-0 action-side port 已完成同 checkpoint 的 `none / anchor / sliding / anchor+sliding` 插拔：4 个配对 seed 中，anchor+sliding 的平均最大阶段 reward 为 `0.1125`，none 为 `0.0750`；它是正向阶段证据，但两者 full success 仍为 `0/4`。
- 审计 released code 后又发现旧 Cover Blocks port 的 sliding 以 action query（每 10 个环境步）而非 environment step 更新；该结果因此降级为粗粒度诊断。当前通用 runner 已支持 cached action 期间单独 `observe()`，并强制审计每个环境步都有一次 executor memory update；Put Back Block 从该忠实 30-step 定义重新开始。
- 进一步审计发现旧 Cover Blocks π0.5 配置既未启用 LeRobot `task_index→prompt`，repack 也丢弃了 `prompt`，因此训练和 memory latent 编码实际使用 global instruction，而部署的 key planner 输出 subtask instruction。旧 `1399` memory checkpoint 与无训练 provenance、使用同类 global-prompt 配置的 `9999` baseline 均从 task 默认配置移除，只保留为历史诊断；task schema 不再携带 checkpoint 路径，rollout 必须显式绑定经 finalizer 审计的新 checkpoint。OpenPI 的 native/memory 两个配置现均强制保留逐帧 subtask prompt，context generator 会在该 contract 不满足时直接失败。Put Back Block 只有一个全局 task prompt，当前训练链不受该修正影响。
- 修正后的 Cover Blocks task template 已按逐帧 `task_index` 将每条轨迹切成 6 个连续 subtask segment；从 79 条可用数据以冻结 seed 选择 50 条，共 300 段、六种真实 prompt 各出现 50 次。manifest 为 `rmbench_runs/emac_cover_blocks_v2_subtask_prompt/task_template.json`，SHA-256 为 `7e4a611764bc3e2323a54858c28b68430a84fb40e42c942e86537b7e2ebfcb74`，后续 π0.5 context/executor 只允许从该新 lineage 重建。
- 修正 lineage 在接入 key planner 前先跑 native π0.5 的 oracle-subtask executor diagnostic：只有显式 diagnostic 模式可读取 simulator 当前子任务，run manifest 必须标记 `deployable=false / executor_skill_diagnostic_only`；默认 clean baseline 继续拒绝该输入。这样可把“低层六个技能没学会”与“key memory 没有被利用”分开，而不把 oracle 结果作为主表 baseline。
- 该诊断已接入确定性的续跑协议：native u1200 有阶段信号时，训练同预算的 empty-memory π0.5 executor，并在同一 checkpoint 与 3 个共享 seeds 上比较 matched `key / no_key` planners；若仍处于零地板，先把 native 学习曲线延长到累计 u3000，再决定是否启动 planner 对照。planner model index、adapter、base model index 与 300-sample key/no-key training-pair manifest 均按 SHA-256 写入 run artifact；任一分支都不会直接放行 controller。
- planner-side `key` 与 `w/o key` 已按 released code 进入同一 architecture facade，并分别完成训练预算匹配的 planner：二者使用相同 300 个阶段标签、75 optimizer steps 和同一 π0.5 executor，只改变 `初始图+完成历史` 与 `当前图` 的输入协议。30 条分层训练格式验证中，key 为 `30/30`，no-key 为 `26/30`。
- 4 个严格配对 seed 的 oracle-boundary diagnostic 中，key 相对 no-key 的阶段结果为 2 胜、1 平、1 负；平均最大 reward 为 `0.0750 vs 0.0625`，平均 total reward 为 `91.175 vs 79.475`，但 full success 均为 `0/4`。这支持 key history 的局部阶段价值，但尚未通过稳定 utility gate。
- MemoryVLA-style adjacent merge 已完成部署一致的 context 生成、同预算 50-update π0.5 fine-tune 和 4-seed matched-training diagnostic。单 seed 的正向筛选没有复现到多 seed：相对 sliding，平均最大 reward 为 `0.0875 vs 0.1000`，平均 total reward 为 `107.925 vs 122.350`，1 胜 3 负且 full success 均为 `0/4`。因此该模块保留为可插拔负结果，不进入 controller 候选集。
- 已将 DiM-WAM 的单 bank、无学习 DHEM maintenance rule 拆为 `dhem_event_store`：它显式保护 anchor/latest，按语义相似度与时间距离比较 incoming 冗余和中间历史冗余，并在需要时做累计质量加权合并。多 bank query、diversity/progress loss 和 world-action prediction 仍被标为 learned operator，不与该 store 的效果混称。
- 在 50 条 Put Back Block causal source stream 上，该 DHEM store 满容量后丢弃 `75.03%` incoming、合并并接纳 `24.97%`，最终 token 的代表年龄中位数为 `179` 步、最大 `362`；它已证明形成不同于 30-step ring 的长跨度稀疏历史，但是否改善 policy 仍等待同 checkpoint paired rollout。
- 为事件写入补齐了 causal delayed-write contract：writer 可在当前时刻确认并写入先前候选 payload，runtime 会审计 source/confirmation delay 并在 lifecycle boundary 清空 writer 状态；OpenPI 部署路径与训练 context generator 均传入 typed robot state。该接口为 KEMO-style kinematic peak writer 服务，但不会把未公开的 peak 判定细节自行猜成“忠实复现”。
- 在该 contract 上新增了明确标注为 inspired baseline 的 `causal_kinematic_peak`，并与 anchor 组合成 `fixed_kinematic_event.json`。50 条 Put Back Block demonstrations 上共选出 `270` 个事件，每条 `4–8` 个、中位数 `5`；它形成了不同于 dense ring/latent novelty 的稀疏 motion-event 写入，但 DINO visual dedup 和 policy utility 尚待独立验证。
- 新增独立 retrieve operator `content_recency`：episode bank 内以当前 contextual latent 做余弦检索，并施加 frame-gap penalty；top-30 恢复时间顺序后复用相同 Mem-0 slots。冻结 penalty `1e-5` 时，50 条 demonstration 的 full-bank query 平均有 `14.95%` 项来自最近 30 步之外，selected lag P90 为 `38`、最大 `158`，因此既非 FIFO 复制也非无约束 stale retrieval；paired policy utility 已排队。
- 新增 MemoAct 启发的 `tiered_chunk_mean` store：容量 6 的短期区无损保留最近 token，溢出时把最老 3 条均值压缩后迁入容量 8 的长期区，长期区满时复用 adjacent merge。它只隔离“两级容量 + chunk migration”生命周期，不冒充 learned sensory encoder、causal compressor、temporal retrieval 或 gate；registry、architecture config、容量边界、episode reset 和 Agent candidate smoke 已通过完整 CPU 回归。真实 Put Back Block bank 的 17,612 个 query 上，它用最多 14 个 token 在 `91.20%` query 中覆盖 sliding-30 之外的历史，且与去除 anchor 的 sliding 仅 `1.99%` query 完全相同，证明结构不重复；policy utility 仍待 fixed module 建立可比较 success/progress 后评测。
- 新增 CycleManip 启发的 `temporal_multiscale` retriever：在相同 30 个 raw-token slot 中，以指数 lag 覆盖离散旧时刻，再从其余全局历史均匀补齐。Put Back Block 的 16,012 个 full-history query 上，每次严格返回 30 项，平均 `66.97%` 来自 latest-30 之外，与 sliding 完全相同的 query 为 `0%`。它只隔离无训练多尺度采样；全历史 joint-state encoder 与 progress objective 保持为 Gate 1 learned operators。
- 新增 RoboMME FrameSamp 启发的 `uniform_global` retriever，作为 `temporal_multiscale` 全局分支的必要消融：相同 30-slot 预算下平均 `76.68%` token 来自 latest-30 之外；它与 sliding、temporal multiscale 完全相同的 query 均为 `0%`，与后者 selected-set Jaccard 仅 `0.1643`。完整 MME-VLA 的 multi-patch representation 与 AdaLN modulator 仍需联合训练。
- 修正多 memory 组合接口：`mem0_context` 现在要求每个 history path 显式声明 slot quota，配额总和必须等于 30，`USE` trace 只记录实际进入 policy 的 item。首个组合 `recent:15 + global:15` 排除跨分支重复；在 16,012 个真实 query 上每次恰好使用 30 项，平均 `42.12%` 超出 latest-30，与 sliding、uniform-global、temporal-multiscale 的完全相同率均为 `0%`。
- 新增 RoboMME-Interference 启发的 `boundary_chunk` retriever：按相邻 contextual-token 相似度切分连贯片段，以当前 query 对片段内 item 的最大相似度选段，再均匀采样到 30-slot。阈值仅由 40 条训练 episode 冻结；10 条 validation episode 的 3,207 个 full-history query 上始终返回 30 项，平均 `64.90%` 来自 latest-30 之外，仅 `0.405%` 与 sliding 完全相同。它隔离 coherent-chunk retrieval，但不冒充原方法的 SigLIP、跨 session interference 复现，也不把结构差异当作成功率收益。

### 2.3 当前尚未满足的条件

- 2026-08-06 修正后的 Cover Blocks full-episode 五个配置仍全部为 `0/10`；单 seed gate 成功没有转化为稳定结果，说明 executor 能力仍是首要风险。
- 当前 `Full / w/o ...` 系统主要围绕 episodic、working memory、gate、reranker 和 prompt routing；它不等价于 proposal 所需的 `none / anchor / sliding / anchor+sliding / key` 同 backbone 对照。
- task-spec 驱动的通用数据转换、训练和 rollout runner 已支持 `cover_blocks / demo_clean` 与 `put_back_block / demo_clean`；跨 `M(1) / M(n)` 的优势图谱仍需完成 Put Back Block 的学习曲线和 full-episode 评测。
- Put Back Block 的 full-memory 学习曲线已从 200 扩展到 1000 optimizer updates（此前 none 200，总 action-expert exposure 1200）。最终 checkpoint 的 40-sample held-out action loss 为 matched `0.127220`、empty `0.128973`，episode-cluster CI 跨 0，matched 与 mismatched 也没有稳定差异。同 checkpoint、共享 3 seeds 的 `none / anchor / sliding / anchor+sliding` 全部为 `0/3` complete success；trace 证明 memory 每环境步写入、读取并显著改变 action，因此不是插件 no-op。u1200 `full-memory / empty-mask / native-none` 三路总 optimizer exposure 均为 `1200` updates、`67,200` examples，使用相同 base、task config 与 3 组 simulator/policy seeds，结果均为 `0/3` complete success、`max_reward=0`；但 full 的真实 schedule 是 `none 200 → anchor+sliding 1000`，empty/native 才是从 base 按终态条件训练 1200。因此该三路结果是总预算匹配 readiness screen，不是 clean condition-from-initial comparison；comparison v3 已将 schedule、各 program exposure、precondition updates 与 evidence scope 写入证据，并拒绝共享 run provenance 冲突。源码复核进一步确认 Put Back 只有完整放回后才产生 reward，移到中心和按按钮没有 dense stage reward；所以旧 readiness 不能据此断言零部分进展。现已增加 `0/1/2/3 = 无进展/到中心/按按钮/放回` 的只读 task progress trace，并让 comparator/readiness/utility gate 在该任务上使用 progress 作为 screening endpoint；success 仍是主指标，progress-only 永不放行 controller。修正证据为 `artifacts/2026-08-16-put-back-sparse-reward-progress-audit.json`。released Mem-0 `m1_mix` 三条件已经得到 `10/10、2/10、0/10`，Cover Blocks matched key/no-key 为 `2/10` 对 `0/10`，证明官方 executor 链路和三类 memory 依赖均可复现。在 π0.5 出现可比较 progress 前不扩大新结构 rollout。
- 历史 1800 条 snapshot 实验适合机制诊断，不能替代从 reset 开始的 full-episode 结论。
- key memory 的旧 π0.5 4-seed输入对照已经完成，但因上述 prompt contract 失配仅保留为历史诊断，不能再标为忠实移植。为先回答 Mem-0 本身能否复现，已在 released Cover Blocks executor 上完成同接口 `key / no_key` planner input condition：key 使用初始图与有序完成历史，no-key 只使用阶段边界处最新图；两者分别使用相同 50 demos、75 updates 的 matched local planner。10 个配对 seed 上 key `2/10`、no-key `0/10`，两个成功 seed 均为 key-only；planner exactness 为 `48/48` 对 `25/46`，提升 `45.65 pp`。发布方未提供论文 planner 消融权重，因此结果标为 matched local reproduction，并与 M(1) anchor/sliding intervention、π0.5 port 分栏。Cover Blocks π0.5 下一步必须从修正后的 subtask-prompt contract 重新生成 context 并训练。
- adjacent merge 的离线内容审计未通过 utility gate，4-seed matched-training 结果也弱于 sliding；当前不应继续为这一 store 扩大搜索或训练预算。

因此，项目当前状态为 **ACTIVE / reproduction in progress**：Mem-0 固定模块复现、π0.5 executor 训练和配对评测持续执行。只有 `controller eligibility` 尚未满足；这不是项目状态，也不会阻断固定模块实验。旧 shared-adapter 分支已被当前 Mem-0 port 取代。

需要特别说明：当前 Put Back Block executor 仍只是 effective batch 56 的单卡 π0.5 port，不是论文训练预算的忠实复现。论文 Mem-0 executor 按任务从头训练 30,000 iterations、global batch 448；因此 u1200 三路 `0/3` 不能作为 memory 无效或项目停止的依据。预注册续跑按 `native-none → full-memory → empty-mask` 顺序为每路增加 1800 updates，形成累计 u3000 的第二个学习曲线点；full 的累计 schedule 是 `none 200 → anchor+sliding 2800`，而非从第 1 步起的 full-memory。该轮先回答 executor 是否脱离地板以及同 checkpoint 固定模块是否可用，不越界宣称 clean training-condition confirmation；若后者成为结论所必需，再从共同 base 启动对应 clean schedule。截至 2026-08-16，native-none 已完成并通过 checkpoint manifest 验证，full-memory 正在稳定训练，后续阶段由同一脚本自动续接。新 rollout 将同时报告 complete success 与 task progress，再做三路配对 Gate-3。executor 出现可比较 progress 后再比较 anchor/sliding；随后在 `M(n)` 任务完成 π0.5 key/no-key 移植，并报告随训练预算变化的学习曲线。

## 3. Related Work 与差异

| 工作 | 已解决的问题 | 对本项目的启发 | 尚未覆盖的部分 |
|---|---|---|---|
| [RMBench / Mem-0](https://arxiv.org/abs/2603.01229) | 提出 `M(1)/M(n)` 操控任务和 anchor、sliding、key memory | benchmark、任务分层和固定模块起点 | 架构由研究者固定，不做自主搜索 |
| [RoboMME / MME-VLA Suite](https://arxiv.org/abs/2603.04639) | 16 个任务强制 temporal/spatial/object/procedural memory；在同一 π0.5 上比较 symbolic/perceptual/recurrent representation 与 context/modulator/expert utilization | 已实现 FrameSamp 启发的 `uniform_global` lower bound；新增 source-audited `temporal_visual_patch_memory`，把 frozen patch encoder/causal bank、`uniform-frame / prefix-causal RGB-change TokenDrop` retriever与 `Context / action-AdaLN / separate-Expert` utilizer拆成 2×3 typed factorial；官方 suite 作为 locked transfer benchmark | FrameSamp 三种融合为 `30.68/44.51/36.25%`，TokenDrop 为 `34.50/38.04/34.86%`，无历史 π0.5 `17.93%`；两个 selector 和 action-modulator source smoke 已通过，六个 checkpoint hash 已核验。真实 RMBench profile 的 selector frame-set Jaccard 仅 `0.098`，证明二者不重复；released offline TokenDrop 与 online heap 只在 `83.23%` query 完全一致，因此新训练强制 prefix-causal parity，原路径只作 checkpoint compatibility ablation |
| [RoboMME-Interference](https://arxiv.org/abs/2606.22338) | 构造“相关 demonstration + 多个无关 sessions”的长上下文，显示 perceptual memory 会随干扰增长而衰减；用相邻视觉相似度分段后检索相关 chunk | 新增与 item top-k/FIFO 不重复的 coherent-chunk retrieve 维度；本仓库已实现 contextual-token `boundary_chunk` lower bound | 完整方法使用 SigLIP 且跨 session；当前 RMBench 插件仅 episode-local，必须在 locked interference benchmark 上另做忠实迁移 |
| [NativeMEM](https://arxiv.org/abs/2607.06678) | 从 π0.5 原生 vision encoder 初始化 action-supervised memory tokenizer，每个历史 frame-view 压成一个 input-sequence token | 提供区别于 Mem-0 contextual latent/cross-attention 的 learned encode + utilize 组合；官方 OpenPI fork可作为实现基准 | 需要 tokenizer pretraining、feature cache 和 task finetune，不能作为免训练 pooled-token 配置；官方代码仍标为 WIP |
| [OptimusVLA](https://arxiv.org/abs/2602.20200) | 从 demonstration bank 检索相似 action trajectories，并按进度构造 flow action prior | 已登记为 source-audited `trajectory_action_prior`：task embedding、跨 episode bank、semantic top-k、progress alignment、flow initializer、adaptive NFE 与 episode session 均可独立组合 | 官方 6500-trajectory 资产与源码 smoke 已通过；GPM 可在不重训 flow policy 时替换推理初始噪声，但 RMBench 仍需 train-only bank 和 Prior Head。LCM release 与 Motion Tail 重叠，不重复登记；正式对照必须含 random/wrong-task/shuffled-progress prior 和固定 NFE |
| [Retrieve-then-Steer](https://arxiv.org/abs/2605.10094) | 在持续部署中写入经过 outcome/progress 验证的成功 observation-action prefixes，过滤轨迹冲突后将 elite prior 按置信度注入 flow sampler 中间状态 | 已把不重复的 outcome-verified lifecycle 实现为 `verified_success_ring` transactional store；trajectory consistency 与 confidence-adaptive sampler guidance 继续作为 GPM/action-prior family 的后续独立 operator | π0.5 LIBERO-10 `92.4→94.4%`，但未验证 memory 为 `87.6%`、直接 replay 为 `87.8%`。当前 `verified_success_latent` 只验证跨 episode 成功提交/失败丢弃协议，不冒充 action-prior 复现；正式比较必须 ordered deployment、独立 bank budget，并报告 warm-up |
| [RTCF](https://arxiv.org/abs/2608.04527v2) | 对每条成功 visual-action trajectory 维护 causal monotonic alignment frontier，用完整在线视觉前缀定位进度；再只把对齐 action chunk 的 clipped non-DC 低频残差加到 frozen policy 输出，保留高频与 gripper | 不新增第二个 trajectory payload；只在既有 `trajectory_action_prior` family 保留 `incremental_monotonic_alignment_frontier_retriever`、`aligned_future_action_chunk_selector` 与 `clipped_low_frequency_action_residual_utilizer` | matched PI-FAST 在 LIBERO-Long 为 `61.6→68.6%`，去 history alignment 为 `63.6%`、改 time-domain 全残差反降到 `50.0%`，支持两个子算子均有价值；但总体只 `86.4→88.4%` 且 Goal 略降。论文无 code/bank/checkpoint，并漏报 PCA dim、`v_max/γ/F_c/δ_max`，故只冻结 [typed paper contract](artifacts/2026-08-16-rtcf-paper-release-audit.json)，等 π0.5 产生 same-policy success bank 后再实现 |
| [LifelongVLA](https://arxiv.org/abs/2607.14852) | 用 short/long LoRA 与 task-aware gate 平衡新 skill 学习和旧 skill 保留，并以少量缓存样本重采样 diffusion training signal | 为后期跨 episode architecture evolution 增加 parameter-memory 与 replay-cache 维度 | 它不是当前 episode memory，不能混入 anchor/sliding/key 对照；只有固定模块与 controller 成立后再进入训练架构搜索，且需 sequential-skill benchmark |
| [VLA-Pro](https://arxiv.org/abs/2605.29562) | 以结构化 procedure state 检索 task-specific LoRA，在每个 action chunk 动态融合、加载并卸载 adapter | 新增 `structured key → parameter-memory value → runtime adapter composition`，与 LifelongVLA 的 sequential update/anti-forgetting 不同 | π0.5 在 RoboTwin `40.4→59.3%`、RLBench `13.8→20.9%`；官方 OpenPI fork无训练权重且默认关闭真实 top-k retrieval，需 held-out task protocol 和独立 LoRA 训练预算，放在 controller 后的跨任务 architecture evolution 层 |
| [DriveVLA-M0](https://arxiv.org/abs/2608.10413v1) / [code](https://github.com/ZebinX/DriveVLA-M0) | memory case 保存结构 key、checkpoint-compatible policy inputs 与 expert/oracle labels；结构检索命中后，对当前场景临时训练 static/dynamic Action Decoder LoRA，下一场景重新初始化 | 新增 paper-only `retrieved_supervision_adaptation_case_memory`，以及 failure-only writer、dual-key retriever、similarity trigger、temporary adapter utilizer 与 reset lifecycle；它不同于 Mem-0 token injection、OptimusVLA action prior、VLA-Pro 预训练 adapter retrieval和 RoboTTT 持续 fast weights | 同一 NAVSIMv1 base 上 no-memory/map/map+agent 为 `91.0/91.7/92.3`，offline LoRA/full TTT/LoRA TTT 为 `91.2/92.4/92.3`。官方 release 只含 Base/Retrieve Model 与两个 gated checkpoint，未发布 memory bank、TTT/reset 或完整 runs；因此只冻结 [source audit](artifacts/2026-08-16-drivevla-m0-source-audit.json)，固定 π0.5 utility 前不制造 executable stub，oracle failure scorer也不得成为部署可选 writer |
| [Consolidator](https://arxiv.org/abs/2608.11701v1) / [code](https://github.com/swgoo/pmnet_consolidator) | 在显式 context boundary 将 routed STM 经共享 slot-local MLP 累积进 LTM；下一段既读取 LTM，又让 LTM 改变同层 slot routing | 复用 learned latent/recurrent payload，只新增 STM→LTM revision、LTM-conditioned addressing、跨段 reset lifecycle 与 identity intervention；独立轴是“memory 改变未来寻址”，不是再造一个 compressor | frozen PMNet 下 learned/forced-identity 为 `87.02/18.32%`；direct routing on/off 为 `87.02/44.38%`，immediate STM 同为 `89.90%`。官方 MIT source、31 checkpoints、30/30 runs 已公开，但仅为两段 synthetic task，需 PMNet phase slots 和跨 boundary 训练。故冻结 [source audit](artifacts/2026-08-16-consolidator-source-audit.json)，不新增重复 family，也不向正在跑的 π0.5 硬塞未训练模块 |
| [Memory for Attention](https://arxiv.org/abs/2607.23797) | persistent object map 根据语言相关性、last-seen 与 observed change-rate 决定有限预算下重看哪些对象 | 增加“memory 控制主动观测”而非“memory 只向 policy 提供 context”的独立轴 | 语言条件 change history 比 relevance-weighted recency 高 `2.5%`，但低观测可靠性下会反转；RMBench 没有可控视点/perception budget，当前只登记未来 active-perception controller，不伪造插件 |
| [BehaviorVLA](https://arxiv.org/abs/2605.22671) | 从完整 vision-action 轨迹学习 behavior prototype bank，episode 首次检索 global prototype，再用当前图像和上一 action 形成 local token并生成 action prior | 提供 GPM/action-prior family 中不同的 learned representation 与三阶段训练 recipe | 与 OptimusVLA 的 bank retrieval、progress alignment 和 action-prior utilization 重叠，不新增第二套同类 architecture；官方在线 local encoder 实际是 stateless，不能误记为 episode recurrent memory |
| [SkillMemo](https://arxiv.org/abs/2608.05970) | 以 MoE routing 隐式分段 demonstration，把 skill latent centroid→expert-routing sequence 存入 bank，再以 confidence-gated top-N retrieval 修正当前 routing | 增加不同于 action prior 和 symbolic skill 的 learned `skill-routing memory` | π0.5 LIBERO 平均 `96.8→98.0%`，但约增加 `0.3B` 参数且需联合训练；未公开代码，故只进入 Gate 1 后 learned 候选 |
| [Instance-Oriented Memory](https://arxiv.org/abs/2607.23702) | 首次 encounter 把探索 trace 蒸馏成去冗余 procedure，以对象实例为 key 跨 episode 复用，并作为可被反馈纠正的 soft bias 注入 policy | 新增“instance identity→procedure”与探索成本摊销目标，区别于 state/trajectory success retrieval | oracle 操作数减少 `16–30%`、VLM 恢复 `69–88%`；需要重复对象 protocol、instance identity 和 procedure-conditioned policy，当前 RMBench 不直接支持且代码未发布 |
| [Beyond Retrieval / QCR](https://arxiv.org/abs/2608.12847v1) | 固定检索结果后，将 selected trajectory 与当前 query/初始 observation 转成 workflow、需重取 bindings、适用条件、验证要求四字段 support | 复用 trajectory/procedure payload，只增加 target-bound reuse utilizer 和 fixed-retrieval utilization intervention；区别于 IOM 的 source-side distillation | QCR/完整旧轨迹 success `62.3/51.6%`，tokens `9.4k/18.4k`；large binding shift stale-binding `10.9/46.9%`。但 targets 由 source 受控改写、四字段未拆分、code/bank/raw ledger 未发布；因此只冻结 [paper audit](artifacts/2026-08-16-qcr-paper-release-audit.json)，用于后期 architecture Agent，不进入当前 π0.5 executor |
| [EchoVLA](https://arxiv.org/abs/2511.18112v3) | 用 calibrated RGB-D/pose 建立 coordinate-keyed persistent voxel map，以重建误差决定坐标级写入、EMA 修订，并按当前局部视锥检索 | spatial-map payload、短期 FIFO 和 dual-stream fusion 已有候选覆盖；只新增 reconstruction-discrepancy gate、coordinate EMA revision 与 frustum retrieval 三个可组合轴 | 同模型 mobile/static full→w/o scene 为 `17/21→9/16%`，但动态冰箱门任务 EchoVLA/π0.5 为 `40/50%`。标题匹配仓库仅含网页模板，未发布模型代码、权重或数据，且 voxel/pose/lifecycle 关键参数缺失；故只冻结 [paper audit](artifacts/2026-08-16-echovla-paper-release-audit.json)，待 out-of-view benchmark 和部署可见 RGB-D/pose 后进入 Agent search，不增加当前 family/executable |
| [DREAM](https://arxiv.org/abs/2606.00576) / [code](https://github.com/BJHYZJ/DREAM) | 保存与 SLAM keyframe 绑定的 RGB-D evidence；pose graph 修正后按 affected ratio 选择 local/regional/global reintegration，并对 observation archive 做 keyframe-aware pruning | spatial voxel payload 与 AtlasVLA/EchoVLA 重复，不新增 family；可执行 `SpatialLifecycleProgram` 组合 `PoseGraphReintegrationPlanner` 与 `KeyframeAwareObservationPruner`，把 pose-revision lifecycle、source-evidence capacity、feature-retention tier 和 dependent-cache invalidation 拆开，并对 calibrated RGB-D archive、optimized pose graph、reintegrator 三项 capability fail-closed | 整套 DREAM 对 DynaMem 的 real-robot long-term success 为 `50/80=62.5%` 对 `39/80=48.8%`，replay memory 为 `0.37–0.63 GB` 对 `1.32–10.43 GB`；但论文没有 RMP 单独消融，release 默认还关闭 global rebuild。故 [source audit](artifacts/2026-08-16-dream-spatial-source-audit.json) 只支持生命周期模块，不支持 `dream=true` 收益归因；当前 RMBench 无 mobile calibrated RGB-D/pose graph，preflight 保持不可选 |
| [MemoryVLA](https://arxiv.org/abs/2508.19236) | 分别保存 256-token perceptual memory 与 1-token cognitive memory，以 timestep-aware cross-attention 检索、learned gate 融合，并分别作为 diffusion 全局条件与逐层 action cross-attention | 已 source-audit 为 `dual_stream_perceptual_cognitive_memory`：双 encoder/bank、检索、gate、相邻合并、两种 utilizer、reset 与 ordered training 可独立组合；现有 adjacent-merge 负结果只覆盖 store 子算子 | 同 family `cognitive/perceptual/dual=63.5/64.6/71.9%`，`FIFO/token-merge=66.7/71.9%`；但没有 matched all-memory-off，公开 checkpoint 又实际 `update_fused=false`，完整路径约 `575.8M` 参数。因此先复现 33.5 GB 官方 checkpoint，再做 π0.5 `none/cognitive/perceptual/dual` 同预算联合训练，不能硬塞未训练模块 |
| [MemoryVLA++](https://arxiv.org/abs/2606.09827) | 在同一 PCMB 上加入 frozen world model 的 partial-denoise future latents，再用 memory context 选择性融合 | 只新增 prospective visual-state encoder/utilizer，不重复登记 MemoryVLA store | Mikasa-Robo memory-guided integration `44.4%`，直接 add `41.2%`；官方 code/weights 仍为 TODO，放在 Gate 1+ |
| [HALO](https://arxiv.org/abs/2606.25136) | observation/action K/V 全历史 + learned top-k sparse attention，并用 memory-dependent VQA 与动作 imitation 联合塑造 query/key | 登记为 source-audited learned recipe，不把已有 contextual-latent cosine top-k 改名冒充；需要新的 typed K/V payload、稀疏 attention utilizer 与 VQA training objective | 模拟平均 standard/no-VQA/HALO 为 `22/31/41%`，top-k 相对 full attention `+9` 点；官方代码与 QA 数据已发布、无 checkpoint，固定 memory gate 后做 `full / top-k / top-k+VQA / shuffled-VQA` 分层移植 |
| [MemoryWAM](https://arxiv.org/abs/2606.20562) | 用 2 个完整初始 frame、4 个完整 recent frame 和每个旧 frame 的 8 个 learned gist tokens 组成 hybrid KV memory | anchor/sliding 沿用已有组件；gist 登记为既有 `causal_consolidator` family 的 `per_frame_gist_kv` 训练 recipe，不新增同义 operator | Cover Blocks/Press Button 平均为完整 `92.5%`、w/o gist `40.0%`、full attention `91.5%`，支持把 gist 作为 Gate 1+ learned consolidation 强变体；官方 Apache-2.0 仓库尚未发布训练/推理代码，当前不做猜测式实现 |
| [AHA-WAM](https://arxiv.org/abs/2606.09811) | 慢 video-DiT planner 维护 6 帧 layerwise FIFO K/V，快 action-DiT 重用 planner context，并由当前观测通过 OVCR 每个 action chunk 重新路由；context 具有 version/age/stale 生命周期 | 登记为 source-audited `asynchronous_layerwise_planner_kv_context`，把 planner encoder、FIFO、current-observation router、action utilizer、异步 refresh/age guard/reset 和 phase-offset training 分成 typed operator；新增双速 producer-consumer 搜索轴 | 同 family 的 `Naive-Async / +KV / +OVCR / full` 为 `88.60/91.01/91.47/92.80%`。但 memory/context 新增约 `1.22B` 参数，官方 RoboTwin checkpoint 文档绑定的 plain config 又关闭 offset training；因此先复现并澄清 checkpoint/config，再做 π0.5 最小 port，不能把无训练 K/V cache 冒充该方法 |
| [WeaveLA](https://arxiv.org/abs/2606.17463) | 在 sub-goal completion event 将刚完成的连续 segment 用 8 个 learnable queries 压缩，并直接调制下一 subtask 的 π0.5 action expert | 已实现 `completed_phase_mean` + `fixed_completed_phase_handoff`，并新增 `key_completed_phase_handoff`；分别与同 planner 的 no-key/key baseline 配对，以 1-token 均值隔离 executor completed-segment handoff 及其组合价值 | 论文 6-task aggregate `19.0→24.7%`，SwingXtimes `32→56%`，最难 `N=3` slice `0→47.8%`；当前使用 RMBench oracle prompt-change，只是参数无关诊断下界，完整复现仍需 8-query pooling、memory-conditioned AdaRMS、staged training 与可部署 trigger |
| [SD-VLA / DySta](https://arxiv.org/abs/2602.03983) | 将视觉 token 分为跨时刻 static 与 dynamic，并用 learned gate 决定何时刷新 static KV cache | static/dynamic representation、KV lifecycle 和 utilization 是 anchor/novelty 之上的 learned 扩展 | 需要 token-level contrastive training 与 recache supervision；不新增免训练同义插件，Gate 1 后再比较 static cache 与普通 anchor/recent |
| [MemoAct](https://arxiv.org/abs/2603.18494) | 最近信息无损保存在短期区，最老 chunk 压缩后迁入长期区，长期区满时再相邻合并；当前 sensory query 检索后由 gate 融合 | 新增可独立搜索的“两级容量 + chunk migration”生命周期；本仓库已实现 `tiered_chunk_mean` lower bound | RMBench Put Back `41%`，高于 FIFO `23%` 和 MemoryVLA-style `34%`；完整方法仍需 learned compressor、retrieval、gate 与顺序训练，项目页尚无可用源码 |
| [PAM](https://arxiv.org/abs/2512.24638) | 分离 action/context frame query，并用 3 个 range-masked query 分别读取短、中、完整历史，再按 action→context 顺序注入 flow head | 新增不同于单一 top-k/FIFO 的 multi-timescale addressability；公开 RMBench checkpoint 的 Put Back 为 `67%` | 需 200k base + cache export + 200k post-training；源码无 license，当前只冻结 operator 接口，不复制实现 |
| [CycleManip](https://arxiv.org/abs/2512.01022) | 对高成本视觉历史做指数间隔与全局覆盖采样，对低成本 joint-state 保留全历史并预测任务进度 | 已实现预算匹配的 `temporal_multiscale` raw-token retrieve lower bound；完整方法再增加 dense pose-history encoder 与 progress objective | π0 四个循环任务由 `19/14/8/1%` 提升到 `72/69/47/41%`；当前官方代码尚无训练/推理实现和 license，因此插件只代表采样子算子 |
| [MAP-VLA](https://arxiv.org/abs/2511.09516) | 以 key-pose/DTW 对齐 demonstration stage，学习 stage soft prompt，并按局部状态轨迹检索 expert action prior | 新增跨 episode 的 stage-addressed prompt 与 dual-forward action-prior utilization | LIBERO-Long base 到 full 为 `76.4→83.4%`，但 prompt、检索和双前向均需训练/示例库，不能硬塞进现有 checkpoint |
| [History-Aware Point Tracking](https://arxiv.org/abs/2509.17141) | 持续追踪 object 上的 3D points，并把显式轨迹压成 object-history token | 新增不同于 contextual token 和 object slot 的几何轨迹 payload | 需要 RGB-D、标定、外部 tracker 和 learned compressor；官方代码为非商业 ShareAlike license，只登记 Gate 1+ 接口 |
| [Embodied-SlotSSM / LIBERO-Mem](https://arxiv.org/abs/2511.11478) | 以持久 object slots、slot SSM 和关系编码维护对象状态 | 新增 learned object-state family，并提供 object motion/sequence/relation/occlusion transfer benchmark | 需要 slot identity 与顺序训练；当前先登记 typed payload 和 benchmark，不把普通 ring 改名为 object memory |
| [LaMem-VLA](https://arxiv.org/abs/2607.07608) | short visual K/V vault 与 long action-hidden-state vault 分别 top-k，再由 query-conditioned condenser 压成 `8+4` 个固定 latent token，以 source tag 前缀织入 VLM reasoning | 复用既有 dual store、adjacent merge 与 reset，只新增 action-hidden writer、multimodal query、双路 top-k、dual condenser 和 native-prefix weaver；这些 operator 可独立组合，不能把整篇论文登记成一个开关 | matched ablation 为无 memory/仅 long/仅 short/双 memory：SimplerEnv `57.3/65.6/64.6/73.9%`、LIBERO-90 `92.1/95.4/94.8/97.0%`；native-prefix 又高于 external policy-side `73.9 vs 71.9`、`97.0 vs 94.8`。但官方 GitHub 截至 2026-08-16 仍只有 README；HF 虽有 30.7 GB 权重包，却没有代码、配置或加载方式，因此只冻结 [typed paper contract](artifacts/2026-08-16-lamem-vla-paper-release-audit.json)，不进入 executable/source-audited suite |
| [MEM](https://arxiv.org/abs/2603.03596) / [HyVLA](https://arxiv.org/abs/2606.14409) / [G0.5](https://arxiv.org/abs/2608.11739v1) | 用 dense video memory 与可更新 language memory 覆盖不同时间尺度；HyVLA/G0.5 都把 causal temporal-spatial mixing 放进 ViT，并只向上层保留 history-mixed 当前帧 token | 新增一个而非三个 source-audited `bounded_multiview_raw_frame_window` family：frame lifecycle、same-patch temporal mixer、current-token compressor、native-prefix utilization 与 whole-history-drop training 可分别组合；language branch 保持独立 | HyVLA 去除 compact encoder 后为 `90.9/90.1→88.8/88.6`。G0.5 公布了完整源码、6-frame/5-second/30%-history-drop recipe，CPU multiframe smoke 也通过；但论文无 matched memory-off，活动 configs 均为 `obs_size=1`，memory checkpoint config 又受 gated access 限制，server 还只在 chunk recompute 写 history。因此先按 [source audit](artifacts/2026-08-16-g05-mem-source-audit.json) 锁定 checkpoint/cadence，再做 π0.5 joint port，不能硬塞当前 checkpoint |
| [StemVLA](https://arxiv.org/abs/2602.23721v2) | 将历史 RGB 经 VGGT 提成隐式 3D geometry，再用 VideoFormer 跨时聚合；另以未来帧 VGGT feature 做 3D future distillation | 不新增第二个 video-history payload；只给既有 bounded-frame family 增加 `VGGT geometry encoder / VideoFormer aggregator`，future geometry 作为独立 training objective | 去 4D history 后 LIBERO-Long `86.0→83.5%`，去 future geometry 为 `67.0%`；但标题虽称 Open-Source，实际无 code/checkpoint/data，history/camera contract 未给全且 RGB-only 与 depth/point-cloud 描述冲突。因此只冻结 [typed paper contract](artifacts/2026-08-16-stemvla-paper-release-audit.json)，不进入 executable/source catalog |
| [LoHo-Manip](https://arxiv.org/abs/2604.21924) | 用 `(completed, remaining)` 文本进度记录和当前图像反复更新计划，再以 2D visual trace 指导短时 VLA | 支持把 progress-state lifecycle 与 planner→executor utilization 分开建模 | 文本记录与 MEM/τ0-VLA 已覆盖的 language/progress memory 重叠；visual trace 是需训练的 action-guidance interface，不新增同义 memory plugin，且论文未给文本 memory 的独立消融 |
| [Goal2Skill](https://arxiv.org/abs/2604.13942) | 用 episodic history、递归 working state 和 error register 支持 post-condition verification、retry 与 replan | 独立价值不是再增加一套 history store，而是把 `verify → outcome/error update → retry/replan` 纳入 lifecycle/controller 搜索维度；其 working-memory 消融在 3 个任务中 2 个下降，也直接支持 task-conditioned selection | base/+episodic/+working/full 平均为 `6.7/27.7/28.0/35.3%`，verifier 令 recovery `8.0→17.5%`；属于 VLM planner 闭环且无公开代码，Gate 1 前只冻结 reference program，不伪装成 executor memory 插件 |
| [eMEM](https://arxiv.org/abs/2606.03374) | 用 SQLite、HNSW semantic index 与 R-tree spatial index 统一支持按语义、空间和时间查询，并以分层 consolidation 维护长期具身世界记忆 | 为未来跨 episode 的 typed spatial/semantic/time payload 与 tool-level retrieval 提供成熟多索引参照 | 其任务是 ProcTHOR 长期事实回忆，不是 RMBench 闭环操控；当前固定相机输入也没有可靠空间坐标，因此不把它缩减成另一个 semantic retriever 插件 |
| [HAMLET](https://arxiv.org/abs/2510.00695) | 用 TCL moment token 与 causal Transformer consolidation 改造现有 VLA | learned consolidation 是不同于 ring/anchor 的候选 operator | 仍使用人工确定的窗口与融合结构 |
| [VPWEM](https://arxiv.org/abs/2603.04910) | sliding working memory 外，把离窗 observation 与历史 summary 递归压成固定宽度 episodic memory | 在 HAMLET 的局部 causal consolidation 上增加 `recursive summary state`，但复用同一 store/ordered-training family | 官方 Apache-2.0 代码可用；需 full-trajectory joint training，不是现有 checkpoint 的推理开关 |
| [Chameleon](https://arxiv.org/abs/2603.24576) | event token binding、token-wise slow trace、control-indexed recall、fast working state 与多尺度未来控制预测 | 把 addressability、slow/fast state 和 prospective objective 拆成可搜索维度，而非只按视觉相似度 retrieve | 完整 DSR/SR `80.8/71.3`，无 memory 仅 `20.4/17.6`；但需约 100k updates 联合训练且官方代码尚未发布 |
| [ECHO](https://arxiv.org/abs/2605.10993) | 将 VLA hidden state 编入 hyperbolic continuous hierarchy，以 semantic tree 做 coarse-to-fine retrieval 和后台 consolidation | 增加不同于 flat buffer/multi-bank 的 hierarchical store、tree retrieval 与结构更新维度 | 当前无官方代码，encoder、tree geometry 与 consolidation 需联合训练；应在简单 flat learned memory 有效后再进入搜索空间 |
| [MemER](https://arxiv.org/abs/2510.20328) | 高层 VLM 联合预测 subtask 与 recent-window keyframe nominations；重复 vote 经时间单链聚类压成中位代表帧，selected+recent visual context 服务下一轮规划，低层使用 π0.5 | 在现有 planner facade 中增加 `nomination writer / temporal vote-cluster store / selected+recent visual utilize`，与 Mem-0 的固定 boundary key history形成主动写入对照 | object retrieval `32→59/60`、wrong scoops `61→1`；但 keyframe supervision 仍用人工 per-subtask first/last/none 规则，公开仓库无 license，需按论文 contract 独立实现并单独计入标注/高层调用成本 |
| [Streaming GRPO memory planner](https://github.com/lucasburgett/vla-memory-new) | 用同 seed 的 K 条完整 episode rollout，将最终 progress 的 group-relative advantage回传给每个 decision point 的 subtask 与 keyframe nomination | 不新增与 MemER 重复的 keyframe family；新增可复用的 `delayed_outcome_streaming_memory_write_policy_optimization`，作为 OnEvo 未公开 online loss 的 source-level 对照 | 课程项目自报 ButtonUnmaskSwap SFT/GRPO 为 `26.2/31.4±3.1%`，persistent/reset/no-FIFO 为 `30.0/9.5/30.7%`；但无 checkpoint/raw log/per-seed records，且 oracle 执行 press/put-down并提供 phase history，不能与全自主 π0.5 直接比较。源码 48 项核心 tests 通过；见 [source audit](artifacts/2026-08-16-streaming-grpo-memory-source-audit.json) |
| [HiMe](https://arxiv.org/abs/2607.03449v1) / [code](https://github.com/HappyWaterXP/HiMe) | Sentry 用最近 8 帧判断 subtask 完成并唤醒 Planner；Planner 检索图文记录，生成 Query/Create/Update/Delete 并修订 procedural plan | 不新增重复 payload；保留 `subtask_completion_sentry_planner_trigger` 与 `planner_generated_multimodal_crud_lifecycle`，正好对应后期 Agent 自主管理“何时思考、如何修改 memory” | Sentry 令 transient `14→26%`、完整系统 `68→90%`；Query/Create-only 到完整 CRUD 为 `86→90%`。但 source 无 license/tests/executor assets，FIFO 论文/代码为 `8/20`，exact-tag Top-K 失效且无界 store、fail-open policy、跨任务共享和非事务 CRUD 都不满足 harness contract。因此只冻结 [source audit](artifacts/2026-08-16-hime-source-audit.json)，待 π0.5 key utility 成立后实现 versioned/evidence-linked/atomic 的 typed lifecycle，不复制 `hime=true` 总开关 |
| [VerMem](https://arxiv.org/abs/2608.03137v1) / [code](https://github.com/Sun-SYSU-24/VerMem) | 一个 learned policy 在 LTM Add/Update/soft-Delete 与 STM Retrieve/Filter/SelectEpisode/Summarize 七种原子操作中决策；局部 verifier 评价已实现的状态转移，全局 verifier 评价终局 memory 一致性 | CRUD、retrieve、filter、summary 与 episodic payload 已有，不新增 family；只保留统一 LTM/STM operation policy、versioned transaction、local/global verifier、operation-normalized hierarchical credit 与 `LTM→STM→joint` curriculum 六个 controller/training operator | Qwen2.5-7B 平均 `base 28.05 / no-verifier 41.76 / full 48.01%`，local/global/both 在三任务均互补；但官方 source 只有 operation schema、没有实际 STM/LTM runtime，发布的 verifier 是 rule heuristic 而非论文 DeepSeek-V3.2，reward loader 又使 verifier/credit 全部为零，且无 checkpoint/data/run。因此只冻结 [source audit](artifacts/2026-08-16-vermem-source-audit.json)，待 fixed π0.5 utility 成立后实现 harness-native typed transaction 与 verifier interface，不增加重复 payload 或不可执行空壳 |
| [MMPO](https://arxiv.org/abs/2605.30159) | 对每个 recursive textual memory 用“当前 progress + 仍缺信息/剩余步骤”anchor question 做一次 autoregressive probe，以平均 token entropy 评价中间 memory，再与 verified final outcome 组合成逐 turn reward | 新增 `progress_gap_anchor_question_builder`、`memory_conditioned_anchor_response_entropy_evaluator`、`verified_outcome_anchored_belief_entropy_reward` 和 `future_aware_turn_level_memory_advantage` 四个 evaluator/training 轴；不新增 payload/store，也不替代 rollout utility gate | RULER-HQA 56K 的 outcome-only/direct-answer/gap-only/progress+gap 为 `80.47/78.17/82.02/82.98%`，证明直接奖励低答案熵会促成过早自信；WebShop 为 MEM1/MMPO `70.87/77.25`。但只验证文本 memory，依赖语言 logits 与任务专属 probe，论文无 code/checkpoint/run ledger。故只冻结 [paper audit](artifacts/2026-08-16-mmpo-paper-release-audit.json)，不向 π0.5 latent memory 硬套；仅供 Gate 1 后 textual-summary controller 选择 |
| [From Signals to Structure](https://arxiv.org/abs/2607.00233v1) | 在相同 20-round window 上比较无持久 store、shared board、private scratchpad、10-slot codebook 与 codebook+meta，并扫描 signal capacity 和 history window | 五种 store 均与 sliding、shared KV、recursive language summary、bounded structured slots 重复，不新增 family；只保留 `architecture × capacity × history` sensitivity、组合 memory conflict/staleness trace 和“task utility 与 representation structure 分报”三个评测契约 | cap=25 的 scratchpad/memory-only 为 `0.867/0.747`，cap=64 为 `0.760/0.580`；cap=64 时 memory-only 把 window 从 20 加到 40 仍为 `0.52`。但它仅是单模型 symbolic signaling game，多数条件单 seed、无 code/data，stale/conflict 解释也未干预验证。因此只冻结 [paper audit](artifacts/2026-08-16-signals-to-structure-paper-release-audit.json)，不改变当前 RMBench gate；Gate 1 后只对入围架构做容量敏感性复核 |
| [Toward a Theory of Hierarchical Memory for Language Agents](https://arxiv.org/abs/2603.21564v1) | 将 hierarchy 分成 extraction `α`、coarsening `C=(partition π, representative ρ)` 与 traversal `τ`；代表项的 self-sufficiency 决定 collapsed search 还是 top-down refinement 更合适 | 不新增 payload 或总开关；保留 `partition / representative / traversal` 独立可选、self-sufficiency–traversal 校准、partition coherence/parent-child pruning error 三个接口与评测契约。现有 `AdjacentMergeStore`、`TieredChunkMeanStore`、`BoundaryChunkRetriever` 把这些选择部分融合，活动链结束后再拆开 | 论文只把 11 个既有系统映射到理论框架，明确称 matched `ρ/τ` 比较仍是 open problem；静态、query-agnostic hierarchy 假设又不覆盖最终的在线 architecture evolution。故只冻结 [paper audit](artifacts/2026-08-16-hierarchical-memory-theory-audit.json)，不制造 executable 插件，也不把 `b=e` 或 high/low SS 当作未经校准的默认值 |
| [HiMem-WAM](https://arxiv.org/abs/2606.10363v1) / [code](https://github.com/Agentic-Intelligence-Lab/HiMem-WAM) / [checkpoint](https://huggingface.co/Xiaoquan06/HiMem-WAM) | 从 motion latent 学习可变长 skill boundary；把当前 state、预测 skill 与低层 action chunk 编成事件，以 boundary-supervised write gate 写入，attention+稀疏 read gate 接入 policy | 不新增重复 payload family；把 `skill_boundary_event_token_encoder`、`boundary_supervised_sparse_memory_write_gate` 与 `teacher_forced_boundary_memory_warmup` 保留为 KC-VLA/observation-action memory 上的可组合训练轴 | RMBench 总分为 π0.5/HiMem-WAM `10.8/26.3%`，但它是整套 WAM 对不同架构，没有同 backbone memory-off 或 Stage-III-off 消融。官方 source 的简化路径未实现 Qwen/三阶段训练，Wan wrapper 又用九个 RMBench task rule 写 memory；公开 12.0 GB LIBERO checkpoint 的文档评测路径直接实例化 base Wan、没有构造 memory wrapper。false write 还会写零向量并占 slot。因此只冻结 [source audit](artifacts/2026-08-16-himem-wam-source-audit.json)，不新增重复 family 或 executable 近似；等 π0.5 fixed-memory utility 后再做 matched boundary-supervised joint training |
| [WorldScape Policy 2.0](https://arxiv.org/abs/2607.18840v1) / [project](https://manifoldai-research.github.io/WorldScape-Policy/) | 每个 action chunk 保存 VLM 当前感知 hidden states 与 4 个自回归 subgoal token hidden states；分别构造全局、最近、语义边界和压缩全历史视图，再由当前 planner event cross-attend | 新增 paper-only `planner_reasoning_trace_event_history` payload；保留 event encoder、gist condenser、multi-view builder 与 event-caption semantic forcing。它记录 planner “看到了什么、决定了什么”的连续轨迹，不等同于 Mem-0 文本 key、RGB keyframe、raw video window 或异步 planner K/V cache | 同训练设置的渐进消融平均分为 none/短时视觉/再加长期事件/再加 latent reasoning `40.91/44.67/46.25/47.89%`，长期事件增量 `+1.58 pp`。但官方 GitHub 只有 README，HF 只有 model card，数据未发布，且 gist/slot/window/boundary 等关键参数缺失；因此只冻结 [paper audit](artifacts/2026-08-16-worldscape-policy2-paper-release-audit.json)，不加入 source catalog 或 executable suite |
| [Mem-World](https://arxiv.org/abs/2606.18960v2) | 用计划 action chunk 经 FK 预测未来腕部视角；将 timestamp/task-relevance surfel 分时刻渲染，以可见性、目标相关性和新近性打分，再经 temporal NMS 选择历史腕部帧 | 不重复增加 spatial/raw-frame payload；只保留 `planned_action_chunk_future_wrist_pose_encoder`、wrist-only surfel writer、future-view retriever 与 temporal-NMS selector，作为 AtlasVLA/BridgeVLA++ spatial family 的新 query 轴 | 同一 random-context world model 上，wrist-view object consistency 为 recent/stride/W-VMem `0.401/0.463/0.502`。但这是 world-model 视频预测，不是 VLA 部署 memory；`58→72%` policy 增益来自人工筛选 synthetic data 后重训。论文无 code/checkpoint/supplement，且当前 Put Back 无移动 wrist calibration，因此只冻结 [paper audit](artifacts/2026-08-16-mem-world-paper-release-audit.json)，不进入 executable/source catalog |
| [MEMORA](https://arxiv.org/abs/2607.14252v1) | 从 10 秒 egocentric clips 形成 environment/entity/activity 三类 episodic store，以 Add/Update/Delete/Noop 维护 object identity 与 `state_history`，再跨 episode 整合 routine/habit/preference；读取时把 procedure 与 object/place grounding 分路，并可按证据时间回滚状态 | 四类 payload 复用既有 object/scene/action-transition/skill memory；只保留 `evidence_linked_entity_state_history_editor`、`evidence_time_snapshot_rollback_retriever`、`evidence_linked_cross_episode_regularities_consolidator` 与 `procedural_grounding_typed_query_router` 四个 lifecycle/retrieve operator | matched EAM-QA 的 episodic/full 为 `60.1/74.5%`，consolidation `+14.4 pp`；但 Qwen planning Replay 上 full 反而为 `0.338<0.345`，说明 consolidation 不能默认常开。项目页当前 404，code/checkpoint/benchmark 尚未发布；2/2 Unitree 仅是 fixed controller 定性演示。当前 Put Back 又没有 participant-level cross-episode target，因此只冻结 [paper audit](artifacts/2026-08-16-memora-paper-release-audit.json)，不新增重复 family 或不可验证 executable |
| [Mimir](https://arxiv.org/abs/2608.04933v1) | 独立维护 ordered task agenda 与 evidence-linked world hypothesis tree；每步用 active goal 检索有界候选，将 object/source/target/evidence 绑定后才规划，无法形成合法 binding 时 fail closed；observation 与 action feedback 分路更新 world state | task/world payload 复用 scene graph、object memory、revisable progress 与 error memory；只保留 `feedback_supported_observation_action_world_state_writer`、`postcondition_verified_task_progress_agenda_updater`、`active_goal_bounded_world_hypothesis_retriever` 与 `fail_closed_active_goal_world_evidence_grounder` | Qwen3-VL-8B full/去 world/去 task 在 EB-Habitat SR 为 `65.0/12.5/54.0%`，在 EB-ALFRED 为 `57.0/52.0/34.0%`，支持 world/task 分工；但未单独消融 dynamic grounding，且 code/checkpoint/raw runs 未发布，postcondition/adapter 是否用 privileged state 也不可审计。因此只冻结 [paper audit](artifacts/2026-08-16-mimir-paper-release-audit.json)，等待 π0.5 key utility 与 deploy-visible feedback contract，不新增总开关 |
| [HAM-VLN](https://arxiv.org/abs/2607.29600v1) | 对 subgoal 相关 place seeds 做一跳 topology expansion，并将 explicit backtrack 的 abandoned path 与失败理由绑定到 place；失败证据仅在该区域被重新检索时返回 | 复用 recent visual、progress、scene/object graph、error 与 joint action/write payload；只新增 `seeded_one_hop_topological_subgraph_retriever` 和 `abandoned_path_place_failure_evidence_writer` 两个可组合 operator | 同 planner/grounder/controller 的 full 相对最佳 raw-history SR/SPL 为 `61.0/48.1` vs `58.3/44.7%`；去 episodic/semantic/reflection 均下降。但无 code/checkpoint/trace，graph 与 prompt 细节不完整，且依赖 RGB-D/pose/navigation backtrack。故只冻结 [paper audit](artifacts/2026-08-16-ham-vln-paper-release-audit.json)，不新增第 17 个 family，也不在当前 fixed-camera RMBench 上伪造输入 |
| [GESTO](https://arxiv.org/abs/2608.10886v1) | 在可替换 4D scene graph 上增加两层 activity graph：原子 `(human, object, action, interval)` 交互组成完整、有序的 goal-event partition，并以显式 link 连接 persistent object/place；可双向查询 object/place→event 与 event→object/place | 新增 paper-only `grounded_hierarchical_human_activity_event_memory` payload；拆成 RGB-D interaction encoder、几何+语义 grounding writer、ordered event consolidator、unique-context unlinked refinement、event reassignment 与 bidirectional relation retriever，scene/object payload 继续复用现有模块 | full/去 event hierarchy 的 Text/Time/Space2Event 为 `0.71/0.70/0.73 → 0.52/0.40/0.53`；去 refinement 的 Time/Event2Space 为 `0.59/0.61`，支持 hierarchy 与 grounding refinement 的独立价值。但只评估回溯 QA，code/prompts/queries/graph outputs 尚未发布，当前 RMBench 也不包含被观察的人类活动；因此只冻结 [paper audit](artifacts/2026-08-16-gesto-paper-release-audit.json)，等待 shared-space RGB-D benchmark，不制造 Put Back 空壳插件 |
| [Spatial Memory Agent](https://arxiv.org/abs/2608.12743v1) / [project](https://aim-uofa.github.io/SMA/) | 首次 experience pass 把 verifier-scored rollout 反思为 `summary + transferable lesson`；后续每次检索按终局 verifier reward 更新 card 的 visit count、累计 reward 和 prior-shrunk Transfer Reliability Score，部署时冻结 bank/value | procedure payload 与 semantic retrieval 复用既有 skill/IOM/MEMORA family；只新增 `verifier_guided_rollout_to_transferable_lesson_reflector`、`one_pass_procedure_memory_writer`、`visit_evidence_shrunk_transfer_reliability_updater`、`semantic_then_transfer_reliability_reranker` 与 `read_only_deployment_memory_value_freeze` | 四个 frozen VLM 的五 benchmark macro average 相对最强非 SMA 基线分别提升 `+2.6/+2.9/+1.7/+2.8 pp`；Qwen3.6-27B RoboSpatial 完整/去 lesson/去 semantic filter 为 `68.5/65.0/62.7%`。但同一终局 reward 平摊给同时检索的三张 card，不是因果 credit；主表又从 10 个 calibration pass 选择最佳 deployment checkpoint，且 code/bank/runs 未发布。因此只冻结 [paper audit](artifacts/2026-08-16-spatial-memory-agent-paper-release-audit.json)，作为后期 Agent 管理 research lesson 的透明 value/rerank lower bound，不替代配对 utility gate |
| [R4DSG](https://arxiv.org/abs/2608.11017v1) / [project](https://dualtransparency.github.io/R4DSG/) | 在无 global map 的 monocular RGB 中维护 persistent object identity，推断 static anchor / dynamic object，并把动态对象写成相对 anchor 的 distance/direction/state；仅当 dominant anchor change 持续存在才提交 source→destination transition，再形成 segment-level relational documents | object/scene/action-transition/episodic payload 全部复用；只新增 conservative identity association、anchor-role inference、anchor-relative state encoding、persistence-gated transition write、segment relational write 与 option-blind retrieve 六个 typed operator | option-blind No-Transition→完整 no-why 的 overall/when 为 `34.9/34.7→37.3/43.1%`，说明 persistent identity+transition 联合有用，但两者没有独立消融。官方 MIT GitHub 完整 21-file tree 仅是静态项目页，论文声称的 released scene-graph outputs、代码、memory JSON 与 runs 均未发布；当前 fixed-camera RMBench 又缺少该 RGB tracking substrate。因此只冻结 [paper audit](artifacts/2026-08-16-r4dsg-paper-release-audit.json)，不新增重复 family 或空壳 executable |
| [DreamFly](https://arxiv.org/abs/2608.12308v1) | instruction-conditioned dense/region candidates 经跨观察证据 track 后，用 persistent 或 single-observation promotion 写入 16 个 `best-instance anchor + optional stable prototype + age` visual slots；当前 visual query 全部 valid slots，再通过 learned gated residual 利用 | 新增 paper-only `instruction_conditioned_evidence_promoted_visual_landmark_slots`，并拆出 candidate encoder、evidence tracker、promotion controller、anchor/prototype store、`decision_read_before_write_memory_boundary` 和 trained utilizer；与 Mem-0 initial anchor/sliding、phase keyframe 均不同 | memory-only SR/SPL `21.55/16.09→24.11/19.85%`，full/去 memory `31.46/27.17→19.60/18.76%`。论文的 causal 只表示 `t` 时刻只能读取 `<t` 历史，不改变本项目研究主题；adapter 需要联合训练，代码/checkpoint 及 FIFO/threshold/write utility/prototype/retention 规则未发布。故只冻结 [paper audit](artifacts/2026-08-16-dreamfly-paper-release-audit.json)，不向现有 π0.5 硬塞未训练模块 |
| [StreamFlow](https://arxiv.org/abs/2608.10949v1) | raw RGB 在 ViT 前按 GOP I-frame residual 筛动态 P-frame patches；旧 GOP latent 以相邻 reference-anchored sparse merge 保持固定容量；autoregressive generation 中视觉 attention mass 低时，按 reasoning prefix 检索并动态插入压缩历史 latents | 复用现有 raw-frame/patch/latent/consolidator payload，只保留 pre-encoder residual selector、adjacent sparse consolidator、generation-prefix max-frame retriever、VAS read controller 与 post-prefix injector 五个 operator | RTVU full/去 mid/去 long `81.55/76.86/80.18%`，VideoMME-Long `62.11/60.33/51.67%`，说明双时域互补；但 VAS 比预算匹配 delimiter/random 仅 `+0.23/+0.49 pp`，且依赖 autoregressive token attention 与 trained compressor。无 code/checkpoint/labels，当前 π0.5 也没有相同 read 控制点，因此只冻结 [paper audit](artifacts/2026-08-16-streamflow-paper-release-audit.json)，不新增重复 family 或 executable |
| [MemCtrl](https://arxiv.org/abs/2601.20831) | 学习决定当前 observation 是否写入 memory | controller 应先从 write/use gate 的有限选择开始 | 主要学习局部写入控制，且短时任务收益下降 |
| [OnEvoMemory](https://arxiv.org/abs/2608.08749) | 用 episode-local short/elite/transition 三 bank、read-before-write、action-conditioned value writer 与成功/失败 online rollouts 更新 frozen VLA 的 memory modules | 是 learned write、outcome-driven online update 和 semantic∪recent retrieval 的直接强基线；保留 `value encoder / elite writer / value-delta writer / three-bank store / rollout update` 五个可组合轴 | memory architecture 仍由作者固定，只演化参数；RMBench 仅测 SwapBlocks/SwapT，online 相对 offline 为 `10→14%`、`8→10%`，且没有逐项拆分三 bank、value writer 与 gated read。官方 code/checkpoint 未发布，容量、阈值、Top-K、merge/eviction 和 online loss 只出现在源码的 TeX 注释块，不能当作 v1 复现规范；见 [paper audit](artifacts/2026-08-16-onevomemory-paper-release-audit.json) |
| [DiM-WAM](https://arxiv.org/abs/2606.27677) | 用多 bank、novelty-aware retention 与累计质量加权融合维护有界事件记忆 | 提供不同于 FIFO 和普通平均合并的 store operator，并显示 RMBench 上完整事件记忆的潜力 | 完整收益依赖 WAM、多 bank learned query、diversity/progress supervision，不能直接归因于无参数 store |
| [KEMO](https://arxiv.org/abs/2606.23589) | 用 joint-motion causal peak、DINOv2 视觉去重和 refractory period 选择事件帧；每帧池化为 16 个 SigLIP tokens，再以 masked cross-attention/gated residual 接入 π0.5 | 已实现延迟确认/回写的 `causal_kinematic_peak` lower bound；Cover Blocks 的 `w/peak/r=30/20/8` 与论文表值对齐，后续把 visual dedup、keyframe encoder、fusion 与 loss weighting 分开 | 六任务同训练 recipe 的 π0.5 `27.8%→51.4%` TSR、`42.3%→76.4%` SCR；但 event selection 单独只在 Cover Blocks 达 `3/12`，加 keyframe-aligned loss 后为 `9/12`。项目页代码仍为 coming soon，且 peak window 端点/tie 规则未公开，所以现有 writer 不冒充忠实复现 |
| [EventVLA](https://arxiv.org/abs/2606.20092) | 从 action-conditioned hidden state 预测未来 chunk 的 keyframe 写入时刻 | 为后期主动写入 controller 提供 foresight-based 方案 | 依赖离线 keyframe 标注、learned head、NMS/cooldown 与联合训练 |
| [Keyframe-Chaining VLA](https://arxiv.org/abs/2603.01465) | 以 3 帧 RGB、固定 task id 和递归 phase 学习状态转移事件；高分候选先立即供 policy 使用，连续 5 帧低分后才确认并推进 phase；读取只是“最多 5 个历史关键帧 + 当前帧”的有界时序前缀 | 登记为 source-audited `task_phase_keyframe_history`，把 phase-metric encoder、task/phase FiLM event encoder、候选 latch、确认 writer、多视角+state store、prefix utilizer 和 reset/training 拆开；新增的是 progress-conditioned **write**，不是把 fixed keyframe 或 dynamic retriever 换名 | ManiSkill 平均 `no-history 16 / best fixed-stride 57 / KC-VLA 92%`，但 context prompt 单独贡献 `88→92`；官方阈值脚本用真值 keyframe 推 phase，而 rollout 递归用预测 phase，任务/阈值也硬编码且仓库无 license。故先做官方 checkpoint 复现和 recursive-prefix calibration，再移植到 π0.5，不进入当前免训练 suite |
| [BPP](https://arxiv.org/abs/2602.15010v2) / [project](https://bigpicturepolicies.github.io/) | 以任务专属二值语义事件的 rising edge 去重写入 keyframe；训练时只暴露 `k≤t−Δ` 的帧，匹配异步 VLM 在部署时 3–5 秒后才返回的可用性 | keyframe payload 与 KC-VLA 重叠，不新增 family；只保留可组合的 `binary_semantic_event_rising_edge_writer`、`detector_latency_aligned_keyframe_availability_mask` 和共享 train/deploy detector provenance contract | 四项实机平均 current/naive/PTP/BPP 为 `14.4/12.8/31.8/53.6%`，BPP 比最佳对照高 `21.8 pp`；但论文未单独消融 latency mask，也没有 code/checkpoint/data/run ledger。故只冻结 [paper audit](artifacts/2026-08-16-bpp-paper-release-audit.json)，待 fixed utility 后在 context-bank 构建与异步 rollout 两端同时实现，不能只加 runtime delay 或硬塞现有 π0.5 checkpoint |
| [TRACE](https://arxiv.org/abs/2606.14551) | 用已执行 robot-state trajectory 的 order-sensitive path signature 作为独立地址，把 visual/state evidence 路由到固定 learned slots，再选择性读取 | 已登记 `trajectory_addressed_evidence_slots` typed family，新增不同于视觉相似度、时间、task label 与普通 recurrence 的 address/update 维度 | 相同 regression base 的 no-memory/unrouted/full progress 为 `25.50/52.17/69.23`；官方代码无顶层 license/checkpoint，且 `auto` fallback 不保序、deploy address history 无界。故 Gate 1 后才做真 signatory、bounded streaming、ordered training 的 π0.5 port；论文 `π0.5=50.47` 只是 comparator，不是 TRACE-on-π0.5 |
| [ChainVLA](https://arxiv.org/abs/2608.02326) | 联合维护 observation-derived progress context 与上一 action chunk 的未执行 Motion Tail | 说明搜索空间不应只含视觉历史，还应允许 prediction-derived prospective action state | tail 依赖 decoder 初始化、token conditioning、ordered training 与 overlap loss，不能作为免训练开关 |
| [Action-Effect Memory (AEM)](https://arxiv.org/abs/2606.12499) | 交错编码历史视觉与已执行 action，并以 masked vision-action reconstruction 学习动作造成的状态变化 | 增加与视觉 ring、语言 key 和 prospective Motion Tail 均不同的 retrospective action-effect representation | 需要独立 32-step 预训练、joint Mamba 与 causal action alignment；当前无官方代码，不能作为现有 checkpoint 的配置开关 |
| [CAMP](https://arxiv.org/abs/2606.21188) | 用 DCT past-action reconstruction + temporal consistency 预训练 recurrent history encoder，经 VQ 得到 compact behavior code | 增加最小的 executed-action-only retrospective memory，可与 AEM 的 vision-action joint 表示直接对照 | Memory-T Multi-Goals `56→94%`，但 VQ、DCT 和 warm-up→joint staged training 都不可省；无公开代码，不能把普通 action FIFO 称为复现 |
| [VQ-Memory / RuleSafe](https://arxiv.org/abs/2603.09513) | 每 20 步用 50-step joint-state window 产生 VQ token，再把 256-code codebook 后聚类为 4 个 phase-like token，以论文设定的 40-token memory 作为专用 language IDs 注入 VLM | 新增 `discrete_proprioceptive_phase_tokens`：state writer、VQ encoder、code-cluster mapper、bounded token store、vocabulary-prefix utilizer 与两阶段训练均可独立替换；与 TFP continuous belief、CAMP action code 和 visual latent 都不重复 | matched π0 单任务中 rule_020 无/raw/VQ 为 `0/0/45%`，20-task 平均 `25.0→56.3%`；但官方项目页截至 2026-08-16 仍明确 Code/Dataset/Model Coming Soon，论文又未公开 VQ-VAE architecture、normalization 与在线 padding/truncation。因此先冻结 [typed paper contract](artifacts/2026-08-16-vq-memory-paper-release-audit.json)，不能用任意 discretizer 冒充复现；本仓库已有部署可见 14D robot-state stream，Gate 1 后可直接进入本地 tokenizer 训练分支 |
| [FM-VLA](https://arxiv.org/abs/2607.18231) | 把全 episode 6-axis wrench history 经预训练 VAE 压成 8 个 force tokens，并把短期 state history 追加到 π0.5 action expert | 增加视觉/语言无法替代的 contact-event sensor memory；论文平均 `27.8→83.3%`，视觉 MEM 对照 `53.7%` | 需要 wrench 数据、Force-VAE 预训练和 joint finetune；当前 RMBench adapter 不暴露部署可见力觉，故先冻结接口、不在 Put Back 上伪造 signal |
| [Gated Memory Policy](https://arxiv.org/abs/2604.18933) | 用当前 RGB/proprio 预测逐 timestep 二值 memory-read gate；标签来自 held-out split 上 all-off/all-on policy 的 action-error 比值，冻结 gate 后重训最终 policy | 提供与固定 `AllPathsController` 不重复的 learned read-controller 强基线；官方 MIT source 已审计并拆成 gate encoder、read controller、error labeler 和 frozen-gate retraining | 只有固定 memory 在 π0.5 上先通过 utility gate 才训练；论文明确指出 Continuous Place Back 常不需要 gate，且 release 当前未真正跳过 attention 计算，不能提前把 gate 当作 Put Back 解法或算力收益 |
| [Worth Remembering](https://arxiv.org/abs/2606.03787) | 用 V-JEPA-2 video latent 的 rolling distributional surprise、MAD threshold 与 NMS 选择稀疏 episode | 增加不同于 current-vs-last cosine novelty 的 learned distributional write 维度；相同 budget 下 QA accuracy `0.761→0.796`，仅保存 `1.7%` frames | 只在长期 QA/event segmentation 验证且无官方 code；作为 Gate 1+ writer contract，不进入当前 π0.5 fixed suite |
| [Analytic Concept-Centric Memory](https://arxiv.org/abs/2606.29774) | 将 persistent object、scene graph、action transition 与 executable skill 组织成结构化 memory，并用 category/part/affordance hard filter 与 state/applicability ranking 检索 | 增加不同于 flat token/keyframe 的 structured-symbolic payload、object identity、transition 与 skill-memory family；直接在 RMBench 报告结果 | 依赖 RGB-D、分割、6D pose、analytic template/primitive API；无公开代码且未做四类 memory 的组件消融，因此只登记 Gate 1 后接口，不能现在伪装成 π0.5 token plugin |
| [BridgeVLA++](https://arxiv.org/abs/2608.05042v1) / [code](https://github.com/BridgeVLA/BridgeVLA) / [checkpoint](https://huggingface.co/datasets/LPY/BridgeVLA) | temporal memory 组合 anchor/recent-2/learned sub-goal keyframe；spatial memory 冻结初始 colored point cloud，并按当前 coarse waypoint/zoom 重渲染为对齐局部 reference，在 fine stage 做对应视角 cross-attention | temporal operator 已有，不重复登记；新增第 15 个 source-audited `viewpoint_aligned_canonical_pointcloud_anchor`，拆为 writer/store/rerender retriever/encoder/utilizer/reset/joint-training 七个 typed operator | 同 backbone RMBench full/去 spatial/去 temporal/none 为 `96.0/95.4/21.3/18.9%`；Put Back 为 `100/100/38/1%`，Cover Blocks 为 `99/91/5/3%`。官方 9.87GB Put Back full checkpoint/config 与源码已核验，CPU mask/gradient/lifecycle smoke 通过；但消融 checkpoint 未发布，memory 模块从头训练且开关绑定 checkpoint，所以先冻结 [source audit](artifacts/2026-08-16-bridgevla-plus-source-audit.json)，待校准 RGB-D/coarse waypoint 与 matched joint training 后移植，不能硬塞既有 π0.5 checkpoint |
| [SERF](https://arxiv.org/abs/2606.12956v1) / [policy code](https://github.com/ExistentialRobotics/SERF-VLA) / [mapping code](https://github.com/ExistentialRobotics/SERF-mapping) / [checkpoint](https://huggingface.co/byeonghyunpak/SERF-VLA) | 将 object-rigid-tracked 环境点与 URDF-FK 更新的关节机器人表面点放入共享 neural-point latent map；分别从 base `1/2/4m`、左右末端 `0.5m`、robot-only、environment-only 与 global 读取，再作为 8 个 π0.5 prefix token | 动态环境 map 与 AtlasVLA/EchoVLA 重叠，不重复登记；新增第 16 个 source-audited `articulated_robot_environment_relational_neural_point_state`，把 robot/env writer、共享 store、task-instance filter、五组独立 retriever、Point-Transformer encoder、prefix utilizer、reset、map pretrain 与 LoRA joint training 拆开，供 Agent 选择 branch 子集 | 同 π0.5 的 image/static/dynamic-env/full 平均 task progress 为 `44.0/54.0/55.4/58.7%`，显式 robot-body state 相对 env-only 为 `+3.3 pp`；Task 22 去掉 base/EE/robot/env/global 分别下降 `3.6/6.1/6.1/1.6/5.0 pp`。两个 MIT repo、full/image checkpoints 与 map assets 已核验，8/6-token official source smoke 通过；但使用 prior map 和 privileged instance labels，公开权重没有 env-only/token ablation，branch 尚未独立配置。故先冻结 [source audit](artifacts/2026-08-16-serf-source-audit.json)，等 mobile calibrated RGB-D、URDF/state 与 matched 20k-step training 后移植，不在当前 Put Back 上做空壳插件 |
| [PhysMem](https://arxiv.org/abs/2602.20323v6) / [code](https://github.com/haoyangli16/PhysMem) | 将 action-level 物理交互写成 episodic evidence，经 cluster 生成 `AVOID/PREFER/SEQUENCE` hypothesis；用支持/反证、置信度和 targeted interaction 验证后提升为跨 episode physical principle，并折叠原始 evidence | 新增第 17 个 source-audited `verified_physical_principle_hypothesis_lifecycle`；procedure/transition/semantic retrieval 复用现有 family，只保留 action attribution、resonance/surprise gate、targeted experiment、promotion/refutation、evidence-linked principle store、symbolic→semantic retrieval 与 decay/folding lifecycle | brick insertion full/no-memory/direct-retrieval 为 easy `89/83/48%`、medium `76/53/23%`、hard `39/28/8%`；去 resonance/verification/working memory 在 medium 为 `58/64/69%`。官方 quickstart 运行 200 episode 得到 3 条 principle，但不是论文复现；源码的 bounded eviction、decay pruning、resume store binding、主动验证 promotion 和 trigger-aware retrieval 均有可复现缺口，且无论文 checkpoint/data/run。因此只冻结 [source audit](artifacts/2026-08-16-physmem-source-audit.json)，修复 parity 并具备 action-level outcome/safe experiment 接口前不加入 executable suite |
| [AtlasVLA](https://arxiv.org/abs/2608.06729v1) | 腕部 RGB 经 streaming depth 与相机/robot-state 标定回投到 global 3D，在 voxel hash 中按 depth confidence、历史权重衰减、时间窗和首帧 anchor 持续更新 latent world state | 新增 `voxel_hashed_spatiotemporal_world_state` payload 和六个 spatial encode/write/store/retrieve/utilize operator；ego intent bank、相邻合并和历史 cross-attention 与 MemoryVLA 重叠，直接复用而不新增 family | matched `without world / without ego / full` 在 LIBERO 为 `93.5/95.0/97.6%`、真实长程为 `54.0/56.5/69.5%`，说明动态 world update 值得保留；但无官方 code/checkpoint，窗口、decay、confidence 与 eviction 未完整说明，且当前 Put Back 固定相机不适合验证。只冻结 [typed paper contract](artifacts/2026-08-16-atlasvla-paper-release-audit.json)，待 wrist out-of-view task 与标定/depth 输入可审计后实现 |
| [OpenSPM](https://arxiv.org/abs/2606.29936) | 存储 object-relative phase key poses，按语义与 geometry feasibility 检索，再用 SE(3) 迁移到新场景 | 增加可执行的 structured spatial-action payload，与 visual scene anchor、symbolic skill 和 trajectory prior 均不同 | 完整 `85.6%`、zero-shot key-pose 替代为 `23.8%`，但 memory 与 3D perception、flow execution、terminal correction 强耦合且无公开代码，需拆分验证 |
| [μVLA](https://arxiv.org/abs/2606.12497) | 固定宽度 recurrent prefix token，经 self-attention 每环境步更新；TBPTT/EMA、action-copy guard 和 reset 均可独立消融 | 登记为 source-audited `recurrent_token_state` family；与 sliding 和 Chronos SSM 分开，需 ordered matched training，不加入免训练 suite | MIKASA matched training `0.42/0.48→0.84`，matched-semantics held-out `0.07→0.23`；K=2 cue recall 显著优于 K=1/K=8/EMA，且 chunk-rate inference 会严重退化，说明 update cadence 是架构变量 |
| [TFP](https://arxiv.org/abs/2607.08283) | 用真实 elapsed time 驱动 LTC task-progress belief，并通过 AdaLN 直接调制 π0.5 flow decoder | 把 update dynamics 与 utilization channel 变成可独立搜索维度；官方代码和训练 loader 已公开 | 是需要 episode-ordered training 的联合 learned system；完整方法不能作为当前 checkpoint 的推理插件 |
| [RB-VLA](https://arxiv.org/abs/2602.20659v2) | 将上一时刻 belief、最近 5 帧、已执行 action 与 proprioception 递归压成 256 维随机 belief，并以 EMA `t+1/t+5` latent prediction、KL 和 inverse dynamics 预训练 | 不新增与 TFP/μVLA/Chronos 重复的 belief payload；只保留 `stochastic action-conditioned update / multihorizon predictive pretraining / inverse-dynamics grounding` 三个可替换轴 | 两物体 P&P matched component ablation 为 `32.5→57.5→62.5→77.5%`；但无 code/checkpoint/data，loss 和传感器契约不完整，还需要论文未在本项目暴露的 velocity/acceleration/force-torque。因此只冻结 [typed paper contract](artifacts/2026-08-16-rbvla-paper-release-audit.json)，固定模块 Gate 1 前不做猜测式实现 |
| [Chronos](https://arxiv.org/abs/2606.30318) | 每个物理步形成 observation+proprioception state token，以选择性 SSM 在完整轨迹上递推历史状态，并保留全序列 temporal gradient | 新增与 LTC、EMA query、token bank 不同的 `selective_ssm_state_update`，且官方 RMBench code/checkpoint 可做 source-level 强基线 | 报告的 RMBench 均值为 `73.6%`，但 0.3B point-cloud policy 同时更换 perception 与 action generator，且没有 same-head memory ablation；先把 SSM state 与 IMLE/二阶 bridge/AR(2) noise/action ensemble 拆开，Gate 1 后再做最小 π0.5 matched port |
| [RoboTTT](https://arxiv.org/abs/2607.15275) | 每步用 K-V binding 自监督梯度更新 fast weights，以固定大小参数状态压缩长历史，并用 learned tanh gate 接回 attention residual | 增加不同于 token bank、EMA query 和 LTC belief 的 `fast-weight state/update/utilize` 搜索维度 | 需要 meta-learned initialization、sequence action forcing 和 TBPTT；当前无官方代码，Gate 1 前不能近似实现或硬塞现有 checkpoint |
| [StreamTTT](https://arxiv.org/abs/2608.13416v1) | 并行 short sliding KV 与 long TTT fast weights；input-dependent momentum/decay KVB update，并跨 forward 续接 conv prefix 与未满 chunk | payload、KVB 与 tanh residual 全复用 RoboTTT，只新增 momentum-decay updater 和 resumable-window lifecycle 两个候选 operator | joint sliding/hybrid 的 RT/ER 为 `68.19/49.58→78.85/59.55%`，但只有一 seed且无 state-content intervention；v1 明示 code will be released，当前无 code/checkpoint/raw runs与关键 state/hyperparameter contract。故只冻结 [paper audit](artifacts/2026-08-16-streamttt-paper-release-audit.json)，不新增 family/executable |
| [AURA-Mem](https://arxiv.org/abs/2606.02775) | 在固定大小 fast-weight state 上学习 action-utility sparse write，并用 target write-rate penalty 控制写带宽 | 复用 fast-weight payload，只新增与 `novelty_write`、GMP read gate 和 RoboTTT every-step update 均不同的 `action_utility_sparse_write_gate` 与 `write_rate` 搜索轴 | LIBERO-Long 只得到成功率持平且写入率 `1.0→0.142`，没有 task-success 增益；代码仍为 private，且论文披露 active-parameter 不匹配，因此仅登记为 Gate 1 后的训练型成本 Pareto 候选 |
| [ReMem-VLA](https://arxiv.org/abs/2603.12942) | 用 frame-rate 与 chunk-rate 两组 recurrent queries 做 gradient-free EMA 传播，以 12-layer bidirectional connector 供 action/hindsight query 读取，并用 ordered slot-streaming 训练 | 不新增第二个 recurrent-token family：复用 μVLA 的 recurrent query/state/reset/TBPTT contract，只新增 `chunk_rate_ema_scheduler` 与独立的 Past Observation Prediction objective | matched 128-query factorial 的无/frame/chunk/dual 平均为 `17.75/87.75/84.5/94.5%`，证明双时间尺度有价值；但它是 Qwen3-VL-2B + diffusion 的 150k-step 联合训练系统，POP 又只在 visual-memory real task 中使 `34→82%`。截至 2026-08-16 未发现官方 code/checkpoint，因此最小 recurrence 有效后才依次加入 chunk-rate 与 POP，不能把整篇论文登记成新开关 |
| [TempoFit](https://arxiv.org/abs/2603.07647) | training-free 地缓存中间层 prefix K/V，以 K-to-K + frame-gap bias 检索并保范数 residual 注入 | 提供与 token cross-attention 不同的 layer-wise KV-native utilize 维度 | 论文 π0.5 报告 `92.6→96.6`，但官方仓库仍为 Coming Soon；当前只实现明确标注的 contextual-latent content+recency proxy |
| [Notes-to-Self](https://arxiv.org/abs/2602.21013) | 用 `Grounding/Plan/Act` 语言 scratchpad 保存空间状态与阶段进度，以 `<done>` 更新 | 为 MEM-style language state 提供具体 schema 与 event-write 方案 | 需要 scratchpad 标注和联合 action/text 训练，不是免训练 prompt 插件 |
| [Explicit Language Memory](https://arxiv.org/abs/2608.04765) | 递归更新 completed/current/next rolling text，并从 expert video、skill label、timestamp 自动生成监督 | 为已有 language-state branch 补充监督生成和 high/low branch 解耦训练 recipe | 与 MEM/Notes-to-Self/τ0-VLA 的语言进度 memory 重复，不新增 store；只在实现该 branch 时复用 recipe |
| [HyMeS](https://arxiv.org/abs/2608.09410) | coding agent 从 rollout trace 修改 symbolic stage、event verification 与 memory-conditioned steering code；显式维护 `plan/stage/binding/count/flag`，以 proprioception 与最多 5 帧 VLM 判断做 `3-of-5` progress vote | 直接证明 rollout-driven code-space memory learning 可行；同权重 π0.5 上总体 CSR/TSR 从 `52.5/41.3` 提到 `66.2/60.1`，one-shot→refined program 为 `63.3/53.3→71.8/71.7`。保留 symbolic state、verified update、program refiner 与 frozen-program selection 等高层 operator | 搜索对象是 task-specific heuristic program，并通过 constraint gradient 直接 steering 动作；memory、PACE 与 steering 没有完全分离。论文未公开代码、agent prompt/探索预算或修正后的 12-task protocol，因此只冻结 [typed paper contract](artifacts/2026-08-16-hymes-paper-release-audit.json)，作为单独计费的强参照，不进入 fixed-memory 或 source-audited suite |
| [ALMA](https://arxiv.org/abs/2602.07755) / [code](https://github.com/zksha/alma) | Meta Agent 从抽象 `update/retrieve` 接口出发，按候选成功率与访问次数从 archive 采样 parent，读取代码和分层成功/失败轨迹，生成、试跑、最多三轮调试并归档新 memory code | 作为最直接的 Agent-memory-search 强基线；已实现 content-addressed 的 ALMA-compatible `success − log(visit)` 非贪心 sampler，并保留 bounded success/failure trace reflection 与 typed sandboxed debug 两个搜索契约 | GPT-5-nano/mini 总体为 `12.3/53.9%`，相对 no-memory `+6.2/+12.8 pp`；ALFWorld greedy/ALMA 为 `11.9/12.4%` 与 `77.1/87.1%`。但它只搜索 prompt-injected text memory，允许任务专属任意 Python 与不同 LLM/DB 成本；released Docker 的 bind mount 可写且未禁网。故不登记 policy payload/plugin，详见 [source audit](artifacts/2026-08-16-alma-source-audit.json) |
| [AgentCanvas / KDLoop](https://arxiv.org/abs/2606.30111) | 把具身 executor 表示为 typed node-and-wire graph，用 coding agent 循环执行 Think/Critic/Experiment/Distill，并与 ADAS/AFlow 在导航、EQA、VoxPoser 上对比 | 直接采用 typed port 的 rollout 前校验、原子 run 目录、post-selection rerun、干预轴覆盖记录和 leak rejection；KDLoop 作为搜索策略强基线 | 它搜索整个 method-seeded embodied graph，而非同一 π0.5 上的 memory-operator 空间；收益常来自 prompt/control/model 配置，且论文明确暴露 rollout noise、local basin 与不完整 credit assignment |
| [Auto-Robotist / When Search Becomes Memory](https://arxiv.org/abs/2605.25832) | 将已评测的机器人形态搜索轨迹蒸馏为包含结构原型、正负规则和支撑样例的可审计 skill library，并用 Add/Diagnose/Merge 更新 | Research Agent 应把每次 architecture trial 固化为“候选—证据—正负经验—适用边界”，检索这些经验辅助下一轮提案，而不是只保留 best score | 它搜索 EvoGym morphology，不搜索 VLA memory；因此只借鉴 research-memory 更新循环，不登记为机器人 policy memory operator |
| [τ0-VLA](https://tau0-vla.github.io/) | high-level policy 维护可 advance/rollback/retry 的 execution-progress record，并用扰动过的 demonstration memory 学习纠错 | 增加不同于 append-only key history 的 `revisable_progress_record / observation_conflict_update` lifecycle 维度 | 官方 release 尚未公开 high-level memory/TTC code；当前只进入 Gate 1 后候选，不自行猜测实现 |
| [MemEvolve](https://arxiv.org/abs/2512.18746) / [code](https://github.com/bingreeky/MemEvolve) | 在通用 Agent 上以 trajectory diagnosis 生成 encode/store/retrieve/manage Python，并按 success、API cost 与 latency 选择 survivor | 已实现 content-addressed、确定性的 `performance_cost_latency_pareto_survivor_selector`；保留 frozen architecture 的 cross-task transfer confirmation protocol | Flash-Searcher 上 GAIA/xBench/WebWalkerQA 相对 no-memory 为 `+4.24/+5.00/+3.53 pp`，但 release 默认关闭 Pareto，K/任务数与论文不一致，后续轮仍以初始 provider 作代码模板；生成代码先写 live tree，再在同进程 import，且无 paper runs/search seeds/core tests。故只作为搜索基线，详见 [source audit](artifacts/2026-08-16-memevolve-source-audit.json) |
| [ENPIRE](https://arxiv.org/abs/2606.19980) | coding agent 通过自动 reset、verification 和 rollout 改进机器人策略 | 提供 physical autoresearch harness 范式 | 搜索对象是广义策略/训练方法，不专门研究 memory architecture |
| [MARS](https://arxiv.org/abs/2602.02660) | 预算感知的自动研究与模块化代码搜索 | 说明昂贵实验必须显式计入预算与分支比较 | 没有具身 memory program 和机器人任务验证 |
| [EvolveMem](https://arxiv.org/abs/2605.13941) / [code](https://github.com/aiming-lab/SimpleMem) | 用失败日志驱动 retrieval 配置搜索，并带 best-so-far 回退与停滞探索；论文 LoCoMo `0.305→0.543`、MemBench `67.9%` | 作为 Gate 1 后 structured-config search 强基线；其诊断、候选冻结、survivor、research lesson 已分别由 ALMA/KDLoop、typed candidate、MemEvolve 和 EvoMem 覆盖，不新增插件 | 官方 runner 默认第 5 轮注入预写 terminal config；“新维度”、LoCoMo flags 与 expected-lift recipes 已硬编码，meta proposal 只落 JSONL，同一 QA 又跨轮参与搜索和评分。故必须在本项目 split-safe typed harness 中重跑，见 [source audit](artifacts/2026-08-16-evolvemem-source-audit.json) |

ALMA、MemEvolve、AgentCanvas/KDLoop、HyMeS、OnEvoMemory 与 VerMem 是当前最接近的 concurrent work，但边界不同：ALMA 在四个 text-action domain 中搜索任意 text-memory Python，并为每个 benchmark 单独提供任务描述；MemEvolve 在通用语言 Agent 中做 diagnose-and-design program evolution 与 Pareto survivor selection；AgentCanvas 搜索 method-seeded 整体 embodied graph；HyMeS 由 coding agent 修改 task-specific symbolic/steering program；OnEvoMemory 在固定三-bank architecture 中用 rollout 学习 value-guided selection 参数；VerMem 学习在一套固定 LTM/STM operation vocabulary 内执行哪项操作。E-MAC 的搜索对象则严格限制为**可复用、可替换的 VLA 历史信息处理架构**：候选必须通过同一 typed contract，不允许把任务名、阶段 plan、success rule 或专属 steering reward 写进核心 operator；改进要在 locked tasks、未见 lag 和状态变化上验证，并与 random/evolutionary search、ALMA open-archive sampling、MemEvolve Pareto survival、KDLoop-style trace-aware search、OnEvoMemory-style learned controller 及 VerMem-style unified operation policy 做候选数、rollout、GPU、LLM-token、memory-token 与 latency 预算匹配。ALMA 与 MemEvolve 是 Agent-memory-search 的必选直接基线；AgentCanvas 与 HyMeS 是整体搜索/任务专属强参照，但都不能与 fixed-backbone memory module 消融混成同一因变量。

## 4. Research Questions 与假设

### RQ1：固定 memory 的有效区间是什么？

在相同 π0.5 backbone、数据、训练更新数和评测 seed 下，anchor、sliding、anchor + sliding 与 key/subtask memory 是否分别适合不同的任务复杂度和 retention lag？

**H1**：不存在全局最优的单一固定 memory。短跨度任务更偏向 sliding/recurrent context，跨阶段决策更依赖 key/subtask memory；在当前观测充分时，额外 memory 可能无效或产生干扰。

Mem-0 自身的 Cover Blocks 消融已经显示这种任务依赖性：完整系统为 `68`，去掉 key 后降到 `5`，但去掉 anchor 或 sliding 分别为 `92` 和 `84`。因此该任务支持“key 必要”，并不支持“三种 memory 同时越多越好”；这正是本项目需要模块选择与架构探索、而不是固定 all-on 的直接动机。

### RQ2：有限 controller 是否优于静态组合？

一个只允许在 `none / anchor / sliding / anchor+sliding / key` 中选择的轻量 controller，能否优于最佳全局固定模块、按任务选择的静态 lookup 和全部模块开启？

**H2**：当同一任务内部同时包含观测充分和历史依赖阶段时，状态相关的选择会优于 task-level 静态选择，并减少不必要的 latency 和 context。

### RQ3：研究 Agent 能否发现更好的 memory architecture？

在相同实验预算下，能读取代码、论文、rollout trace 和实验表格的研究 Agent，能否比随机搜索、常规 evolutionary search 和人工 seed designs 找到更好的 memory program？

**H3**：受约束的 Agent 能通过组合或修改 write、representation、retrieve、utilize 和 forget operator，找到固定 catalog 之外、且可迁移到未见任务条件的设计。

## 5. Benchmark 与任务设计

### 5.1 RMBench 的实际边界

RMBench 正式任务包含 5 个 `M(1)` 和 4 个 `M(n)`，**没有原生 `M(0)` 任务**。因此 proposal 中的 `M(0)` 必须明确写成受控负对照，而不能称为 RMBench 官方任务。

- `M(1)`：优先选择 Put Back Block、Swap Blocks；它们适合分析 anchor 与短时 sliding context。
- `M(n)`：优先保留已有基础最强的 Cover Blocks，再加入 Battery Try；它们适合分析 key/subtask memory。
- `M(0)-control`：在相同场景和动作目标下保留关键线索可见，或在决策时重新展示必要 cue，使当前 observation 足以决定动作。该变体只用于检测不必要 memory 的干扰。

最小论文矩阵至少应包含两个 `M(1)`、两个 `M(n)` 和一个 `M(0)-control`。只做 Cover Blocks 可以完成工程 pilot，但不足以支撑“Agent 会按任务需要探索架构”的论文结论。

### 5.2 优势图谱的坐标轴

每条评测记录至少带有以下标签：

- memory complexity：`M(0)-control / M(1) / M(n)`；
- retention lag：关键 observation 到使用时刻之间的 action chunk 数或环境步数；
- current-observation sufficiency：当前图像是否保留完成决策所需信息；
- state change：物体位置、遮挡、颜色—位置 binding 或 task phase 是否发生变化；
- memory module：none、anchor、sliding、anchor + sliding、key；
- resource cost：memory token 数、额外参数、延迟、GPU-hours 和 rollout 数。

最终输出不是只给一个平均成功率，而是给出“任务条件 × memory module”的有效、无效和有害区间。

### 5.3 RoboMME locked transfer

RMBench Gate 1 通过后，引入 RoboMME，而不是现在分散训练资源。它包含 Counting、Permanence、Reference、Imitation 四套共 16 个任务，分别覆盖 temporal、spatial、object、procedural memory，并提供统一 π0.5 baseline 和 14 个 memory variants。搜索阶段只开放预注册的开发任务；其余 task/condition 作为 Agent 不可见的 locked transfer，验证发现的 program 是否跨 memory 类型迁移。第一步直接复用官方 checkpoint/evaluator 校验结果，再接 harness adapter，避免把 benchmark port 错误误判成 architecture failure。

## 6. Methodology

### Module A：稳定的 π0.5 no-memory baseline

对每个入选任务分别 fine-tune π0.5，并冻结：

- 数据、instruction、checkpoint 初始化和更新数；
- camera、action horizon/chunk、最大 episode steps；
- environment seed、policy RNG 和成功判定；
- 从 reset 开始的 full-episode protocol。

无记忆 baseline 不得使用 stage router、历史 phase、episodic prompt 或 memory token。oracle subtask prompt 只能用于确认 executor 是否具备基础动作能力，不能进入主结果。

### Module B：同 backbone 的固定 memory baselines

所有模块共享同一个 π0.5 初始化、policy 主体和训练数据，只改变 memory path：

1. **None**：只输入当前 observation 和 task instruction。
2. **Anchor**：保留任务或 subtask 起始时刻的视觉 token，在后续决策中持续提供。
3. **Sliding**：维护最近 `K` 个 observation embedding 的 ring buffer；`K` 作为预注册小网格。
4. **Anchor + Sliding**：同时提供长期起点 cue 与近期运动/状态变化。
5. **Key/Subtask**：在阶段边界保存关键帧或阶段摘要，由高层模块产生当前 subtask，再交给同一个 π0.5 executor。

这里区分两类 operator：anchor/ring/recent/reset 等 program operator 无需训练；fusion、video encoder、action-effect encoder、consolidator 和 planner update 等 learned operator 必须携带 checkpoint 与训练 recipe。忠实 Mem-0 executor 每步保存一个 VLM 最终层 contextual image latent，使用 anchor=1、sliding=30、相对位置和两个独立 Pre-LN cross-attention，输出 `[sliding_fused, anchor_fused, text]` 三个 action-conditioning token。anchor/sliding 消融共享同一个 full-memory checkpoint并通过 mask 实现，不得各自重训；key memory 则是 planner 侧多模态历史，必须单独计入 planner 训练与调用成本，不能伪装成相同的 latent 注入。AEM-style history 还必须携带与 observation 时间对齐的已执行 action；它是新的 representation family，而不是把 action 塞进现有 latent metadata。

Mem-0 的 key 消融必须按 released code 建立两个不同输入协议：完整 planner 读取 episode 初始图和有序的 completed-subtask 文本/结束图；`w/o key` planner 每次只读取当前 observation。由于输入分布不同，`w/o key` 需要独立 SFT checkpoint；用 key-trained planner 清空 history，或让 no-key 始终看初始图，都不是忠实消融。

训练表示必须由部署时的同一 program 按 `RETRIEVE → USE → WRITE` 顺序生成，不能使用另一套离线压缩表示。需要训练的新 operator 可以进入后期架构搜索，但必须预算匹配；Research Agent 不能把“多训练一个 checkpoint”当作无成本的配置组合。

训练分为两个预注册层级：单卡主实验冻结 2B contextual encoder，但训练完整 π0.5 action expert、Mem-0 fusion 与 projection；双卡资源可用时再运行全模型联合 fine-tune，并把“是否更新 contextual encoder”作为训练消融。两者不能混在同一 fixed-module 主表中。

为避免把“更大模型”误当成“更好 memory”，每个模块同时报告固定 token budget 版本；key/subtask 的额外 planner 调用次数和 latency 单独计费。

主结果继续使用统一固定 token budget，不能让 Agent 通过扩大 context 获胜。Gate 1 之后只对最佳固定模块和 Agent 最终入围 program 做预注册的 capacity sensitivity：在不读取 locked-test 结果的前提下，分别改变可注入 token 数与可寻址 history/store 容量，检查架构排序是否稳定。多 store、structured+summary 或 all-on 组合还必须报告重复、过期、冲突和 revision trace；表示覆盖率、压缩率或结构性指标与闭环 task utility 分栏，不能以“记住得更结构化”代替成功率证据。

### Module C：记忆优势图谱与内容干预

在固定模块评测后进行 remove、shuffle、cross-episode mismatch、stale memory 和 oracle memory 检查。其目的只是回答：

- 改善是否依赖正确的历史内容；
- 错误或过期内容是否会稳定干扰动作；
- 问题来自 memory 内容、注入通道，还是 π0.5 根本不会使用该通道。

这部分是 **memory mechanism validation**，不是论文主线，也不把项目改写成因果推断工作。

### Module D：最小 memory controller

只有固定模块通过 utility gate 后才训练 controller，并把两个不同问题分层处理，不能用一个总开关混在一起。

**D1：逐 timestep read gate。** 先只控制一个已经通过 Gate 1 的固定 memory program 是否参与当前动作，输出

`use_memory ∈ {0, 1}`

初版采用 GMP 的最小可复现协议：在独立 held-out calibration split 上，用预算匹配的 `all-off / all-on` policy 计算逐步 action error；当 `no-memory error > 10 × with-memory error` 时标为需要读取。gate 只读取当前 RGB、proprioception 和 task instruction，不读取 rollout success、未来动作、simulator task pointer 或 oracle phase。随后冻结 gate，以相同 optimizer/example 预算重新训练最终 policy，避免把额外训练量误计为 controller 收益。

D1 必须比较 `all-off / all-on / oracle-error gate / activation-rate-matched random gate / learned gate`，并报告 success、task progress、总体及分阶段 activation rate、`M(0)` false activation、实际 used token、policy latency 和 memory latency。只有 runtime 在 attention 计算前真正短路 memory path 时才能报告计算节省；仅在已计算的 cross-attention 输出上乘零不算省算力。

**D2：架构 selector。** 只有 D1 在 held-out 条件下提升成功率，或在成功率相当时稳定降低实际 memory 成本，才让更粗粒度的 selector 在 episode 或 subtask boundary 输出：

`none / anchor / sliding / anchor+sliding / key`

输入只能包含部署时可见的当前 observation、task instruction、已有 buffer summary 和资源状态，不能读取 simulator task pointer、task ID shortcut 或未来结果。初版 selector 保持小型，只回答“此时选择哪一种已有模块”，不生成新代码，也不同时学习新的 write/store/retrieve operator。

D2 必须比较：

- best global fixed module；
- validation 上选择的 per-task static module；
- all-on；
- random controller；
- learned controller。

其中 per-task static baseline 很重要，否则 selector 可能只学习 task ID，而不是真正根据状态选择。D1 与 D2 分栏报告：前者回答“当前是否需要读取一个已验证 memory”，后者回答“当前应使用哪套架构”，二者不能用同一个结果共同宣称 controller 有效。

### Module E：Typed Memory Program Space

通过 controller gate 后，把候选架构表示为可执行 program：

```text
encode(history item)
  -> write(trigger, representation, store)
  -> partition(units, affinity/structure)
  -> represent(group, compression)
  -> retrieve(query, store, top-k/abstain)
  -> utilize(channel, fusion, target module)
  -> update_or_forget(TTL, event, capacity)
```

初始 operator catalog 包括：

- write trigger：always、phase boundary、visual change、controller event；
- representation：raw keyframe、pooled visual token、native action-supervised token、perceptual/cognitive dual tokens、latent token、structured subtask、continuous-time belief、causal action-effect token、fast-weight state；具备 RGB-D grounding 与 skill backend 后再开放 object/scene/transition/skill structured records；
- store：anchor、ring buffer、two-tier recent/compressed store、key store、vote-preserving temporal cluster store、dual-stream store、cross-episode episodic store；在 flat learned memory 有效后再开放 semantic tree，在 structured payload 条件满足后再开放 object/scene/transition/skill stores；
- retrieval：recent、uniform-global、phase match、semantic、fixed exponential/global multiscale、range-masked multi-timescale query、content cross-attention、selected+recent visual context、spatial、hybrid、abstain；structured payload 后续增加 category/part/affordance hard filter 与 geometry/state/applicability ranking；
- utilization：prompt、input/prefix token、cross-attention、ordered action→context conditioning、adaptive gate、action-head AdaLN、dedicated memory expert、flow initial prior、gated fast-weight residual、planner-only；
- lifecycle：one inference、fixed steps、phase transition、capacity eviction、chunk migration、adjacent consolidation、recursive summary update。

`partition / represent` 是 hierarchy store 内部的 typed sub-operator，不替代顶层 `store`：store 继续拥有状态、容量与 parent-child provenance，但不能把 grouping、代表项生成和 traversal 永久绑成一个不可拆的名字。活动 Put Back/Cover Blocks 链结束后，现有 `adjacent_merge / tiered_chunk_mean / boundary_chunk` 通过 exact trace replay 迁移到该组合接口；不保留旧 fused 配置兼容层。self-sufficiency 只作为 workload-calibrated traversal prior，不能未经 matched `representative × traversal` 实验就禁止 Agent 组合。

所有 operator 都有明确输入输出类型、部署可观测性声明和成本统计。Agent 可以先组合/调参，后续才允许在 sandbox 中实现新 operator。

组合不是简单拼接 path 名称：utilizer 必须显式声明每个 history path 的 token quota，所有 quota 总和等于固定窗口预算；runtime 分别截取各 path 的有效输出，并在 `USE` 事件中记录逐路径 retrieved/used/dropped 数量与实际 item IDs。任何未分配路径或预算不完整的 program 在构建时失败，从接口层避免“Agent 选择了两个模块，但后一个静默覆盖前一个”的伪组合。

实现上按 payload family 逐层扩展，不预先构造一个可以容纳任意对象的弱类型容器。当前已端到端可执行的 executor contract 仅是 `dense_tokens[M,D] + mask[M]`，已覆盖 Mem-0、两级 recent/compressed store 与其他无训练 write/store/retrieve 对照；planner key 使用独立 typed facade。动作历史、force、routing profile、fast weights、point cloud 和 SE(3) pose 暂不冒充成已注册插件。首个异构候选进入实现时，同步新增它的最小 payload type、组件兼容性 preflight 和成本计量，而不改写 runner 或 evaluator。

首个非 Mem-0 write operator 已实现为 `novelty_write`：它按当前 latent 与最后保留 latent 的余弦距离决定是否写入，并设置最大无写入间隔。它不是 MemCtrl learned head 的替代，而是 controller 必须超过的无训练 active-write baseline；所有 write/skip 决策及理由进入统一审计。

首个多路 retrieve operator 已实现为 `semantic_recent_union`：在同一个 episode bank 中固定保留 latest-10，再从 semantic ranking 回填 20 个不重复项并恢复时间顺序。直接集合并集在 Put Back Block 上因平均重合 `8.48` 项只使用 `21.52/30` 个 slot，正式配置因此采用预算匹配的 disjoint quota，并审计原始重合量。它与 `content_recency` 的软 content-minus-age 排序形成直接对照；value-guided elite/transition writer 和 online rollout update 仍作为后续 learned operators，不能把该无训练 retrieve 的结果冠为 OnEvoMemory 复现。完整 port 不是硬塞配置：action-conditioned value/key/value encoder、两个 learned writer 与 gated action-query cross-attention 都需在 ordered history 上训练；若官方实现仍未发布，所有阈值、容量和 loss 选择必须明确标为 E-MAC 新设计，并用 `short-only / elite-only / transition-only / all` 以及 correct/shuffled/stale memory 做 matched 消融。

首个两级 store operator 已实现为 `tiered_chunk_mean`：短期原始 token 与长期压缩 token 由同一 store contract 输出，WRITE trace 明确记录 `append_short / migrate_chunk / long-term adjacent consolidation`，合并后仍保留覆盖的 source step 范围和数量。真实轨迹 profiler 已确认它形成不同于 sliding 的压缩长时域表示。它是 MemoAct-style lifecycle 的无训练 lower bound；若 paired rollout 有增量，才把均值替换成 learned causal chunk compressor，并分别消融 temporal retrieval 和 adaptive gate。

第二个时间覆盖 retrieve operator 已实现为 `temporal_multiscale`：它不依赖 query 内容，而以指数间隔优先保留多尺度时刻，并用全局均匀分支严格补满预算；每个返回项都审计实际 frame gap、所属分支和目标 lag。它与 sliding 的 recent-only、`content_recency` 的软 content-minus-age、`semantic_recent_union` 的 semantic+recent 以及 `tiered_chunk_mean` 的压缩 lifecycle 均不同，可在同一 Mem-0 token contract 下做严格零样本筛选。

其配套消融 `uniform_global` 只按完整 causal history 的均匀位置取样，并审计每个 token 的 frame gap 与归一化历史位置。它对应 RoboMME FrameSamp 的 `when-to-sample` 原则，但仍使用一时刻一个 contextual token 和 Mem-0 utilizer。完整 perceptual family 已依据官方 Apache-2.0 源码登记为非执行 typed candidate：16 patch/frame、temporal-spatial position、固定 512-token mask 与 1024-D projector是共享边界；`uniform-frame / prefix-causal RGB-change TokenDrop` 是可替换 retriever，`Context / action cross-attention→AdaLN / separate Expert` 是可替换 utilizer。真实 Put Back profile 已证明两个 selector 的输出不重复；新训练与部署必须共用 prefix heap，released full-episode heap 只用于 checkpoint compatibility。先在 locked RoboMME checkpoint 上复现 2×3 因子后再移植到 RMBench。

首个跨 path 组合 `recent_global` 将 30-slot 预算固定拆为 recent 15 与 global 15；global 分支先排除 recent tail，再对更旧历史均匀采样，因此不会重复消费同一 source step。真实轨迹画像显示其平均 `42.12%` item 位于 latest-30 之外，selected lag 中位数 `15.5`、P90 `169`，兼具连续局部动作上下文和长期覆盖。它是检验 Agent 是否能可靠组装 typed operators 的最小端到端模板，policy utility 仍服从 executor gate。

### Module F：Autonomous Memory Research Harness

借鉴 ENPIRE 与 AgentCanvas/KDLoop，将闭环限定为：

`读论文与历史实验 → 提出 memory 假设 → 选择或修改 program → 单元测试 → 低成本 rollout → 读取阶段 trace → 保留、修改或回滚 → 下一轮`

Harness 提供：

- 只读的 benchmark、seed split、reward 和 evaluator；
- 可写的 memory operator、controller 和配置目录；
- 自动类型检查、CPU 单测、smoke rollout 和 full evaluation；
- 每轮的假设、diff、训练预算、rollout、结果和失败原因；
- 超预算、接口违规、privileged-state leakage 或回归时自动拒绝/回滚。

直接采用 AgentCanvas 已验证的工程约定：候选作为 typed graph/config patch，在昂贵 rollout 前完成 port 类型与 deployability 校验；每个 run 原子写入候选 hash、diff、episode trace 与资源成本；search-time 最优候选必须用独立 seed 重跑确认。同时保留一个明确差异：E-MAC 不允许 Agent 修改 prompt、control flow、model 配置或 evaluator，只搜索 typed memory program。

ALMA 提供最直接的搜索算法基线：archive parent 以 `sigmoid(success − no-memory) − 0.5 log(1 + visit)` 计分，再以温度 `0.5` 的 softmax 做不放回采样，避免只沿当前 best 贪心下钻。当前 harness 已将它实现为独立的 `memory_harness.search_archive`，但候选身份改为 program content SHA-256，重复内容直接拒绝，且在 Gate 1 前不激活。后续 trace reflection 只接收 bounded、配对的成功/失败样本、parent→child delta 与完整资源账本；ALMA 原版的任意 Python、benchmark-specific 描述、可写 bind mount 和未匹配 LLM/DB 成本均不继承。

MemEvolve 补充多目标 survivor 基线：对每个已评测 candidate 同时最大化 success、最小化 cost 与 latency，按非支配层选 parent。当前 `search_archive` 已实现 `ParetoRecord / pareto_ranks / select_pareto_survivors`，以 behavior content hash 破平局并拒绝缺失、非有限或重复记录；同一 selection pool 还必须声明并共用一个 `cost_metric`，禁止把美元、GPU-hours 与 memory tokens 混成可比较数字。没有照抄 release 的可选 `0.60/0.25/0.15` min-max scalarizer，因为它与论文“Pareto rank 内优先 performance”不一致，且缺失 latency 在源码中会被当作理想的 `0`；正式实验将把 ALMA sampling 与 Pareto survival 作为两个独立、同预算 search baseline。MemEvolve 的 trajectory diagnosis、生成和 debug 与 ALMA/KDLoop 合并去重，只额外保留“选中架构冻结后跨任务确认”的 transfer protocol。

EvoMem 增加一个不重复的 Research-Agent writer：不是把所有正向 trial 都写入经验库，而是用 `introduction vs strongest parent / same-parent siblings / inherited descendants / downside rate / independent recurrence` 形成 lineage evidence，再决定是否长期保存该 mechanism lesson。当前已实现为独立 `LineageEvidencePromoter`；输入、输出和拒绝原因均 content-addressed 且可审计，阈值必须显式声明。它不进入 π0.5 observation memory，也不作单机制归因。论文公开附件没有所称 EvoMem runtime，因此 LLM extraction、semantic merge 与 multi-view retrieval 只保留协议，不制造 paper-exact 复现。

EvolveMem 不再复制成第五套搜索器：其有效抽象——typed config action、failure diagnosis、best-so-far guard、停滞探索和 frozen config transfer——都已由上述模块覆盖。Gate 1 后把它实现成**同一 harness 上的 baseline 配置**，而不是运行官方 benchmark-tuned 路径：禁止第 5 轮 terminal-config 注入，禁止 LoCoMo answer-surface/evaluator patch，search、selection 与 confirmation episodes 分离，bundle 按真实 typed diff 计步。官方源码中的 meta new-dimension 只写日志而不创建 operator，因此不能作为“Agent 已自动扩展架构”的实现证据；完整边界见 `artifacts/2026-08-16-evolvemem-source-audit.json`。

为把搜索算法真正接到可插拔架构，而不提前开放任意代码执行，第一层 candidate boundary 已实现为 `memory_harness.search_candidate`：Agent 一次只能提交 `submission + architecture + executor program` 三个 JSON，候选必须完全由已有 typed registry 组装并为 deployable；入口拒绝额外文件、Python、symlink、路径逃逸和未知搜索轴，随后原子冻结 runtime、配置、hypothesis、parent、真实 smoke trace 与 behavior content hash。behavior identity 包含 runtime 和全部执行选项，但排除候选名称与研究文字，因此重命名不能制造伪多样性。只有 Gate 1 通过后，才把这些 preflight artifact 接入 ALMA/random/evolutionary archive；Agent 生成新 Python operator 仍需下一层独立 sandbox、capability manifest 与 code review gate，不能借配置入口绕过。

借鉴 Auto-Robotist，搜索侧另维护可审计的 research-memory record：每条记录绑定不可变 candidate/run hash，保存被证据支持的正向规则、失败诊断、适用任务/跨度和反例，并以 `add / diagnose / merge` 更新。它只帮助 Agent 提出下一轮架构，不进入 policy rollout 的 observation memory，也不能读取 locked-test 单条结果；因此不会把“研究过程记忆”与“机器人执行记忆”混成同一个插件。

其中 `add` 前先经过 EvoMem-style lineage promotion；最终分数高但没有可识别引入点、没有继承后代、downside 过高或支持信号不足的 lesson 不进入长期 research memory。该门只控制经验写入，不改变候选 evaluator 或搜索得分。

资源审计不只统计最终注入的 30 个 slots：runtime 逐步记录每个 path 和整个 program 的实际 store item 数，validated run 汇总 peak storage、写入/跳过次数、retrieve/use token、policy latency 与 memory-observe latency。这样 full-history bank、bounded ring 和压缩 store 即使 action context 同为 30 tokens，也会在 Pareto 成本中被区分。

当前已补齐从 program 到 rollout 的冻结层：`candidate_suite` 自动发现所有 architecture，逐一执行真实 build/smoke，并把 runtime、完整 config tree、component catalog、smoke summary 与哈希封装为一个 immutable artifact。RMBench runner 只接受 checkpoint 自带 snapshot 或一个完整 suite，不允许分别替换 runtime/config；run manifest 与比较器继续绑定 suite/runtime/config/architecture/program 哈希。runner 先依据 checkpoint 的训练 manifest/config snapshot 对整套候选做 CPU shape preflight，policy 构造时再与实际加载的 Mem-0 fusion 二次核对。当前 v6 的 19 个候选均通过真实 Put Back full-memory checkpoint 的 `[31,2048]` preflight；`verified_success_latent` smoke 证明成功 episode 原子提交且失败 episode 不污染 bank。新增的真实 context distinctness audit 在 10 条 validation episode、323 个 sampled queries 上比较最终 `memory_tokens + mask`：14 个可公平比较的 episode-local latent executor 没有完全等价对；唯一 `Jaccard>0.95` 的 `anchor_sliding / sliding` 被明确标为隔离 anchor 的必要嵌套消融，不计作两个独立创新；`kinematic_event` 和 `verified_success_latent` 因缺少 robot-state/outcome protocol 被显式排除。v1–v4 保持不可变，用于复核已绑定的 run。因此 Agent 新增模块后可以先拒绝接口不兼容或行为重复候选，再形成可复现候选集合，而不是依赖手工 PYTHONPATH、修改旧 checkpoint、偷读新任务字段或把 shape error 留到 rollout 中途。

第一阶段禁止 Agent 修改 π0.5 backbone、训练数据、success checker 和 hidden test。否则无法判断提升来自 memory architecture 还是其他改变。

## 7. 实验协议

### 7.1 数据与评测划分

- `development`：允许人工和 Agent 反复查看，用于调试。
- `search/validation`：供搜索过程比较候选，但与 development seed 分开。
- `locked test`：搜索结束后只运行一次；Agent 不可读取单条结果或视频。
- 任务、场景随机化和 retention lag 都要参与划分，不能只换 simulator seed。

### 7.2 多阶段评测

为控制成本，每个候选采用 successive evaluation：

1. 类型检查与单元测试；
2. 3 个 shared-seed screen episodes；
3. fixed ablation 与 screen 正向的探索候选扩到 20 个 shared validation episodes；
4. 只有 20-episode fixed candidates 中仍有正向 success/阶段信号者扩到 50 个 episodes；zero-shot 新结构在 20 episodes 后先做 matched training，不直接借用旧 checkpoint 扩到 50；
5. 最终候选在 locked test 上按 RMBench 协议运行 100 episodes。

所有方法复用相同 seeds 和 policy RNG。successive stages 使用不重叠的 seed shards，组合比较器拒绝重复 `(layout seed, policy seed, layout fingerprint)`；当前实现用 `3 + 17 + 30 = 50`，避免重复 rollout 被当成独立证据。20 episodes 只能形成 pilot，不能作为论文最终显著性结论。

### 7.3 指标

主要指标：

- full-task success rate；
- 相对 no-memory 与 best fixed 的提升；
- 在给定 rollout/GPU/LLM-token 预算下发现的最佳 held-out performance。

次要指标：

- subtask completion、最大 task progress、错误阶段切换；
- latency、memory token、峰值显存和 planner calls；
- M(0)-control regression；
- 搜索过程中 valid candidate rate、best-so-far curve 和最终 Pareto frontier。

### 7.4 搜索基线

在完全相同的 candidate-evaluation budget 下比较：

- 人工固定 modules；
- 人工设计的有限 controller；
- random search；
- 常规 evolutionary search；
- ALMA-compatible visit-penalized open-archive search；
- AgentCanvas 式的 ADAS/AFlow archive search 与 KDLoop trace-aware search；
- 只看最终成功率、不读细粒度 trace 的 Agent；
- 完整 E-MAC research Agent。

这样才能证明贡献来自 Agent 的研究与架构探索能力，而不只是“自动跑了更多实验”。

另报告一个 `task-specific code` 上界/参照：在资源允许且有可审计实现时复现 HyMeS-style symbolic state + progress verification + steering。它允许读取任务 schema，搜索空间和 action utilization 都不同，因此不进入上述严格同预算架构搜索排名；它用于回答“通用 memory architecture 相比逐任务 heuristic engineering 牺牲或获得了什么”。最小复现必须把 `symbolic state / progress verifier / flow steering / rollout-driven code refinement` 分栏计费，并以 frozen program 在 locked conditions 重跑；当前公开材料不足时保持 paper contract，不自行猜 task reward。

## 8. Go / No-Go Gates

### Gate 0：Executor Readiness

- `M(0)-control` 或 oracle-current-cue 条件下，π0.5 能稳定完成基础操作；
- success checker、layout、seed 和 full-episode reset 可复现；
- 结果不能长期停在所有方法 `0/N` 的地板。

未通过：继续修数据、policy 或 evaluator，直到获得可比较的 executor；该门槛只限制对 memory utility 下结论，不会把项目标成 HOLD。

### Gate 1：Fixed Memory Utility

- 20-episode pilot 中至少一个固定模块相对 no-memory 有明确正向趋势；
- 扩到至少 50 shared episodes 后，paired 95% interval 不跨 0；
- oracle memory 有效，shuffle/stale/mismatch 的表现方向更差；
- `M(0)-control` 不出现不可接受的回归。

未通过：固定 memory 复现和 executor 改进继续进行，但不训练 controller，也不启动开放式 Agent architecture search。

Gate 判定由机器可读的 `utility_gate` 固化：`<20` 只标记 screen，`20–49` 只标记 pilot，`>=50` 才允许确认；success 是 primary endpoint，阶段奖励只用于决定是否追加 shared episodes 或进行预算匹配训练，不能单独放行 controller。单个模块通过 success 条件也不等于完整 Gate 1 通过，仍需 oracle、shuffle/stale/mismatch 与 `M(0)-control` 组成诊断 bundle。早期输出统一给出 `ACTIVE` 子状态、证据层级、尚缺 episode 数和下一动作。

### Gate 2：Controller Value

- D1 learned read gate 在 held-out 条件上优于 activation-rate-matched random gate，并相对 `all-on` 提升成功率，或在成功率相当时降低**实际** used token/latency；同时在 `M(0)` 与不需要 memory 的阶段保持低 false activation；
- D2 architecture selector 随后必须优于 best global fixed、per-task static 和 all-on，或在成功率相当时显著降低实际 memory/latency cost；
- 两层优势都不能只来自识别 task ID，且必须在未参与 label calibration/controller selection 的 seeds 与 task conditions 上复现；
- oracle-error gate 若无收益，则停止 D1；D1 若无收益，则不启动 D2，更不启动开放式 architecture evolution。

未通过：保留固定模块结论，停止开放式 architecture evolution。

### Gate 3：Autonomous Search Value

- 在相同候选数、rollout、GPU-hours 和 LLM token 预算下，Agent 搜索的 best-so-far curve 或最终 Pareto frontier 优于 random/evolutionary search；
- 最终 program 在 locked tasks、未见 lag 或状态变化上仍优于有限 controller；
- 新 program 的提升能在独立重跑中复现。

## 9. Research Phases

### Phase 0：基线与任务扩展

跑稳 π0.5 no-memory；把当前 Cover Blocks-only runner 扩展到最小的 `M(1)+M(n)+M(0)-control` 任务集。

### Phase 1：固定 memory modules

在同一 π0.5 上完成 none、anchor、sliding、anchor+sliding、key/subtask，并建立优势图谱和内容干预结果。

### Phase 2：有限 controller

先证明在现有模块间动态选择有价值，避免直接让 Agent 在无效信号上开放搜索。

### Phase 3：受限 program search

Agent 只能组合和调节 catalog 中的 typed operators；验证闭环、预算和回滚机制。

### Phase 4：开放 operator evolution

允许 Agent 在 sandbox 中修改 operator 实现或添加新 operator，但 π0.5 backbone、数据和 evaluator 继续冻结。

### Phase 5：泛化与有限真机验证

在未见 RMBench task condition、lag 和状态变化上复测；只把通过 simulation、安全和预算检查的少量候选部署到真实机器人。

## 10. 预期贡献

1. **Autonomous Embodied Memory Architecture Search setting**：定义研究 Agent 如何在具身任务中自主探索历史信息的写入、表示、检索、使用和遗忘方式。
2. **Typed and budgeted E-MAC framework**：提供兼容 π0.5 的 memory program space 与可审计的 autoresearch harness。
3. **RMBench memory advantage map**：系统展示不同 fixed memory 在复杂度、lag、状态变化和观测充分性下的优势与干扰区间。
4. **Autonomous-search evidence**：在预算匹配下验证 Agent 是否真正超过固定设计、有限 controller 和常规搜索，并分析其发现的架构能否迁移。

## 11. 主要风险与降级方案

- **π0.5 地板效应**：优先通过 M(0)-control、oracle cue 和 primitive-level success 区分 executor 问题；未通过 Gate 0 就停止 memory 实验。
- **搜索成本过高**：先配置搜索、后代码搜索；用 smoke → 10 → 50 → 100 episodes 的多保真筛选。
- **Agent 只会调超参数**：最终贡献要求至少出现 operator composition、lifecycle 或 utilization 结构变化，并报告与纯参数搜索的差异。
- **Agent overfit validation**：使用 locked test、任务条件级切分和一次性最终评测。
- **与 AgentCanvas/HyMeS 的贡献重叠**：不声称首个 embodied architecture search 或 code-space embodied memory；核心证据必须收紧为“同一 VLA 上的 typed memory-only search”、跨任务结构迁移、memory-specific advantage map 与 cost Pareto，并与 KDLoop 等 trace-aware search 同预算对比。若最终只能生成 task-specific stage rules、改 prompt/control flow，或无法超过已有 AAS 搜索策略，则不能支撑原定贡献。
- **不同模块预算不公平**：同步报告成功率和 cost Pareto frontier，不允许隐藏 planner 调用或额外训练预算。
- **proposal 范围过大**：最小可发表版本止于 RMBench simulation + 一个有限 controller + 受限 typed search；开放代码生成和真实机器人均可作为扩展。

## 12. 最终判断

这个 proposal 的核心逻辑是成立的，而且与现有代码基础匹配。当前不应把注意力转向新的因果方法，也不应立刻搭建一个可以任意改代码的 Agent。最稳妥且最能支撑最终论文的顺序是：

`稳定 π0.5 → 同 backbone 固定 memory → 优势图谱与内容验证 → 有限 controller → Agent 自主架构探索`。

项目真正的成败点有两个：第一，能否在至少一个 RMBench 任务上获得可重复的 fixed-memory utility；第二，能否证明 Agent 的搜索过程在相同预算下超过普通搜索，而不只是重新发现 anchor/sliding/key。
