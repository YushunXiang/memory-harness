# Put Back Block Gate Execution Plan

日期：2026-08-15

## 目标

在同一 π0.5 数据、seed protocol 和预算口径下，区分以下三类问题：

1. full-memory checkpoint 是否学到可执行 motor skill；
2. anchor/sliding 内容是否相对 empty mask 带来 full-episode utility；
3. 新 write/store/retrieve operator 是否只是改变 trace，还是能超过其最接近的固定基线。

## 训练链

1. `anchor_sliding_u200_b2a28_v3`：从 none-u200 初始化，再训练 1000 optimizer updates；总 exposure 1200 updates，target checkpoint `27999`。
2. `none_u1200_b2a28`：Mem-0 参数结构存在但 memory mask 始终为空，独立训练 1200 updates。
3. `native_none_u1200_b2a28`：不包含 Mem-0 memory modules 的 π0.5 control，独立训练 1200 updates。

第 2/3 项用于区分 memory 内容、额外参数结构和训练 exposure，不能与第 1 项的同-checkpoint mask ablation 混称同一种证据。

### 中断与恢复审计

full-memory 先完成 `0→5599`，随后从 `5599` 续训；第二次进程在 raw step `19152` 后无 Python/JAX 异常尾迹地终止，最近完整 Orbax checkpoint 为 `16800`，因此 `16801..19152` 的未提交计算不计入最终训练链。第三次进程从 `16800` 恢复完整 optimizer、EMA 与 data-loader state，继续写入同一 experiment，目标仍是 `27999`。训练入口支持显式 `RESUME=1`：只允许已有 experiment，并向 OpenPI 传递 `--resume`；默认 `RESUME=0` 拒绝覆盖已有日志。finalizer 现在绑定全部三段日志，要求日志恰好包含“一次初始运行 + 每个 same-run restore 对应的一次运行”，并验证每次 restore 已完成。仍保留的 `16800` 绑定 `_CHECKPOINT_METADATA` 哈希；已被 Orbax retention 删除的 `5599` 则必须在更早日志中同时具备 finalized-save 和训练 finalizer 完成证据。验证后，全部日志按顺序复制进最终 checkpoint 的 `training_logs/` 并再次校验哈希，避免 `/tmp` 生命周期破坏 provenance。后续阶段等待最终 `_CHECKPOINT_METADATA` commit artifact，而不再以临时 PID 消失作为完成信号。

### Program 配置迁移审计

训练 context manifest 绑定了生成时的 program config SHA-256。后续配置仅增加显式 `anchor_path/history_paths` 角色字段，但这仍会改变哈希；不能仅凭人工判断继续评测。当前保留原始哈希对应的冻结配置，并用 `memory_harness.audit_program_migration` 对旧、新配置回放 anchor 缺失/存在及 history 长度 `0/1/29/30/31/35`，要求 token、mask 和角色分配逐项完全相同。full-memory 与 empty-memory 的审计产物分别为 `anchor_sliding_program_migration_audit.json` 和 `none_program_migration_audit.json`。训练 finalizer 在哈希不一致时强制要求与 context manifest SHA 绑定且 `ready_for_context_reuse=true`、全部 replay checks 为真的审计文件；缺失或包含其他配置变化时直接失败。

组合接口随后进一步改为显式 `history_path_quotas`。旧 API 已从 live runtime 移除；已冻结 checkpoint 继续使用自身 runtime snapshot。现有 Put Back Block bank 到 quota config 的新审计为 `anchor_sliding_quota_program_migration_audit.json` 与 `none_quota_program_migration_audit.json`，两者的 source hash、角色、完整 30-slot budget 和逐 token layout replay 均通过。正在运行的 u1200 control 仍绑定训练启动时的冻结 runtime/config/audit，不受 live schema 修改影响。

### Runtime 冻结

full-memory finalizer 把实际 `memory_harness` Python sources 和整棵实验 JSON 配置冻结到 checkpoint，逐文件校验 SHA-256 并生成聚合 identity。最终训练前的独立 preflight 覆盖 31 个 Python files（runtime SHA-256 `f1f80816abf5f564872eaf14a2cdc24f02a5e0a0e46276fa1d0b2fd9280edebe`）和 27 个 JSON files（config SHA-256 `30e3194657f8523834c9eee748ba9f3c65bd253d130c558ea520600689724129`）；finalizer 与每个 run manifest 必须复核相应 identity。所有 Gate-3/20/50 fixed runs 从 checkpoint snapshot 加载代码、task、architecture 和 executor 配置，并把完整 config snapshot 及实际使用的子集复制进 run artifact。empty-memory 训练分支显式复用 full-memory 的 runtime 与 config snapshots；比较器拒绝 memory runtime、完整 config snapshot、task config 或 successive shards 的 architecture/executor config identity 漂移。native π0.5 不执行 memory runtime，因此只在 native-vs-memory 方法级比较中标记该检查不适用。

## 同 checkpoint screen

首个 shard 使用 layout seeds `100000..100002`、policy seeds `120000..120002`：

- `none → anchor / sliding / anchor_sliding`：`fixed_ablation`；
- `sliding → novelty_sliding / dhem_event / content_recency / semantic_recent_union`：`zero_shot`；
- `anchor` 与 `anchor_sliding → kinematic_event`：`zero_shot`。

Novelty 必须以 sliding 为 reference；`none → novelty` 只能说明“任意 memory vs empty”，不能隔离 write gate。

### Final checkpoint offline diagnostic

`27999` checkpoint 完成后，在 10 个 validation episodes 的 40 个分层样本上比较 matched、empty、mismatched、移除/替换 anchor 或 sliding 以及 sliding shuffle。matched loss 为 `0.127220`，empty 为 `0.128973`，mean `matched-empty=-0.001753`，但 episode-cluster bootstrap 95% CI 为 `[-0.004721, 0.002069]`；matched 与 mismatched 的差为 `+0.000037`，CI 同样跨 0。因此离线 action loss 尚未显示稳定的 memory-content signal。该结果只作为诊断，不能替代正在运行的 RMBench paired success；也不能用来提前停止 fixed Gate-3/20。

### Gate-3 运行进度

- `none`：`0/3` full success，三个 episode 的 `max_reward/total_reward` 均为 `0/0`。
- `anchor`：`0/3` full success，三个 episode 的 `max_reward/total_reward` 均为 `0/0`。审计记录了每个 episode 一次 anchor 写入、共 `1500` 次 retrieve 和 `1497` 次非空 token 使用，说明该结果不是 memory path 未执行；首个 rollout 的动作轨迹与 none 高度相似，当前仍处在 executor skill floor 或 weak-utilization 两种解释之间。
- `sliding`：`0/3` full success，三个 episode 的 `max_reward/total_reward` 均为 `0/0`；每步写入和最多 30-token retrieval 均已执行。
- `anchor_sliding`：`0/3` full success，三个 episode 的 `max_reward/total_reward` 均为 `0/0`。与 none 的首个共享 rollout 相比，前 10 个 cached actions 完全相同；memory 可用后的逐步 action L2 已明显分离，且最多使用完整 31-token context，因此不是 no-op 接线，但变化没有转化为第一次抓取。

四个 fixed 条件在 3 个共享 seed 上均为 `0/3` complete success，pairwise utility gate 最初返回 `no_detectable_direction`。按分层开发约束，已停止尚未完成的 novelty/DHEM/kinematic/content-retrieval zero-shot screen；这些候选在基础 fixed module 尚无 rollout utility 时不可解释，也不应抢占预算匹配 control。随后完成的 `empty-mask u1200` 与 native no-memory u1200 也均为 `0/3` complete success、`max_reward=0`。三路 control 每路总计 `1200` updates、`67,200` optimizer examples，并回溯同一 π0.5 base。源码审计确认 Put Back 没有 dense stage reward，因此 `max_reward=0` 不能证明没把 block 移到中心；旧 utility/readiness 只保留为 complete-success 证据。fixed Gate-20 仍暂不放行，进入预注册 u3000 学习曲线并用新 task progress 重新判断。

三组 rollout 完成后由 `memory_harness.decide_executor_readiness` 自动读取 `empty↔full / native↔full / native↔empty` 三份预算匹配 comparison，交叉核对每个 shared episode 的绝对指标，并产生机器可读的唯一下一步。v2 contract 对 Put Back 使用 `task_progress_score`，对原生有 dense reward 的任务使用 `max_reward/total_reward`；full-memory 出现 success 或非零 screening progress 才允许 Gate-20，仅 control 出现信号则先重训 full-memory，三组均无可观测信号才扩大联合训练预算。三种情况都不会直接放行 controller。

下一预算分支在结果出现前预注册为总计 3000 optimizer updates（effective batch 56）：现有三条 1200-update chain 各通过一个独立 `+1800` stage 延长，不改写旧 checkpoint。若仅 control 有信号，先只延长 full-memory 并做 3-seed executor check；出现信号后再把 empty/native 补齐到 3000，随后才允许预算匹配比较。若三条均全零，则按 native→full→empty 顺序全部补齐到 3000，再重复三 control gate。若当前 full 已有信号，则不增加训练，直接把四个 fixed conditions 扩到共享 20 seeds。`+1800` full/empty 必须沿用父 checkpoint 内冻结的 runtime/config 与 quota migration audits；full/empty 父快照已经验证为相同的 runtime SHA `f1f80816abf5f564872eaf14a2cdc24f02a5e0a0e46276fa1d0b2fd9280edebe`、config SHA `30e3194657f8523834c9eee748ba9f3c65bd253d130c558ea520600689724129`。3000 仍只是下一学习曲线点，不声称达到论文 30k×448 的训练规模。

上述分支已实现为 `scripts/continue_put_back_executor_gate.sh`，control-only 分支在 full u3000 Gate-3 后用 `assess_run_signal` 生成独立 artifact：full 仍无 success/progress 才停止；出现信号后再把 native/empty 补到 u3000，随后重新做三组预算匹配 comparison。u1200 readiness 已在 2026-08-16 生成；它仍是有效的 `0/3` complete-success 证据，但其“all-floor”解释被 `artifacts/2026-08-16-put-back-sparse-reward-progress-audit.json` 取代。watcher 已调用 continuation；native `+1800` stage 已完成并通过 manifest 验证，当前 full-memory `+1800` 正常训练，之后按 `full Gate-3 → empty → empty/native Gate-3` 自动续接。未来 rollout 固定每 10 步记录 task state，并在每个 episode 原子汇总最大 progress。旧 u1200 三路另由 `memory_harness.replay_put_back_progress` 校验并回放已记录 action，在不加载 policy 的情况下恢复相同 progress；任何 action 缺失或 replay/source success 不一致都会使该路证据失败。脚本对 decision schema、checkpoint、父 runtime/config、输出 checkpoint manifest 和 paired comparison 都 fail-closed，不会启动 controller。

这里的 screen 是 Mem-0 action-side module 向 π0.5 的迁移实验，不是官方 Mem-0 executor 的重复评测。仓库中已有的官方 executor + reproduced planner 原生 10-seed pilot 为 `2/10`；论文的 `68/100` 是不同 backbone、30,000-iteration/global-batch-448 训练和完整 key planner 条件，不能作为当前低预算 π0.5 port 的预期下限。

## Successive evidence

- screen：已有 3 pairs；
- pilot extension：新增 17 pairs，layout seed 从 `100003`、policy seed 从 `120003` 开始；
- confirmation extension：新增 30 pairs，layout seed 从 `100020`、policy seed 从 `120020` 开始。

组合后恰为 `3 + 17 + 30 = 50` 个不重复 pairs。比较器以 `(seed, policy_seed, layout_fingerprint)` 为 evidence identity 并拒绝重复。fixed ablation 无论 3-pair screen 是否有方向都进入 20-pair pilot；zero-shot 只有 success 正向或 max/total reward 同时正向才进入 pilot。20-pair zero-shot 正向后先做 program-matched training，不能继续借旧 checkpoint 宣称结构有效。

## Gate 1 仍需补齐

即使 50-pair fixed ablation 的 success interval 为正，也只满足 candidate utility requirement。完整 Gate 1 还需：

- oracle content；
- shuffled / stale / mismatched content；
- RMBench `M(0)-control` regression；
- executor readiness 与 success checker audit。

只有完整 bundle 通过后才训练有限 controller。
