# Mem-0 三种 Memory 复现矩阵

日期：2026-08-15

项目状态：**ACTIVE**。这里把论文结论、released Mem-0 复跑和 π0.5 移植分开，任何一栏的失败都不能覆盖另外两栏。

| Memory | 论文 Table 2 | 当前最接近原版的本地证据 | π0.5 移植证据 | 下一动作 |
|---|---|---|---|---|
| anchor | M(1) 平均 `52.8→26.8`（移除后）；Put Back `90→35` | released `m1_mix` Put Back full `10/10`，mask-anchor `2/10`；8 个 full-only、2 个共同成功，绝对下降 `80 pp` | 同 checkpoint Gate-3 为 `0/3`；动作改变但 executor 位于零能力地板 | released dependency 已验证；等待非零 π0.5 executor 后做 matched-training 移植 |
| sliding | M(1) 平均 `52.8→40.4`；Put Back `90→78` | 同一 released full `10/10`，mask-sliding `0/10`；10 个全部为 full-only success，绝对下降 `100 pp` | 同 checkpoint Gate-3 为 `0/3`；动作改变但 executor 位于零能力地板 | released dependency 已验证；移植后仍需 matched training，并保留 Swap T/Cover Blocks 负干扰反例 |
| key | M(n) 平均 `28.5→4.8`；Cover Blocks `68→5` | released Cover Blocks executor + matched local planner：key `2/10`、no-key `0/10`；两个成功 seed 均为 key-only，planner exactness `48/48` 对 `25/46` | 旧 π0.5 4-seed 两边 `0/4`，但训练丢失 subtask prompt，已降级为历史诊断 | 用修正的 prompt contract 重建 π0.5 executor/context |

## 证据边界

1. 论文 Table 2 使用每任务 50 demonstrations、100 rollouts；Execution Module 的论文训练预算为 30k iterations、global batch 448。
2. 2026-07 released `m1_mix` 是五个 M(1) 任务联合训练 50k steps 的新 checkpoint；其 Put Back model-card 结果为 `100/100`，不是论文单任务 `90/100` 的同一训练产物。
3. `mask-anchor / mask-sliding` 在 released full checkpoint 上只替换对应 memory 输出，保留 write/lifecycle；它检验训练后模型是否依赖 memory，不是论文 Table 2 的精确重放。论文正文与当前 release 均未说明或提供 anchor/sliding 消融的独立训练产物，因此不对其是否分别重训作无依据推断。
4. 发布方只公开 M(n) executor，没有 key/no-key planner weights。当前 key/no-key planners 使用相同 50 demos、相同 75 optimizer steps；key/no-key 训练格式 exact-match 为 `30/30` 与 `26/30`。因此只能标为 **matched local reproduction**。
5. π0.5 port 使用不同 backbone、输入和远小于论文的训练预算；在 native/empty/full control 建立可比较的 complete success 或 task progress 前，`0/3` 不能解释成 memory 无效。Put Back 的 benchmark reward 为 final-only，必须读取显式 `stage_id` progress，不能把 `max_reward=0` 误称为“零阶段 reward”。
6. M1 evaluator 逐 episode 记录实际通过 expert-feasibility check 的 simulator seed 与 Success/Fail；summary 仅在三条件 seed 序列、逐 episode 日志和 aggregate success 三者互相一致时生成，并输出 `full-only / ablation-only / both` paired transitions。RMBench `setup_demo` 同时用该 seed 重置 NumPy/Torch RNG，因此场景与 diffusion noise 都配对。
7. M(n) no-key summary 使用同一逐 episode contract，并同时核对既有 key reference 的 simulator seeds、独立 policy seeds、successful seeds 与 aggregate；paired transitions 直接报告 key-only、no-key-only、共同成功和共同失败。
8. 旧 Cover Blocks π0.5 memory checkpoint 的训练配置未启用 `prompt_from_task`，且 repack 丢弃 `prompt`，实际使用 global instruction；部署 planner 却输出 subtask instruction。旧 `1399` memory checkpoint 和缺少训练 provenance、使用同类 global-prompt 配置的 `9999` baseline 已从默认 task spec 移除，不能作为忠实 key port lineage。task schema 不再接受 checkpoint 字段，rollout 必须显式绑定经 finalizer 审计的新 checkpoint；修正后的 native/memory 配置和 context generator 均强制逐帧 subtask prompt。
9. 修正版 Cover Blocks template 从 79 条可用 episode 以冻结 seed 选择 50 条，再按逐帧 `task_index` 生成 300 个连续 subtask segment；每条 episode 恰好 6 段，六种真实 prompt 各出现 50 次。manifest 位于 `rmbench_runs/emac_cover_blocks_v2_subtask_prompt/task_template.json`，SHA-256 为 `7e4a611764bc3e2323a54858c28b68430a84fb40e42c942e86537b7e2ebfcb74`。

## 自动执行顺序

`empty π0.5 u1200 → released M1 full/mask-anchor/mask-sliding → released M(n) executor + no-key planner → native π0.5 u1200 → readiness decision`

结果入口：

- M1 固化结果：`artifacts/2026-08-15-mem0-m1mix-put-back-intervention-summary.json`；完整 videos/logs 保留在 `/tmp/mem0-m1mix-put-back-official-gate10-20260815/`
- M(n) 固化结果：`artifacts/2026-08-15-mem0-cover-blocks-key-no-key-summary.json`；完整 videos/logs 保留在 `/tmp/mem0-cover-blocks-no-key-gate10-20260815/`
- π0.5 readiness：`/tmp/put_back_block_executor_readiness_u1200_gate3.json`

当前 released M1 三条件已于 2026-08-15 完成：`full=10/10`、`mask-anchor=2/10`、`mask-sliding=0/10`。三组 aggregate marker、逐 episode 官方日志和 seeds `100000–100009` 完全一致，checkpoint SHA-256 为 `752937933440d3825a5463ea4c3a68e28094558b0d0a301982feef6b648a64f6`，运行中无 runtime error。配对转移分别为 anchor `8 full-only + 2 both-success`、sliding `10 full-only`。这是 released full checkpoint 的强执行时依赖证据，不是论文 Table 2 的独立训练消融重放。

当前 released M(n) executor 的 matched local planner 对照也已于 2026-08-15 完成：key `2/10`、no-key `0/10`；key 的成功 seeds `100000/100007` 在 no-key 下均失败，其余 8 个 seeds 共同失败。planner 输出 diagnostic exactness 为 key `48/48 (100%)`、no-key `25/46 (54.35%)`，差值 `45.65 pp`。两侧 simulator seeds、policy seeds、executor 和训练 demonstration 数一致；no-key summary SHA-256 为 `6f8a8d451b58dd797711d901aa1d80a8228ae14b08fd17468cdfc17f7997346c`。由于发布方未提供 planner ablation weights，它是 matched local reproduction，而不是 Table 2 的官方权重 replay。
