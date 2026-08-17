# E-MAC 仓库基线审计（2026-08-14）

## 结论

当前仓库已有可用的 π0.5、RMBench、paired-layout 和固定 policy RNG 基础设施，但**尚没有完成并冻结的 E-MAC clean no-memory pilot**。本周应从单 checkpoint 的 clean runner 开始，不能把 R23 的 `without_episodic_memory` 当作 no-memory。

开发集固定为 `configs/cover_blocks_dev_seeds.json` 中的 `100000..100009`；这些 seed 已被反复用于开发，明确不是未来 locked test。

## 现有系统边界

| 路径 | 实际组成 | 已能回答 | 不能回答 |
|---|---|---|---|
| `run_rmbench_baseline_local.sh` 默认参数 | 单 π0.5；memory/router/oracle 默认关闭 | 指定正确 RMBench checkpoint 后，可作为 clean baseline 基础入口 | 默认 checkpoint 是通用 `pi05_libero`；未显式开启 paired-layout；尚无 E-MAC memory program trace |
| `scripts/run_rmbench_cover_blocks.py` 的 R23 `full` | 六阶段 checkpoint router + working memory + episodic memory | 当前完整 R23 系统在固定开发 seed 上的表现 | 单一 π0.5 的 memory 增益 |
| R23 `without_episodic_memory` | 仍保留 stage router 与 working memory | episodic 部分的系统消融 | clean no-memory；不能作为 `none` program |
| Mem-0 reproduced | 官方 executor + reproduced planner | pipeline sanity reference；最新结果 `2/10` | 与 π0.5 memory module 的公平横向结论 |
| snapshot diagnostic | 从中间状态启动的局部干预 | prompt/memory 对局部动作的诊断 | 从 reset 开始的 full-episode success |

## 已核实结果

- R23 corrected full：`0/10`，来源 `rmbench_runs/weekly_20260806_cover_blocks_main_corrected_v2/ours/full/summary.json`。
- Mem-0 reproduced：`2/10`，来源 `rmbench_runs/weekly_20260804_mem0_rgb_balanced_v3_period_fixed_paper_l8_10seed_v2/full/final_summary.json`。
- snapshot run 每个 plan 为 36 个局部 rollout；它不是 full-episode benchmark，不能与上述成功率混合。
- 历史 clean baseline checkpoint `rmbench_checkpoints/pi05_aloha_pen_uncap/rmbench_cover_blocks_oracle_50plus_pi05_base_20260613T062923Z/9999` 存在，但后续 prompt-contract 审计确认其缺少训练 provenance，且关联配置丢弃逐帧 subtask prompt；它仅保留作历史诊断，不进入修正后的 π0.5 port lineage。

## Clean no-memory 的硬约束

正式 `none` 条件必须同时满足：

1. 单一 checkpoint，无 `policy_router_manifest`；
2. `episodic_memory=false`、`working_memory=false`、`memory_injection=none`；
3. 无 phase-aware subtask prompt、prompt schedule 或 simulator pointer；
4. `task_state_trace_frequency=0`，避免 privileged state 进入 policy-side 数据通路；
5. paired layout 开启，simulator seed 与 policy seed 分离；
6. observation 与 RNG 相同时，`none` wrapper 向 π0.5 传入同一个 observation 对象并保持相同 RNG 演化。

上述第 6 项已有 CPU contract test；真实 checkpoint 的 action parity 仍需 GPU smoke 后冻结。

## 固定 memory 接口的可用程度

仓库内 OpenPI 已有 `memory_tokens` / `memory_mask` 和 cross-attention 路径，可复用为统一注入通道。新 `memory_harness` 已把 `encode / write / store / retrieve / utilize / lifecycle` 分开，并以同一 registry 构建 `none / anchor / sliding / anchor+sliding / key`。

当前 `key` 是 phase-change 写入的**接口骨架**。Deployable 版本不能读取 RMBench 的 `current_state_pointer`；只有接入视觉/语言 phase estimator 后才可作为有效 key baseline。Oracle phase 只能另建 `deployable=false` diagnostic，不能覆盖此配置。

## 本周实验判定

- 先完成 clean none 的 3-episode run 与同 seed 重跑 parity。
- 再用 memory-enabled context-adapter checkpoint 做固定 module smoke；`none` 与所有 memory 条件必须使用该组实验中的同一 checkpoint。
- 如果 oracle/M(0) capability check 仍无基本阶段进展，结论为 executor bottleneck，停止扩大 memory search。
- 现有 `0/10` 和 `2/10` 仅作为起点，不构成 memory utility gate 已通过的证据。
