# E-MAC Fixed Memory 实现与 Smoke 记录

日期：2026-08-14

## 当前结论

统一 memory program 已完成第一版端到端接线。`none / anchor / sliding / anchor+sliding / key` 均可由同一 runner、同一 π0.5 context-adapter checkpoint 启动；非 `none` 条件使用 OpenPI 已有的 query encoder、`memory_tokens/memory_mask` 和 cross-attention，不修改 π0.5 backbone。

本文件记录的是 2-step 结构 smoke。3-episode shared-adapter utility pilot 见 `2026-08-14-dev3-pilot.md`；该旧分支已停止，并被后续 Mem-0 port 取代。controller 仍为 **deferred until Gate 1**。

协议复核发现，本文件中的早期 smoke 继承了 `adapt_to_pi=true`，而 checkpoint 按 `no-adapt-to-pi` 训练。因此这些记录只保留为接口接线证据，不能作为性能证据。正式 runner 已强制 `POLICY_ADAPT_TO_PI=0`，validator 也会拒绝不一致结果。

## 自动化测试

- `memory-harness/tests`：24 passed。
- 仓库既有 memory/evaluator/runner 回归：72 passed。
- 合计：96 passed。
- 五个 CPU program trace 均包含可审计的 `RESET / SELECT / RETRIEVE / USE / WRITE` 子集。

覆盖点包括：strict schema、none observation identity、RNG 透传、anchor reset、sliding eviction、anchor+sliding 共用 8-token budget、phase-change key、privileged metadata 拒绝、JSONL audit 和 run contamination validation。

## Clean π0.5 smoke

路径：

- `rmbench_runs/emac_clean_none_smoke_20260814`
- `rmbench_runs/emac_clean_none_smoke_repeat_20260814`

两次运行均使用 simulator seed `100000`、policy seed `120000`、相同 prompt 与相同 layout fingerprint `[1, 2, 0]`，且均通过 clean manifest 检查。两步 action 的最大绝对差为 `2.3115e-4`，不是 bitwise identical；这说明 seed/protocol 已稳定，但当前 GPU renderer/JAX 路径仍有小量数值非确定性。正式 parity 应在保存并复用完全相同 observation tensor 的离线测试中做 bitwise/容差验证，不能把 simulator 重跑直接当作同输入。

## 五个真实 π0.5 smoke

统一 checkpoint：

`rmbench_checkpoints/pi05_aloha_pen_uncap_context_adapter/cmci_aligned_adapter_v3_101_20260725T101000Z/100`

| Program | Run | 2-step error | 最大实际 memory tokens | 结构行为 |
|---|---|---:|---:|---|
| none | `emac_fixed_none_smoke_20260814` | 0 | 0 | 两步均 USE 0 |
| anchor | `emac_fixed_anchor_smoke_20260814` | 0 | 4 | step 0 WRITE；step 1 RETRIEVE/USE anchor |
| sliding | `emac_fixed_sliding_smoke_20260814` | 0 | 2 | step 1 使用上一 moment，并继续写入 ring |
| anchor+sliding | `emac_fixed_anchor_sliding_smoke_20260814` | 0 | 4 | anchor 与 sliding 独立存储，共享总预算 |
| key | `emac_fixed_key_smoke_20260814` | 0 | 4 | phase 为空时退化为单 key；仅证明接口可运行 |

这些 GPU smoke 发生在 fixed-all controller 字段加入 schema 前；fixed-all 与当时“所有 path 总是激活”的 runtime 语义相同。它们是接线证据，不作为正式结果。runner 现已改为在启动前把 program 固化到 run 目录，后续正式实验不会再依赖可变的全局配置文件。

## 已知限制

1. 当前节点未安装 cuRobo 和 pytorch3d，RMBench 日志显示 executor backend 为 `mplib_RRT`。因此本轮 smoke 不能与使用官方 cuRobo executor 的 Mem-0 `2/10` 作性能比较。
2. Deployable key 还没有视觉/语言 phase estimator，不能使用 RMBench `current_state_pointer`。当前仅是 operator 骨架。
3. 本轮 latency 含首次 JAX 编译（none 约 10.4s mean，memory 条件约 17–18s mean），不能视为稳态成本；正式 pilot 需单独报告 warm-up 后 p50/p95。
4. 尚未跑 oracle phase、M(0)-control、3-episode fixed pilot，也未触发 remove/shuffle/stale 内容检查。
5. 两条 full-run 计时诊断分别在 chunk=1 的 step 204 和 chunk=10 的 step 161 主动终止，目录名均带 `interrupted`，没有 summary，也不计入任何结果。正式 runner 已统一冻结为 `3 episodes × 1500 steps × chunk 10`。

## 下一执行门

先补完全相同 observation tensor 的 none action parity，然后在明确 executor backend 的前提下跑 3 个 development episodes 的 none 与 oracle capability check。只有 oracle 显示 executor 有阶段推进能力，才扩展五个 fixed program；只有 fixed program 出现一致正趋势，才运行内容干预并准备 controller 数据。
