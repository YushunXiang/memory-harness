# Mem-0 Memory Injection Audit

日期：2026-08-14

## 结论

Mem-0 会直接保存未经额外压缩的视觉 latent，但不会把它们接到一个未经过 memory 训练的既有策略上。其 anchor/sliding memory 与 execution module 的 cross-attention 一起训练；key memory 则由单独微调的 planner 使用。因此，“直接保存并注入 latent”是论文做法，“推理时给普通 π0.5 临时外挂一个未训练 memory 分支”不是论文做法。

当前 E-MAC fixed pilot 使用的 context-adapter checkpoint 不能用于判断 anchor/sliding utility：checkpoint 不含后来新增的 `memory_attn_1/residual_scale`，加载器将它补为零，而 action-expert memory readout 又乘以 `tanh(residual_scale)`，所以 memory 对 action branch 是严格恒等关闭的。

## Mem-0 原始机制

- Key memory：planner 输入 episode 初始图像、全局任务，以及所有已完成 subtask 的文本与结束 RGB 图像；planner 本身使用 key-memory 数据做 LoRA fine-tune。
- Anchor memory：保存当前 subtask 起点的 mean-pooled image latent，在该 subtask 内保持不变。
- Sliding memory：保存最近 `K` 个 mean-pooled image latents。
- Utilization：当前 image latent 分别对 anchor/sliding 做带残差的 cross-attention，两个融合结果再与 text latent 拼接，作为 DiT action policy 的条件。
- Training：官方 execution module 使用 Qwen3-VL-2B + DiT，并按任务训练；官方说明为每任务 30K iterations。它不是 π0.5 的 inference-only wrapper。

论文：https://arxiv.org/html/2603.01229v3

官方实现说明：https://github.com/RoboTwin-Platform/RMBench/blob/main/policy/Mem-0/README.md

## 本地证据

使用 checkpoint：

`rmbench_checkpoints/pi05_aloha_pen_uncap_context_adapter/cmci_aligned_adapter_v3_101_20260725T101000Z/100`

1. checkpoint metadata 时间为 `2026-07-25 10:01:28 UTC`；包含 `memory_gate`、`memory_in_proj` 和 `memory_attn_1`，但不包含 `residual_scale`。
2. 当前 `gemma.py` 的零初始化 `residual_scale` 修改时间为 `2026-07-25 10:03:59 UTC`，晚于该 checkpoint。
3. `BaseModelConfig.load()` 会从零初始化 reference model 补齐缺失的 `memory_*` 参数。
4. 当前 action-expert memory gate 为：

   `gate * tanh(residual_scale) * memory_residual_scale`

   因此补齐后的 `residual_scale=0` 使 memory readout 恒为零。

## 复核实验

命令使用 GPU 1、固定 checkpoint 和 held-out Cover Blocks observations，比较：

- no context；
- matched memory；
- mismatched memory；
- empty memory。

新跑的 8-sample 结果：

| Condition | Mean loss |
|---|---:|
| no context | 0.04595798094233032 |
| matched | 0.04595798094233032 |
| mismatched | 0.04595798094233032 |
| empty | 0.04595798094233032 |

所有 paired difference 均为 `0.0`。输出位于：

`memory-harness/artifacts/2026-08-14-hard-injection-check/offline_eval.json`

已有的 64-sample 检查也得到四个条件完全相同：

`rmbench_runs/cmci_20260727_vla_vlm_candidate_audit/aligned_adapter_step100_full.json`

为了排除“旧 adapter 已学会利用 memory，只是被新增零门关闭”的可能，又在不修改 checkpoint 的前提下，仅在内存中将 `residual_scale` 设为 `3`，即 `tanh(3)≈0.995`。8-sample 结果为：

| Condition | Mean loss | 相对 no context |
|---|---:|---:|
| no context | 0.04559499041148229 | — |
| empty | 0.04559499041148229 | 0.0000 |
| matched | 2.0190874175491627 | +1.9735 |
| mismatched | 2.0177440356349687 | +1.9721 |

matched 与 mismatched 都使 loss 约变为 no-context 的 44 倍，且 matched 并不优于 mismatched。这说明强制打开旧 adapter 会产生巨大无语义扰动，不能作为有效 hard-injection baseline。输出位于：

`memory-harness/artifacts/2026-08-14-hard-injection-check/offline_eval_forced_scale3.json`

## 对 E-MAC 的直接影响

1. 保留现有 π0.5 Cover Blocks fine-tune 作为共享初始化与 no-memory baseline。
2. Key/subtask memory 先复用已经跑通的 Mem-0 planner → subtask prompt → π0.5 executor 路径；该路径不依赖 latent adapter。
3. Anchor/sliding 采用 Mem-0 的最小表示：直接保存当前 π0.5 vision latent，不额外训练 encoder。
4. 但 utilization 至少要训练 memory projection、gate 和 action-expert cross-attention。第一版冻结 π0.5 主干，只训练这些 adapter，并固定数据、更新数和 token budget。
5. “强行把 residual scale 设为非零”的 stress diagnostic 已确认会把 held-out action loss 放大约 44 倍，且无法区分 matched/mismatched memory；它不能作为 Mem-0 baseline 或 memory utility 证据。
