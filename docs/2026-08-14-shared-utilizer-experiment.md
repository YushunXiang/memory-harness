# 共享 Memory Utilizer 实验结论（2026-08-14）

## 结论

Memory architecture 可以并且已经拆成可插拔 program；但 π0.5 的 action expert 需要先训练一个共享 `utilize(memory_tokens, memory_mask)` 接口。该接口只训练一次，随后冻结，Agent 才能公平地自由组合 `anchor / sliding / anchor+sliding / key`，而不是每个组合重训一个 policy。

本轮还没有获得可用于闭环评测的 utilizer。所有 checkpoint 均未通过“matched memory 必须优于 mismatched memory”的内容敏感性门槛，因此没有启动 RMBench rollout，也没有训练 controller。

## 已完成实现

- 训练与部署共用 `build_moment_tokens_from_request` 的四类 moment token：query、prefix、history、instruction。
- 训练上下文由真实 harness program 执行 `RETRIEVE → USE → WRITE` 生成，而不是另建近似实现。
- 8,677 个 action-chunk segment，79 个 episode，保持原 63/16 train/validation split。
- 三种 program 近似均衡：anchor 2,919、sliding 2,895、anchor+sliding 2,863。
- matched 只读同 episode 的过去观测；mismatched 来自另一 episode；没有未来帧或 simulator privileged state。
- context bank 为 8×2048 float16 token，约 273 MB；99.1% segment 有非空历史。
- checkpoint 审计要求显式包含 memory cross-attention 的 `residual_scale`，旧的零门 checkpoint 不再误判为可用。

## 实验结果

离线评测固定为 16 个 held-out episode、64 个配对样本。数值为 action loss 差；负值更好。

| 训练协议 / step | matched − no memory | matched − mismatched | 结论 |
|---|---:|---:|---|
| v2：matched/empty/mismatched，249 | +4.56e-5 | +1.32e-8 | 学会忽略 memory |
| v3：仅 valid memory，249 | -6.43e-6 | -7.69e-7 | 方向改善，置信区间跨 0 |
| v3：仅 valid memory，500 | -5.10e-4 | 0 | 通用 context 修正，不读内容 |
| v3：仅 valid memory，750 | -7.35e-4 | 0 | 通用 context 修正，不读内容 |
| v3：仅 valid memory，999 | +1.48e-4 | 0 | 退化 |

v2 同时把正确和错误 memory 配给相同动作标签，优化上鼓励模型关闭 memory 通道。v3 按 Mem-0 的方式只用有效 memory 训练后，step 500 确实显著降低 loss，但替换成另一 episode 的 memory 后结果逐样本不变。context bank 中两者的 token 通常不同，因此这不是数据引用错误，而是模型只利用“存在 memory/mask”这一结构信号，没有利用具体内容。

## 当前判断

该简化 adapter 分支状态为 **STOP**，整个 E-MAC 项目仍为 **ACTIVE**：插件 runtime 已跑通，但 `input-prefix pooled moment token → shared action-expert cross-attention` 没有学到内容依赖。继续增加该分支训练步数没有依据；step 999 已显示退化。当前主线已换成最终层 contextual latent、独立 anchor/sliding fusion、subtask prompt 与 phase-reset 的 Mem-0 port。

后续改为忠实 Mem-0 port：从 VLM 最终层中按图像 token 做 mean pooling，得到一个已经与语言上下文化的 image latent；使用 anchor=1、sliding=30、相对位置和两个独立 Pre-LN fusion branch。这里的问题不是 mean pooling 本身，而是旧分支池化了尚未经过 VLM contextualization 的 π0.5 input embeddings，且 utilization 结构也与 Mem-0 不同。

## 产物

- 数据构造：`../memory_harness/build_training_data.py`
- 配对清单：`../../rmbench_runs/emac_runtime_moment_v2/pairing_manifest.valid_memory_only.json`
- context bank：`../../rmbench_runs/emac_runtime_moment_v2/context_bank.npz`
- 数据审计：`../../rmbench_runs/emac_runtime_moment_v2/pairing_audit.json`
- v3 checkpoints：`../../rmbench_checkpoints/pi05_aloha_pen_uncap_context_adapter/emac_shared_utilizer_v3_valid_memory_20260814/`
- 离线结果：`../artifacts/2026-08-14-shared-utilizer-v3/`
