# E-MAC 当前执行计划 v2

## 本周目标

在同一个 Cover Blocks π0.5 checkpoint 上完成可训练、可消融的忠实 Mem-0 executor port，并得到第一轮内容敏感性证据；不训练 controller。

## 顺序

1. **忠实接口**：最终层 contextual image latent；anchor=1；sliding=30；相对位置；两个独立 Pre-LN fusion branch；3-token action conditioning。
2. **数据重建**：使用完整 anchor+sliding program 的 `RETRIEVE → USE → WRITE` 生成 31-slot matched history；executor 输入当前 subtask prompt，phase 变化时清空两类 memory；mismatch 只用于 held-out 检查。早期轮换三种 program、使用全局 prompt 的 v3 数据已废止。
3. **训练预检**：从已 fine-tune 的 Cover Blocks π0.5 初始化，先跑 1-step 和短 smoke，确认显存、梯度、checkpoint 与 audit。单卡联合更新全部参数实测需要约 53.1GiB，超过 48GB；单卡 recipe 因此冻结 2B contextual encoder，训练完整 action expert、fusion 与 projection。
4. **小规模训练**：先运行上述单卡 recipe，检查 matched 是否优于 empty/mismatch；之后用双卡 full-model recipe 单列“是否更新 contextual encoder”的训练消融。官方配置还使用 action model `1e-4`、其余模块 `1e-5` 的分组学习率；当前 OpenPI 单一 LR recipe 只作为可运行校准，正式训练前需实现等价 param groups 或记录为明确差异。
5. **忠实 key 消融**：key planner 使用初始图和 completed-subtask text/RGB；`w/o key` 使用当前 observation 和独立训练的 planner。两者绑定不同模型 ID，但共享 executor、seed、边界和预算。
6. **任务匹配评测**：anchor/sliding 先放到 RMBench `M(1)`；Cover Blocks 重点评 key planner，避免用单一 M(n) 任务否定 anchor/sliding。
7. **下一模块**：先筛选无需学习参数的 MemoryVLA adjacent merge store；有增量后做 matched finetune。MEM video/language、HAMLET learned consolidator 和 Chameleon 依次等待 fixed utility gate，避免同时扩大表示、融合与 controller 三个变量。

## 本周停止项

- 不再训练旧 8-token input-prefix pooled adapter。
- 不把 planner-side key memory 当作 action latent token。
- 不增加 RNN/LSTM、直接拼接或重复 keyframe store。
- fixed module 没有通过内容敏感性与闭环 utility gate 前，不训练 controller 或开放 Agent search。

实现差异记录：官方 Mem-0 fusion 使用 0.1 attention dropout；当前 JAX/NNX 训练器会把 `MultiHeadAttention` 内部 RNG state误纳入 EMA 参数树，导致首步类型错误。因此 π0.5 port 暂时固定 dropout=0；这不改变 anchor/sliding、位置编码、Pre-LN、残差或 3-token conditioning 结构，后续若需要复现正则化项，应在训练器支持非参数 state 后单独加入。
