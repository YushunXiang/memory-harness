# Mem-0 × π0.5 v4 校准记录（2026-08-14）

## 结论摘要

项目状态为 **ACTIVE / reproduction in progress**。旧 shared adapter 分支已经停止；当前主线是 Mem-0 的 anchor、sliding 和 planner-side key memory 在 π0.5 上的模块化移植。

本轮发现并修正了四个会妨碍复现的实现差异：

1. v3 context bank 使用全局 prompt，且没有在 subtask phase 变化时清空 executor memory；v4 已改为当前 subtask prompt、完整 anchor+sliding 训练和 phase reset。
2. π0.5 action expert 原先额外使用了零初始化 `residual_scale × sigmoid gate`，而官方 Mem-0 action DiT 直接 cross-attend `[sliding, anchor, text]`。500 steps 后该额外门的有效量级仍约为 `10^-7`，因此已删除；当前实现使用纯 memory K/V cross-attention，并只在 mask 全空时保持 identity。
3. 旧 Cover Blocks port 只在每次 π0.5 action query 时写 memory；由于每个 query 连续执行 10 个 actions，30-slot sliding 实际覆盖约 300 个环境步。官方 Mem-0 在 cached action chunk 内仍逐环境步调用 `update_obs()`，因此 sliding 始终覆盖最近 30 个环境步。通用 evaluator 现已加入独立 `observe()` 路径，训练 context 默认 stride 也改为 1；Put Back Block 从这一修正版开始复现。
4. 旧 port 的 Mem-0 fusion dropout 为 `0.0`，released executor config 为 `0.1`；Put Back Block full-memory 训练已改为 `0.1`。none 条件所有 history mask 均为 false，因此正在运行的 none checkpoint 不受此改动影响。

## 数据与训练对齐

- 数据：RMBench Cover Blocks，train/validation episode 隔离。
- 表征：最终层 contextual image latent，每个时刻一个 token。
- 布局：anchor 1 slot + sliding 30 slots；执行顺序为 `RETRIEVE → USE → WRITE`。
- 时间单位：每个 environment frame 写一次；action chunk caching 只减少 action sampling，不降低 memory update 频率。
- 生命周期：subtask phase 变化时同时 reset anchor 和 sliding。
- 训练：只使用 matched causal history；empty/mismatched 只用于 held-out intervention。
- 学习率：action expert 与 memory input projection 为 `1e-4`，其余可训练模块为 `1e-5`，global clip `2.5`。

## 已完成校准

所有数值均为 16 个 held-out episode、64 个配对时刻、相同 flow noise 下的 action loss；它们不是 RMBench success rate。

| checkpoint | matched | empty | mismatched | 主要结论 |
|---|---:|---:|---:|---|
| 额外零门，500 steps，batch 2 | 0.226842 | 0.226843 | 0.226848 | 几乎严格不读取 memory |
| 直接 cross-attention，500 steps，batch 2 | 0.236038 | 0.238333 | 0.236012 | matched 相比 empty 改善 0.00230，但尚不能区分另一 episode 的同阶段 memory |
| 直接 cross-attention，2000 steps，batch 2 | 0.482289 | 0.482964 | 0.482647 | validation loss 上升；小 batch 配论文 LR 不稳定 |
| 直接 cross-attention，50 optimizer updates，effective batch 56 | 0.088976 | 0.089125 | 0.089204 | 训练稳定；删除/替换 memory 总体变差，开始出现内容敏感性 |

2000-step 模块干预的平均 loss：

| 条件 | loss | 相比 matched 的恶化 |
|---|---:|---:|
| matched | 0.482289 | 0 |
| replace anchor | 0.482282 | -0.000007 |
| remove anchor | 0.482721 | +0.000431 |
| replace sliding | 0.482525 | +0.000235 |
| replace all | 0.482647 | +0.000358 |
| shuffle sliding | 0.482721 | +0.000432 |
| remove sliding | 0.483091 | +0.000801 |
| empty | 0.482964 | +0.000675 |

主要方向与论文消融一致：sliding 删除的负面影响大于 anchor 删除；但当前 cluster bootstrap CI 仍跨 0，不能宣称 utility gate 已通过。

effective-batch-56 checkpoint 的 64 个配对时刻结果：

| 条件 | loss | 相比 matched 的恶化 |
|---|---:|---:|
| matched | 0.088976 | 0 |
| remove anchor | 0.089293 | +0.000316 |
| remove sliding | 0.089218 | +0.000242 |
| replace anchor | 0.089144 | +0.000167 |
| replace sliding | 0.089149 | +0.000173 |
| replace all | 0.089204 | +0.000227 |
| shuffle sliding | 0.089045 | +0.000069 |
| empty | 0.089125 | +0.000149 |

其中 `replace anchor` 的 episode-cluster bootstrap 95% CI 不跨 0；扩大到 256 个时刻后各干预方向大体一致，但因独立 episode 仍只有 16 个，cluster CI 跨 0。因此这是“executor 已开始使用 memory 内容”的证据，还不是成功率提升的最终结论。

## 当前动作

这里的 effective batch 56 是本地单卡稳定性校准值，不是论文官方训练预算。RMBench v3 论文报告的 Mem-0 executor 是按任务从头训练，global batch 448、30,000 iterations、8 张 A800；当前 π0.5 port 只进行了 50 次 optimizer update，约少三个数量级。因此该 checkpoint 只能用于验证接口、梯度和初步 memory-content sensitivity，不能据此判定 Mem-0 utility 是否复现成功。runner 使用 batch 2、28-step gradient accumulation；裁剪发生在梯度求均值之后。

闭环测试分为两层：

- oracle-phase 仅用于 executor diagnostic，结果显式标记为不可部署；
- 正式 Cover Blocks 需要 planner 产生当前 subtask prompt，并由 key memory 保存初始 observation 与 completed subtasks，不能用 simulator phase 替代。

## 初步闭环配对结果

同一 checkpoint、simulator seed `100000`、policy seed `120000`，oracle phase prompt：

| 程序 | success | max reward | total reward | 阶段轨迹 |
|---|---:|---:|---:|---|
| anchor + sliding | 0 | 0.15 | 174.25 | 进入第一个 uncover 阶段，未完成全任务 |
| none（同一 memory-trained checkpoint） | 0 | 0.15 | 173.10 | 进入第一个 uncover 阶段，未完成全任务 |

两者输出 action 不同，但单 seed 的成功率和最大阶段 reward 相同；当前不能宣称闭环提升。首次 `none` 尝试还暴露了 harness 形状接口错误：Mem-0 checkpoint 需要每步接收固定 31-slot 输入。已修正为 `none = 31 个零 token + 全 false mask`，并通过 21 项定向测试。

扩展到 4 个严格配对 seed 后：

| seed | anchor+sliding max / total reward | none max / total reward | 阶段判定 |
|---:|---:|---:|---|
| 100000 | 0.15 / 174.25 | 0.15 / 173.10 | 持平 |
| 100001 | 0.10 / 123.20 | 0.00 / 0.00 | memory 改善 |
| 100002 | 0.10 / 125.75 | 0.10 / 126.40 | 持平 |
| 100003 | 0.10 / 124.40 | 0.05 / 66.50 | memory 改善 |
| 均值 | 0.1125 / 136.90 | 0.0750 / 91.50 | 阶段 reward 更高 |

两条件的整任务成功数仍都是 `0/4`。因此当前结论是：action-side anchor+sliding 在 2/4 seed 上带来阶段级进度改善，但 50 次 effective-batch-56 optimizer update 还没有产生整任务 success 改善。这仍只是 action-side executor diagnostic，不是 Mem-0 在 Cover Blocks 上的 full system 复现；论文该任务的主要增益来自 planner-side key memory。

此外，这组旧 Cover Blocks rollout 使用的是每 10 个环境步写一次的粗粒度 sliding。它可以保留为接口/方向性诊断，但不能再作为对官方 30-step sliding 的忠实复现；新的 Put Back Block 表格只接受逐环境步 `observe()` 且审计中 `executor_observation_updates == action_steps` 的 runs。

## Key 对照审计更正

一次早期单 seed smoke 曾得到 `key: max_reward=0.15, total_reward=175.15`，以及 `planner_no_key: 0.05, 66.55`。该差异不能作为论文 key 消融复现：审计 Mem-0 released code 后发现，`memorymatters_planner_without_key.py` 在每个阶段读取**当前 observation**，而早期 harness 的 no-key 分支错误地一直复用 episode 初始图，因而会人为造成 planner 重复第一阶段。

当前已采用严格协议：

- `key`：初始 observation + 有序 completed-subtask 文本/结束 RGB，使用 key planner checkpoint；
- `w/o key`：每次只输入 global task + 当前 observation，使用单独 SFT 的 no-key planner checkpoint；
- 两者共享同一个 π0.5 executor checkpoint、simulator seed、policy RNG、阶段边界和预算。

因此，旧 `175.15 vs 66.55` 只保留为接口 smoke，不进入 utility gate、proposal 结果或论文表格。修正后的配对 rollout 才是有效证据。

复现边界：仓库公开了 no-key 的 `prepare_qwen_input()` 和逐阶段 current-observation 示例，但没有公开它引用的 `planning_module_inference_without_key.yaml` 或训练数据。因此本地 SFT 严格复现可见输入格式，system prompt 则按该格式重建；结果应称为 local reproduction，而不是官方 checkpoint 复跑。

## 忠实 Key / w/o Key 本地复现结果

两个 planner 使用相同 50 个 episodes、300 个阶段标签、Qwen3-VL-8B base、LoRA rank 8、75 optimizer steps 和 seed 7；执行时共享同一个 π0.5 checkpoint、policy RNG、layout、1500-step budget 与 oracle phase boundary。30 条分层训练格式验证为 `key 30/30`、`w/o key 26/30`；no-key 的 4 个错误均为 left/middle/right cover 方位判断错误。

| seed | key max / total reward | w/o key max / total reward | delta(total) |
|---:|---:|---:|---:|
| 100000 | 0.15 / 172.85 | 0.10 / 125.15 | +47.70 |
| 100001 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 |
| 100002 | 0.05 / 67.15 | 0.10 / 126.35 | -59.20 |
| 100003 | 0.10 / 124.70 | 0.05 / 66.40 | +58.30 |
| 均值 | 0.0750 / 91.175 | 0.0625 / 79.475 | +11.70 |

两组 full success 均为 `0/4`，key 为 2 胜、1 平、1 负。因此本地结果支持 key history 的**局部阶段收益**，但没有复现论文 `68 -> 5` 的成功率幅度，也没有通过启动 controller 所需的稳定 utility gate。汇总 artifact：`rmbench_runs/emac_key_vs_no_key_faithful_4seed_20260814.json`。

## Adjacent-merge matched-training 结果

MemoryVLA-style `adjacent_merge_store` 先在 seed 100000 的同-checkpoint zero-shot screen 中相对 sliding 得到 `+49.30` total reward，因此按规则进入 matched finetune。训练 context 由部署时同一个 consolidating program 生成，共 8677 个片段，非空 context 比例为 `94.54%`；checkpoint 使用与 sliding 相同的 π0.5 base、50 次 optimizer update、effective batch 56 和学习率 `1e-5`，并通过 Mem-0 参数结构审计。

64 个 held-out 时刻的离线审计中，matched 相比 mismatched 的 loss 低 `0.000186`，但相比 empty 高 `0.000697`，bootstrap 区间均跨 0，未通过内容 utility gate。4 个共享 seed 的 oracle-phase matched-training diagnostic 为：

| seed | sliding max / total reward | consolidating max / total reward | delta(total) |
|---:|---:|---:|---:|
| 100000 | 0.10 / 125.75 | 0.15 / 174.50 | +48.75 |
| 100001 | 0.05 / 65.55 | 0.05 / 65.30 | -0.25 |
| 100002 | 0.15 / 174.35 | 0.10 / 125.90 | -48.45 |
| 100003 | 0.10 / 123.75 | 0.05 / 66.00 | -57.75 |
| 均值 | 0.1000 / 122.350 | 0.0875 / 107.925 | -14.425 |

两边 full success 均为 `0/4`，consolidating 为 1 胜 3 负。结论是：单 seed 正向信号不稳定，matched finetune 没有使 adjacent merge 优于 sliding；该模块不进入 controller 候选集。这个比较改变了 store 和各自匹配训练后的 checkpoint，属于预算匹配的架构比较，不是同-checkpoint mask 消融。
