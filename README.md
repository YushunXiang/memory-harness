# Memory Harness

Memory Harness 是一个面向具身策略实验的可组合记忆框架。项目目标是在固定 benchmark、backbone、数据划分和评测协议的前提下，公平比较不同 memory program，并最终学习何时使用、选择哪种 memory architecture。

每条 memory path 由以下可替换组件组成：

```text
encoder → writer → store → retriever → lifecycle → utilizer
                         ↑
                     controller
```

运行时会记录 `RESET / RETRIEVE / USE / WRITE_DECISION / WRITE` trace，使 memory 的写入、读取、注入和资源成本可审计。

## 项目路线

1. **冻结实验协议**：固定 π0.5 backbone、RMBench 任务、数据划分、seed、success checker 和预算。
2. **建立无记忆基线**：验证 `none` program 与原始 π0.5 在相同输入和 RNG 下行为一致。
3. **复现 Mem-0**：分别验证 executor 侧的 anchor/sliding memory 和 planner 侧的 key memory。
4. **通过 executor readiness gate**：先让 matched-training executor 获得稳定的非零任务成功率，避免在失效策略上比较 memory。
5. **运行固定消融**：在共享 checkpoint 和 seed 上比较 `none / anchor / sliding / anchor+sliding`，至少进入 20-episode pilot。
6. **筛选候选架构**：在统一 token、训练和 rollout 预算下比较无训练候选；只让有正向信号的候选进入 matched retraining。
7. **学习 memory controller**：先学习逐 timestep 的 `use_memory` gate，再评估 episode/subtask 级 architecture selector。
8. **确认与报告**：使用未见过的 seed 做不少于 50 episodes 的 confirmation，并同时报告 success、延迟、token、写入次数和 peak storage。

## 当前状态

| 项目 | 状态 |
|---|---|
| 可插拔 runtime、registry、typed config 和审计 trace | 已完成 |
| 21 个候选架构的 build、smoke 和 checkpoint shape preflight | 已完成 |
| Mem-0 anchor/sliding 推理干预复现 | 已完成：完整 memory `10/10`，去 anchor `2/10`，去 sliding `0/10` |
| Mem-0 planner key/no-key 配对诊断 | 已完成：key `2/10`，no-key `0/10` |
| π0.5 Put Back u1200 executor readiness | 未通过：完整任务均为 `0/3`；第 1 个 subtask 完成率为 full-memory `0/3`、empty-mask `1/3`、native-none `2/3`，均未完成第 2 个 subtask |
| u3000 延长训练与 progress-aware 复评 | 待恢复并完成 |
| 固定 memory Gate-20 | 待 executor readiness 通过后执行 |
| 候选 matched retraining、controller 和 50-episode confirmation | 待完成 |
| Planner memory 的完全插件化与可部署 SEC 边界 | 待完成 |

当前首要瓶颈不是候选数量，而是 π0.5 executor 尚未形成稳定的基础任务能力。在 readiness gate 通过前，项目不会用零成功率 rollout 宣称某种 memory 有效或无效，也不会提前训练 controller。

## 接下来要做

- 恢复并完成 Put Back u3000 的 `native-none / full-memory / empty-mask` 训练链。
- 使用 `0/1/2/3` task-progress 回放检查训练是否产生部分进展，并决定是否进入 Gate-20。
- readiness 通过后，在共享 checkpoint、suite 和 seed 上运行四个固定 memory 消融。
- 对通过 screen 的新架构进行独立、预算匹配的 context generation 与 finetuning。
- 将 planner 的 `none/key` 固定分支替换为统一 registry，并接入论文一致的可部署 SEC 连续判定协议。
- 补齐公开复现所需的环境安装说明、artifact 发布方式和一键最小示例。

## 快速检查

安装为 editable package：

```bash
python -m pip install -e .
```

导出当前组件目录：

```bash
python -m memory_harness.catalog --output /tmp/memory-component-catalog.json
```

对任意 program 运行 build/runtime smoke：

```bash
python -m memory_harness.smoke \
  --config configs/fixed_anchor_sliding.json \
  --output-dir /tmp/memory-harness-smoke
```

运行测试：

```bash
pytest -q
```

详细的架构说明、实验协议、论文审计和逐日进展见 [progress.md](progress.md)，阶段性专项报告见 [docs/](docs/)。
