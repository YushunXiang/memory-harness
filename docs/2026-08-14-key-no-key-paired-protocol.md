# Mem-0 Key / w/o Key 配对协议

日期：2026-08-14

## 目的

只检验 planner-side key history 是否提供了当前 observation 无法恢复的阶段历史。该实验不比较不同 executor，也不把 oracle phase boundary 视为 deployable 方法。

## 输入协议

| 条件 | Planner 输入 | Planner 模型 |
|---|---|---|
| key | global task、episode 初始 RGB、有序 completed-subtask 文本与结束 RGB | `mem0-cover-blocks-key-planner` |
| w/o key | global task、每个阶段边界的当前 RGB | `mem0-cover-blocks-no-key-planner` |

`w/o key` 协议来自 released `memorymatters_planner_without_key.py`。仓库未公开其 YAML/system prompt 和权重，因此本地模型按可见的 `prepare_qwen_input()` 格式独立 SFT；不能称为官方 checkpoint 复跑。

## Planner 训练匹配

- 数据：相同 50 个 Cover Blocks demo episodes、相同 300 个阶段标签；只改变 planner 输入表示。
- global task：使用下载数据中的 left-to-right 指令，修正 released converter 的 right-to-left 硬编码矛盾。
- base：Qwen3-VL-8B-Instruct。
- recipe：LoRA rank 8、target all、LR `1e-4`、25 epochs、seed 7。
- 单卡等效 batch：`1 × gradient accumulation 128 = 128`。
- optimizer steps：key 与 no-key 均为 75。
- key 数据 SHA256：`399e4836dd9755130da4161febd83a0fcfc2ca9014b2e41e121f1a69321e558b`。
- no-key 数据 SHA256：`468bf61fa80c7ea2dddab8a20a8af1dbcea9236f0cc08fc2d02c8cd94ef574a9`。
- 配对审计：`rmbench_runs/emac_mem0_planner_key_no_key_data_pair_20260814.json`，确认 300 个 labels 与 global tasks 逐项一致；key 图像数按阶段为 1—6，no-key 每条恰好一张当前图。

## Closed-loop 固定项

- executor checkpoint：`rmbench_checkpoints/pi05_aloha_pen_uncap_mem0/emac_mem0_pi05_v4_direct_accum28_calibration1400_b2_20260814/1399`；
- executor memory program：none（31 个零 token 与全 false mask）；
- simulator seed、policy seed、layout、camera、chunk=10、max steps=1500 完全配对；
- planner boundary：`oracle_prompt_change`，仅作 nondeployable diagnostic；
- planner temperature：0；每个模型独立记录 calls、latency、PLAN/USE/WRITE trace。

## 运行顺序与判定

1. 先验证两个 planner 对 300 条训练格式样本的 parse validity，并抽查六种颜色—位置 permutation。
2. 先跑 shared seed `100000` 作为 end-to-end pilot；配对器必须确认全部 protected variables 一致。
3. 若阶段轨迹有差异，扩到同一组 10 seeds；单 seed 不用于 Gate 1。
4. 报告 full success、max reward、total reward、planner 序列、calls 和 latency。
5. 旧的 frozen-initial-image no-key run 已标记 `invalidated_protocol_mismatch`，不得合入结果。

## 4-seed 结果

- 训练格式 exact match：key `30/30`；w/o key `26/30`，且 4 个错误均为 cover 方位识别。
- full success：两组均为 `0/4`。
- 平均 max reward：key `0.0750`；w/o key `0.0625`；delta `+0.0125`。
- 平均 total reward：key `91.175`；w/o key `79.475`；delta `+11.70`。
- paired direction：key 2 胜、1 平、1 负。

结论：忠实本地复现支持 key history 的局部阶段价值，但尚未复现论文成功率幅度，也未通过稳定 utility gate。机器可读汇总位于 `rmbench_runs/emac_key_vs_no_key_faithful_4seed_20260814.json`。
