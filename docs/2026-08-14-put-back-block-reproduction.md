# Put Back Block：Mem-0 × π0.5 复现协议

日期：2026-08-14

## 目标

在 RMBench `put_back_block / demo_clean`（M(1)）上复现 anchor 与 sliding memory 的增益，再把二者作为可插拔 executor paths 交给后续 controller。当前阶段不训练 controller。

## 与 released Mem-0 对齐的部分

- 官方 50 个 HDF5 episodes，动作标签为下一帧 qpos；评测上限 500 environment steps。
- 一个 episode 使用同一全局 instruction，不使用 simulator phase 或 oracle state。
- 最终层 contextual image latent，每个 environment frame 一个 token。
- `anchor=1`，`sliding=30`；sliding 从旧到新右对齐，relative position 中 `1` 表示最新。
- 两个 memory cross-attention 分支均使用官方配置的 dropout `0.1`（评测时关闭）。
- 每步顺序为 `retrieve prior history → use → write current observation`；第一帧同时写入 anchor 与 sliding。
- cached action chunk 内仍逐环境步更新 memory。validator 要求 executor `USE` 次数等于实际 action steps。
- full-memory checkpoint 上以 mask 实现 `full / w/o anchor / w/o sliding / empty`，不为每个消融单独训练模型。

这里以 released implementation 为准：官方 `memory_bank.py` 实际把 conditioning 排成 `[sliding, anchor, text]`，本地 `Mem0Fusion` 保持同一顺序。论文正文公式写成 `[anchor, sliding, text]`，两者存在文字/代码顺序差异；复现实验不能在训练中途据公式交换 token 位置。

## π0.5 port 的明确差异

- 主干是 π0.5，而不是论文的 Qwen3-VL-2B + DiT-B；使用三相机、标准 14-dim Aloha state/action layout 和 π0.5 的 50-step action horizon（官方 Mem-0 executor 使用单头相机与 padded 16-dim layout）。
- 为保持已有 π0.5 RMBench 控制协议，每次 action query 最多执行 10 个 actions；这只缓存 action，memory 仍每环境步更新。
- 单卡 48GB 配置冻结 2B contextual encoder，训练完整 action expert、Mem-0 fusion 和 projection；官方使用 8×A800 联合训练。
- 官方 AdamW 使用 weight decay `0.005` 和 cosine schedule；π0.5 port 沿用 OpenPI 的近零 weight decay，当前阶段采用常数 base LR `1e-5`、action expert/projection LR `1e-4`。这是 backbone optimization recipe 的明确差异。
- 当前学习曲线从 200 optimizer updates 扩展到 1,000 full-memory updates，单卡 effective batch 56；官方为 30,000 iterations、global batch 448。结果必须称为 π0.5 port 的预算学习曲线，不能冒充官方 checkpoint 复跑。
- 官方 executor 使用单头相机 latent；π0.5 port 的 contextual image latent 汇聚三相机图像 token。这是保留 π0.5 perception interface 的有意差异。

## 冻结数据与审计

- LeRobot dataset：`rmbench_lerobot_data/local/rmbench_put_back_block_demo_clean`
- 规模：50 episodes，17,612 frames，30 FPS，14-dim state/action，三相机。
- split seed：`20260814`；40 train / 10 validation，episode 不交叉。
- task spec：`configs/tasks/put_back_block.json`
- template/context：`rmbench_runs/emac_put_back_block_v1/`
- source conversion manifest：`rmbench_runs/put_back_block_data_conversion_20260814.json`

## 实验顺序

1. 在 Mem-0-capable π0.5 backbone 上用全 false `31×2048` mask 训练 none checkpoint，并比较 50/100/150/200-update held-out loss。
2. 从选定 none checkpoint 逐帧生成 matched anchor+sliding context；context bank 必须为 stride 1、无 future leakage、40/10 split 一致。
3. 同预算训练 full-memory checkpoint，先做 held-out `matched / empty / mismatched / remove / replace / shuffle` 内容干预。
4. 用共享 checkpoint 与共享 simulator/policy seed 跑 `none / anchor / sliding / anchor+sliding` full episodes。
5. 只有多 seed success 或阶段进度稳定优于对照，anchor/sliding 才进入 controller 候选集。

## 当前状态

- 数据转换、split、空 context、逐帧 anchor+sliding context 和 GPU 训练链路均已通过；context bank 覆盖 `17,612/17,612` frames，train/validation episode 不交叉。
- none 的 50/100/150/200-update held-out loss 依次为 `11.258 / 5.139 / 1.948 / 1.043`，选定 200-update checkpoint。
- 从该 checkpoint 继续训练的 200-update full-memory checkpoint 已完成；dropout 为论文配置 `0.1`，anchor/sliding fusion、projection 与 action expert 均存在并通过 checkpoint 审计。
- 40 个 held-out 样本的 matched loss 为 `0.2855`，empty 为 `0.2910`，差值 `-0.00552`，cluster-bootstrap 95% CI 为 `[-0.00991, -0.00192]`。去掉 sliding 后为 `0.2888`；去掉 anchor 后为 `0.2863`，后者区间跨 0。
- matched 与 cross-episode mismatched 的差异接近 0，说明当前预算下模型利用了历史通道，尤其是 sliding，但尚未稳定区分具体历史内容。
- 相同 simulator/policy seed 的首个 full-episode smoke 中，anchor+sliding 与 none 均为 `0/1`；视频显示两者都到达 block 附近但未稳定完成第一次抓取。两条动作轨迹平均逐步 L2 差为 `0.278`，说明插件已改变 policy 行为，失败瓶颈仍是低预算 executor 技能。
- anchor+sliding 运行完成 `500` 次环境步 memory update、`501` 次 write，最多使用 `31` tokens，验证了逐环境步 `observe()` 和完整审计链路。
- 200-update full-memory checkpoint 已从完整 optimizer/EMA/data-loader state 精确续训，目标为累计 1,000 full-memory optimizer updates；结束后自动执行 40-sample 离线干预及 `none / anchor / sliding / anchor+sliding / novelty_sliding` 的 3-seed gate。
- 另行排队两个从 π0.5 base 训练 1,200 updates 的预算对照：`empty-mask Mem-0-capable` 保留 memory projection/cross-attention 但永远输入空 mask；`native π0.5` 从模型定义中完全移除 memory。两者均匹配完整 pipeline 的 `200 none + 1,000 memory`、即 67,200 个 optimizer examples。前者控制 memory 内容，后者才是普通 π0.5 方法级 baseline；二者都不替代同一 full-memory checkpoint 上的 mask 消融。

因此当前结论是 **ACTIVE / training-budget scaling**。200 updates 只用于确认端到端实现和学习趋势；在 no-memory baseline 具备非零且稳定的 full-episode success 前，不用 `0/1` 判断 Mem-0 论文消融能否复现，controller 标记为 **deferred until utility gate**。
