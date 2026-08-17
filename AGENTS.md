# AGENTS.md

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
- Study how established products solve the problem before designing a solution. Adopt their proven patterns and conventions rather than inventing an approach from scratch.

# Memory Harness 开发约束

本目录内的设计与实现必须坚持 **模块化、接口化、可插拔、可组合**，目标是让后续 Research Agent 能在不修改主干代码的情况下，自由组装和替换 memory architecture。

## 核心约束

1. Memory 系统应拆分为独立组件，至少包括：
   - `encode`：历史信息如何表示；
   - `write`：何时写入；
   - `store`：保存位置、容量与数据结构；
   - `retrieve`：何时、如何查询；
   - `utilize`：如何注入 planner、policy 或其他下游模块；
   - `lifecycle`：更新、遗忘、TTL 与 reset；
   - `controller`：选择和组合上述模块。
2. 每个组件必须具有稳定、明确、可测试的输入输出接口，不得依赖其他组件的内部实现。
3. 新方法应通过注册表、配置文件或 typed program 接入；不得把任务专属条件硬编码进核心执行流程。
4. `none / anchor / sliding / anchor+sliding / key` 必须是同一接口下可替换的实现，而不是彼此独立的 runner 分支。
5. 组合关系必须由配置描述，使 Agent 能通过修改 program/config 完成组装；只有提出新 operator 时才需要修改组件代码。
6. π0.5 backbone、数据、evaluator、seed split 与实验预算默认独立于 memory program。任何改变必须显式记录，不能与 memory 改动混在一起。
7. 部署候选只能使用真实部署时可观测信息。Simulator privileged state 仅允许用于明确标注的 oracle/diagnostic，不得进入 deployable program。
8. 每次运行必须记录 program 配置、组件版本、WRITE–RETRIEVE–USE–RESET trace、资源成本和结果，保证可审计与可复现。

## 测试要求

- 每个组件需要独立单元测试和接口契约测试。
- 必须测试跨组件组合、容量边界、TTL、episode reset 和异常输入。
- `none` program 在相同输入与 RNG 下应与原始无记忆 π0.5 保持一致。
- 新组件不得破坏已有组件；新增实现后应运行 registry 与组合回归测试。

## Agent 修改边界

Research Agent 可以在允许的 memory operator、controller 和配置目录内提出、实现和组合方案，但默认不得修改 benchmark、success checker、locked test、π0.5 主干和数据划分。违反接口、预算、可观测性或测试约束的候选应自动拒绝并回滚。
