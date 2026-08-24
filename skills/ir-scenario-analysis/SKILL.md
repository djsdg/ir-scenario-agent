---
name: ir-scenario-analysis
description: 将 IR 的 5W2H/DFX 信息匹配到一个或多个场景与 UC，并判断复用或新增
tags: [IR, 场景, SC, UC, use case, 5W2H, DFX, 匹配]
---

# IR → SC → UC 分析 Skill

本 Skill 必须与 `config/ir_sc_uc_spec.json` 一起使用。Spec 是业务规范，不是模型提示词的替代品：它定义字段映射、枚举、影响因素维度、质量输出和写入前校验。

## 1. 解析 IR

逐项提取，不从常识补写原文没有的信息：

- code、title、description、source
- Who、When、Where、What、How、Why、How Much
- Restrict/constraints
- performance、reliability、serviceability、maintainability、sales、delivery time

缺失值保留为 `null` 或空数组，并列入 `missing_required_fields`。

## 2. 匹配顺序

先调用 `match_ir_requirement`，再核对候选详情。若需要新建或细化，调用只读的 `draft_scenario_from_ir` / `draft_use_cases_from_ir`，先处理其 `missing_required_fields`，最后才考虑写工具。匹配不是只看名称，必须比较：

1. 业务目标、故障表现和 Action 是否一致；
2. Actor 是否一致；
3. When/Where 与生命周期、影响部件是否兼容；
4. 影响因素名称及选中值是否兼容；
5. IR 约束是否被场景覆盖；
6. UC 的前置条件、触发事件、主成功场景、扩展场景和保证是否覆盖 IR 行为链。

场景影响因素按 Spec 的六个维度检查：

- 环境因子：硬件环境、组网场景、协议连接；
- 活动因子：存储架构、业务场景、运维场景。

同时从人机交互、设备交互、系统外部接口、系统边界、运维场景、周边设备/工具六个视角识别，避免漏场景。

## 3. 决策

- `reuse_scenario_and_uc`：场景维度一致，已有 UC 已覆盖触发—处理—保证链。
- `reuse_scenario_create_uc`：场景上下文可复用，但需要新增行为分支或新的 UC。
- `create_scenario_and_uc`：Actor、生命周期、影响因素、业务目标等场景边界不兼容。
- `needs_clarification`：IR 或待创建对象缺少必填字段。

一个 IR 可以命中多个 SC，也可以映射到多个 UC。不要为了减少数量而强行合并独立行为链。

## 4. 新建约束

场景硬性必填：

- description、category、business_goal、actor、actions、lifecycle、constraints、owner
- influence_factors（至少一项，每项有 kind、dimension、name 和至少一个 selected_value）

UC 硬性必填：

- description、actor、preconditions、trigger_event
- success_guarantee、minimum_guarantee
- 至少一个 main_success_scenario 步骤

缺少任何硬性字段时只输出待补清单，不调用写工具。新增默认使用 `draft`，保留 `source_ir_ids`，所有写入均需审批。

禁止把模块直接当场景、把 IR 原文直接套成场景，或绕过场景要素库临时造场景。
