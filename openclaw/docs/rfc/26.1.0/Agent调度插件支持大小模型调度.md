# 1. 概述

## 1.1 简介

在OpenClaw Agent执行复杂任务时，单一Agent处理全流程存在效率低、成本高、质量不稳定等问题。本提案提供一套Agent调度插件系统（Subagent Coordinator），包含三个协作插件：任务规划插件（Taskr）、可观测性插件（Observability）和执行监控插件（Exec Monitor）。通过启发式复杂度评分将任务分为L1-L5五个操作符级别，并支持按文件、步骤、领域三种策略分解任务，帮助Agent合理规划工作流，提升复杂任务的处理效率与质量。

## 1.2 动机

当前OpenClaw Agent面临以下核心痛点：

- **任务处理粗放**：Agent直接处理复杂任务，缺乏结构化规划，容易遗漏步骤
- **重复执行浪费**：相似任务每次从零开始，缺乏任务复用与状态继承机制
- **进度不可见**：长任务执行过程中缺乏进度跟踪，用户无法了解当前状态
- **失败恢复困难**：任务执行中断后需从头开始，缺乏断点续做能力
- **并行化不足**：无依赖的子任务无法并行执行，浪费计算资源

**用户案例：**

- 某开发者要求Agent重构一个大型项目，Agent一次性处理导致内存不足，且遗漏部分文件
- 某用户要求Agent分析多份文档并生成报告，Agent顺序处理耗时过长

**不做此提案的影响：**

- 复杂任务成功率低，用户体验差
- 大模型Token消耗高，运营成本上升
- 长任务不可控，用户信任度下降

## 1.3 目标

**目标：**

- 提供任务管理插件，支持任务创建、更新、查询、列表等CRUD操作
- 提供启发式复杂度评估，从步骤数、文件数、上下文依赖、确定性、关键词五个维度评分
- 支持L1-L5五层操作符级别分类，对应不同的处理策略建议
- 支持三种任务分解策略：按文件分解、按步骤分解、按领域分解
- 提供任务依赖图管理，支持循环依赖检测、关键路径计算、并行分组识别
- 提供检查点保存与恢复机制，支持断点续做
- 提供任务笔记系统，支持上下文记录、进度跟踪、错误记录

**非目标：**

- 不替代模型选择决策（复杂度评估提供建议，最终由主Agent决策）
- 不涉及多Agent并行执行的运行时调度（由OpenClaw核心调度）
- 不提供自动化的子Agent创建与委托执行（当前阶段为任务规划层）
- 不涉及跨机器分布式调度

# 2. 用例分析

## 2.1 用例1：复杂任务分解与规划

**场景描述：** 用户提交一个涉及多文件、多步骤的复杂开发任务，Agent需要合理分解并规划执行顺序。

**功能点：**

- 创建主任务并评估复杂度
- 根据任务特征自动推荐分解策略（按文件/步骤/领域）
- 生成子任务列表并建立依赖关系
- 识别可并行执行的子任务组

**DFX要求：**

- **准确性**：分解策略与任务特征匹配
- **完整性**：子任务覆盖主任务全部范围
- **可追踪性**：子任务与主任务关联清晰

## 2.2 用例2：任务执行状态跟踪

**场景描述：** 长任务执行过程中，用户需要了解当前进度与已完成的子任务。

**功能点：**

- 更新子任务状态（open/wip/done/skipped）
- 添加进度笔记
- 查询任务树与依赖关系
- 查看可执行的下一个子任务

**DFX要求：**

- **实时性**：状态更新及时反映
- **可视化**：任务依赖关系清晰可查询
- **可审计性**：笔记记录完整操作历史

## 2.3 用例3：任务分析与建议

**场景描述：** Agent需要评估任务复杂度并获取处理建议。

**功能点：**

- 对任务描述进行复杂度评分（1-10分）
- 分类操作符级别（L1-L5）
- 获取处理建议（分解、优先级调整、委托建议）
- 记录分析结果到任务笔记

**DFX要求：**

- **准确性**：复杂度评估与实际情况相符
- **可解释性**：评分维度与依据透明
- **可操作性**：建议具体可执行

# 3. 方案设计

## 3.1 总体方案

Agent调度插件采用插件化架构，作为OpenClaw Plugin SDK插件运行。系统由三个协作插件组成：

```text
┌───────────────────────────────────────────────────────────────────────┐
│                        Subagent Coordinator                            │
│                         插件系统架构                                    │
├─────────────────────────┬─────────────────────────────┬────────────────┤
│      Taskr 插件         │    Observability 插件       │  Exec Monitor  │
│   (任务规划与分解)      │     (可观测性监控)           │   (执行保障)    │
├─────────────────────────┼─────────────────────────────┼────────────────┤
│  Tools:                 │  Tools:                      │  Tools:        │
│  - create_task          │  - get_token_usage           │  - quality_    │
│  - get_task             │  - get_tool_metrics          │    gate_check │
│  - update_task          │  - get_agent_metrics          │  - retry_     │
│  - list_tasks           │  - get_trace                 │    strategy   │
│  - decompose_task       │  - get_cost_breakdown        │  - save_      │
│  - add_note             │  - get_trend_analysis        │    checkpoint │
│  - score_complexity     │  - detect_sensitive          │  - restore_   │
│  - classify_operator    │  - sanitise_data             │    checkpoint │
│                         │  - check_rate_limit          │                │
│  Services:              │  - get_budget_status        │  Services:     │
│  - TaskStore             │                              │  - Checkpoint │
│  - TaskGraph             │  Services:                   │    Manager    │
│  - ComplexityScorer      │  - MetricsCollector          │  - Retry      │
│                         │  - CostTracker                │    Strategy   │
│  Hooks:                 │  - TraceRecorder              │                │
│  - task_analyzed         │  - RateLimiter                │  Hooks:       │
│  - decomposition_        │  - TrendAnalyzer              │  - before_    │
│    requested            │  - Sanitiser                  │    delegation │
│  - checkpoint_save      │  - DashboardServer            │  - after_     │
│  - checkpoint_restore   │                               │    execution  │
│                         │  Hooks:                       │  - quality_    │
│                         │  - task_analyzed              │    gate       │
│                         │  - route_decision             │  - checkpoint_│
│                         │                               │    save       │
│                         │                               │  - checkpoint_│
│                         │                               │    restore    │
└─────────────────────────┴─────────────────────────────┴────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      OpenClaw Plugin SDK       │
                    │      (MCP工具注册+事件系统)     │
                    └───────────────────────────────┘
```

**核心设计思路：**

1. **多插件协作**：系统由三个专项插件组成，通过统一事件总线协调
2. **插件化集成**：作为OpenClaw Plugin SDK插件运行
3. **启发式评估**：基于任务元数据（步骤数、文件数、描述关键词）进行复杂度评分，无需额外LLM调用
4. **策略化分解**：支持按文件、按步骤、按领域三种分解策略，适应不同任务类型
5. **图结构管理**：任务依赖关系以有向图管理，支持拓扑排序、循环检测、并行分组
6. **运行时路由**：基于复杂度评估结果，决策任务应分配给subagent还是主agent处理

**技术平台选择：**

- **插件框架**：OpenClaw Plugin SDK
- **类型系统**：TypeScript + @sinclair/typebox（参数校验）
- **状态存储**：内存存储 + 可选持久化接口
- **事件通信**：统一事件总线，支持插件间通信

## 3.2 技术选型

### 3.2.1 复杂度评估方案对比

| 方案 | 优势 | 劣势 | 结论 |
|------|------|------|------|
| 启发式评分（关键词+元数据） | 无额外LLM调用开销，响应快 | 对语义理解有限 | **采用** |
| LLM-based评分 | 语义理解准确 | 增加Token消耗与延迟 | 未来扩展 |
| 规则引擎 | 可解释性强 | 维护成本高，灵活性差 | 放弃 |

### 3.2.2 任务存储方案对比

| 方案 | 优势 | 劣势 | 结论 |
|------|------|------|------|
| 内存存储 + 可选持久化 | 简单高效，插件内自包含 | 进程重启数据丢失（无持久化时） | **采用** |
| 外部数据库 | 数据持久化 | 增加部署依赖 | 未来扩展 |
| 文件存储 | 简单持久化 | 并发性能差 | 备选 |

### 3.2.3 放弃方案说明

| 方案 | 放弃理由 |
|------|----------|
| 独立守护进程服务 | 保持插件轻量，通过Plugin SDK集成 |
| 自动化子Agent委托 | 当前阶段聚焦任务规划层，执行调度由核心负责 |
| 动态阈值机器学习 | 复杂度评估以启发式为主，ML作为未来增强 |
| 跨机器分布式调度 | 单机场景为主，超出当前范围 |

## 3.3 功能与性能设计

### 3.3.1 复杂度评估设计

**评分维度：**

| 维度 | 权重 | 评估依据 | 高分特征 |
|------|------|----------|----------|
| steps | 3 | 任务步骤数 | 步骤多（>10） |
| files | 3 | 涉及文件数 | 文件多（>10） |
| dependency | 2 | 上下文依赖程度 | 强依赖前文结果 |
| determinism | 1 | 任务确定性 | 创造性/开放性任务 |
| keywords | 1 | 描述关键词匹配 | 含analysis/design等关键词 |

**关键词分类：**

| 类别 | 关键词示例 | 影响 |
|------|-----------|------|
| 高复杂度 | analysis, design, architecture, algorithm, optimize, migration | 增加复杂度评分 |
| 低复杂度 | copy, move, rename, list, format, validate | 降低复杂度评分 |
| 创造性 | generate, create, write, creative, novel | 降低确定性得分 |
| 调试类 | debug, fix, error, bug, crash | 中等复杂度 |
| 安全类 | auth, security, encrypt, credential | 中等复杂度 |

**操作符级别分类：**

| 级别 | 名称 | 复杂度范围 | 描述 | 委托规则 | 推荐运行时 |
|------|------|-----------|------|----------|-----------|
| L1 | Simple | ≤2 | 简单单步操作，高确定性 | ALWAYS_DELEGATE | subagent |
| L2 | Batch | 3-4 | 批量多文件/实体操作 | DELEGATE_WITH_SPLIT | subagent |
| L3 | Processing | 5-6 | 数据处理、分析或转换 | DELEGATE_WITH_CHECKPOINT | subagent |
| L4 | Analysis | 7-8 | 复杂分析、设计或多步推理 | DELEGATE_WITH_SUPERVISION | acp |
| L5 | Complex | ≥9 | 高度复杂，需完整上下文 | MAIN_AGENT_ONLY | acp |

**评分计算流程：**

```text
任务描述 + 元数据
       │
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 步骤数评分   │    │ 文件数评分   │    │ 依赖程度评分 │
│ (0-3分)     │    │ (0-3分)     │    │ (0-2分)     │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 确定性评分   │    │ 关键词评分   │    │ 加权汇总     │
│ (0-2分)     │    │ (0-2分)     │    │ (归一化1-10) │
└─────────────┘    └─────────────┘    └─────────────┘
                                          │
                                          ▼
                                    ┌─────────────┐
                                    │ 操作符级别   │
                                    │ 分类(L1-L5) │
                                    │ + 置信度    │
                                    └─────────────┘
```

### 3.3.2 任务分解设计

**分解策略：**

| 策略 | 适用场景 | 分解方式 |
|------|----------|----------|
| by_file | 涉及多文件的任务 | 按文件批次分组，每批最多5个文件 |
| by_step | 步骤明确的任务 | 按步骤顺序拆分，识别then/next等连接词 |
| by_domain | 跨领域复杂任务 | 按数据层/API层/UI层/安全/测试/部署等领域拆分 |

**by_domain领域识别：**

| 领域关键词 | 对应层 |
|-----------|--------|
| data, database, storage | data_layer |
| api, endpoint, service | api_layer |
| ui, interface, frontend | ui_layer |
| auth, security, permission | security |
| test, testing, qa | testing |
| deploy, devops, infrastructure | deployment |

**分解后依赖关系：**

- 同策略内默认建立顺序依赖（子任务i依赖子任务i-1）
- by_file策略中多批次可标记并行组（parallelGroup）
- by_domain策略中不同track可并行（Track A: 研究→分析, Track B: 实现→测试→文档）

### 3.3.3 任务依赖图设计

**图结构：**

```text
┌─────────────────────────────────────────────────────┐
│                    任务依赖图                         │
│                                                     │
│   主任务                                            │
│     │                                               │
│     ├─► 子任务A (data_layer) ──► 子任务B (analysis) │
│     │          Track A                              │
│     │                                               │
│     ├─► 子任务C (implementation) ──► 子任务D (test) │
│     │          Track B              (可并行)         │
│     │                                               │
│     └─► 子任务E (file_batch_1) ──► 子任务F (file_batch_2)
│     │                                               │
│   循环依赖检测 ──► 拓扑排序 ──► 并行分组识别          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**图分析能力：**

| 能力 | 说明 |
|------|------|
| 构建图 | 从任务存储构建完整依赖图 |
| 可执行任务 | 获取当前无未满足依赖的任务 |
| 循环检测 | DFS检测并返回所有循环路径 |
| 关键路径 | 按estimatedDuration计算最长路径 |
| 拓扑排序 | 返回满足依赖顺序的任务ID列表 |
| 并行分组 | 按深度分组，识别可并行任务 |

### 3.3.4 检查点机制设计

**检查点数据结构：**

```typescript
interface CheckpointData {
  taskId: string;           // 关联任务ID
  subtasks: Subtask[];      // 子任务列表
  completedSubtasks: string[]; // 已完成子任务ID
  results: Map<string, ExecutionResult>; // 执行结果
  timestamp: number;        // 保存时间戳
}
```

**恢复流程：**

```text
检查点ID/任务ID
       │
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 查找检查点   │    │ 验证有效性   │    │ 计算恢复状态 │
│ (精确匹配/   │ -> │ (时效性/     │ -> │ (剩余任务/  │
│  最新匹配)   │    │  完整性)     │    │  进度百分比) │
└─────────────┘    └─────────────┘    └─────────────┘
                                          │
                                          ▼
                                    ┌─────────────┐
                                    │ 返回恢复建议 │
                                    │ + 警告信息  │
                                    └─────────────┘
```

**恢复验证：**

- 时效性检查：检查点超过24小时发出警告
- 完整性检查：已完成的子任务是否仍存在于当前任务
- 依赖检查：剩余子任务的依赖是否已满足

### 3.3.5 插件工具接口设计

**注册工具列表：**

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| create_task | 创建任务 | description, parentId, priority, steps, files |
| get_task | 获取任务详情 | taskId, includeNotes, includeDependencies |
| update_task | 更新任务 | taskId, status, priority, description |
| list_tasks | 任务列表 | status, parentId, priority, sortBy, limit |
| decompose_task | 分解任务 | taskId, strategy, maxSubtasks |
| add_note | 添加笔记 | taskId, content, type, author |
| score_complexity | 复杂度评分 | task (description, steps, files) |
| classify_operator | 级别分类 | complexity (total, breakdown, keywords) |

### 3.3.6 事件钩子设计

| 钩子 | 触发时机 | 功能 |
|------|----------|------|
| before_delegation | 任务委托前 | 基于复杂度评分决策运行时类型（subagent/acp） |
| after_execution | 任务执行后 | 记录执行结果，更新任务状态 |
| task_analyzed | 任务分析完成 | 记录分析结果到笔记，生成处理建议 |
| route_decision | 路由决策时 | 综合评估后确定任务分配策略 |
| quality_gate | 质量门禁 | 执行前检查任务合理性，执行后验证结果 |
| decomposition_requested | 收到分解请求 | 确定分解策略，生成子任务列表 |
| checkpoint_save | 保存检查点 | 验证数据完整性，记录进度笔记 |
| checkpoint_restore | 恢复检查点 | 查找检查点，验证有效性，计算恢复状态 |

## 3.4 安全隐私与DFX设计

### 3.4.1 安全设计

- **输入校验**：所有工具参数通过@sinclair/typebox进行类型校验
- **状态隔离**：插件状态独立，不影响其他插件或核心系统
- **错误处理**：工具执行异常返回结构化错误，不泄露内部状态

### 3.4.2 DFX设计

**兼容性：**

- OpenClaw Plugin SDK兼容
- TypeScript类型安全

**可维护性：**

- 服务层与工具层分离，便于单元测试
- 复杂度评分参数可配置（权重、阈值）
- 分解策略可扩展（新增策略类型）

**可测试性：**

- 复杂度评分可独立测试（给定任务描述验证评分）
- 任务图算法可独立验证（拓扑排序、循环检测）
- 分解策略可独立测试（验证子任务覆盖完整性）

**可靠性：**

- 可选持久化接口，支持状态保存
- 检查点恢复时多重验证，避免无效恢复
- 错误处理完善，单工具失败不影响其他工具

## 3.5 编程与调用设计

### 3.5.1 编程模型基本设计

**开发环境设计：**

- **软件环境**：Node.js, TypeScript
- **开发框架**：OpenClaw Plugin SDK
- **类型校验**：@sinclair/typebox

**开发约束：**

- 插件需遵循OpenClaw Plugin SDK接口规范
- 工具参数定义需完整且类型准确
- 状态管理需考虑并发安全

**可验收设计：**

- 插件注册成功，工具可用
- 复杂度评分与预期结果一致
- 任务分解结果覆盖完整
- 检查点保存与恢复正确

### 3.5.2 接口定义与设计

#### 3.5.2.1 task.create（任务创建接口）

**接口描述：** 创建新任务，支持层级结构

**接口原型：**

```typescript
api.registerTool({
  name: "create_task",
  parameters: Type.Object({
    description: Type.String(),
    parentId: Type.Optional(Type.String()),
    priority: Type.Optional(Type.Union([
      Type.Literal("low"),
      Type.Literal("normal"),
      Type.Literal("high"),
      Type.Literal("urgent"),
    ])),
    estimatedDuration: Type.Optional(Type.Number()),
    steps: Type.Optional(Type.Number()),
    files: Type.Optional(Type.Array(Type.String())),
  }),
})
```

**返回结果：**

- 成功：返回创建的任务对象（含ID、创建时间）
- 失败：返回错误信息（描述为空、父任务不存在等）

#### 3.5.2.2 task.decompose（任务分解接口）

**接口描述：** 将任务分解为子任务

**接口原型：**

```typescript
api.registerTool({
  name: "decompose_task",
  parameters: Type.Object({
    taskId: Type.String(),
    strategy: Type.Union([
      Type.Literal("by_file"),
      Type.Literal("by_step"),
      Type.Literal("by_domain"),
    ]),
    maxSubtasks: Type.Optional(Type.Number({ default: 10 })),
  }),
})
```

**返回结果：**

- 成功：返回父任务、子任务列表、并行分组、预估总耗时
- 失败：返回错误信息（任务不存在、已存在子任务等）

#### 3.5.2.3 task.scoreComplexity（复杂度评分接口）

**接口描述：** 对任务进行多维度复杂度评分

**接口原型：**

```typescript
api.registerTool({
  name: "score_complexity",
  parameters: Type.Object({
    task: Type.Object({
      description: Type.String(),
      steps: Type.Optional(Type.Number()),
      files: Type.Optional(Type.Array(Type.String())),
    }),
  }),
})
```

**返回结果：**

| 字段 | 类型 | 说明 |
|------|------|------|
| total | number | 归一化总分（1-10） |
| breakdown | object | 各维度原始得分 |
| keywords | string[] | 匹配到的关键词 |

#### 3.5.2.4 task.classifyOperator（操作符分类接口）

**接口描述：** 基于复杂度评分分类操作符级别

**接口原型：**

```typescript
api.registerTool({
  name: "classify_operator",
  parameters: Type.Object({
    complexity: Type.Object({
      total: Type.Number(),
      breakdown: Type.Object({...}),
      keywords: Type.Array(Type.String()),
    }),
  }),
})
```

**返回结果：**

| 字段 | 类型 | 说明 |
|------|------|------|
| level | string | 操作符级别（L1-L5） |
| name | string | 级别名称 |
| description | string | 级别描述 |
| confidence | string | 置信度（high/medium/low） |

# 4. 缺点和风险

## 4.1 潜在风险

| 风险项 | 风险描述 | 影响等级 | 应对措施 |
|--------|----------|----------|----------|
| 评分偏差 | 启发式评分无法准确理解任务语义 | 中 | 提供置信度指标，复杂任务建议人工复核 |
| 分解不完整 | 自动分解可能遗漏关键步骤 | 中 | 分解后提供子任务列表供确认 |
| 状态丢失 | 无持久化时进程重启数据丢失 | 中 | 提供可选持久化接口，重要任务及时保存检查点 |
| 循环依赖 | 手动设置的依赖可能形成循环 | 低 | 提供循环检测工具，创建时自动检测 |
| 性能瓶颈 | 大量任务时图算法性能下降 | 低 | 任务数量在合理范围内，算法复杂度可接受 |

## 4.2 负面影响

- **学习成本**：用户需理解操作符级别与分解策略的概念
- **评估开销**：复杂度评分增加任务处理的前置步骤
- **存储开销**：任务状态与检查点占用内存

## 4.3 实现成本

- **开发工作量**：插件框架集成、工具实现、服务层开发
- **测试验证**：复杂度评分准确性测试、分解策略覆盖测试
- **维护成本**：关键词库更新、分解策略优化

## 4.4 兼容性考虑

- 插件遵循OpenClaw Plugin SDK规范，与核心版本兼容
- 复杂度评分参数可配置，适应不同场景需求
- 检查点数据结构稳定，升级时保持向后兼容

# 5. 现有技术

## 5.1 参考项目

### 5.1.1 OpenClaw Plugin SDK

- **借鉴点**：插件注册机制、MCP工具接口、事件系统
- **差异点**：Taskr插件专注于任务调度领域，提供领域特定工具集

### 5.1.2 MCP (Model Context Protocol)

- **借鉴点**：工具调用接口标准化
- **差异点**：在MCP基础上增加任务管理领域语义

## 5.2 技术差异优势

| 维度 | 手动任务管理 | Taskr插件调度 |
|------|-------------|--------------|
| 复杂度评估 | 主观判断 | 多维度量化评分 |
| 任务分解 | 人工拆分 | 策略化自动分解 |
| 依赖管理 | 无 | 图结构+拓扑排序 |
| 进度跟踪 | 无 | 状态+笔记系统 |
| 断点续做 | 无 | 检查点保存恢复 |

# 6. 未解决问题

1. **LLM-based复杂度评估**：当前为启发式评分，未来可引入LLM进行更准确的语义理解
2. **自动化委托执行**：当前仅提供任务规划与建议，子Agent的自动创建与委托执行由核心调度
3. **持久化存储增强**：当前为内存存储+可选持久化，未来可支持数据库存储
4. **动态阈值学习**：根据历史任务数据自动优化评分阈值
5. **多Agent并行调度**：跨多个Agent实例的并行任务分配与结果聚合

---

附录

- **术语表：**
  - **Task**：任务单元，包含描述、步骤、文件、优先级等元数据
  - **Subtask**：子任务，从主任务分解产生的独立执行单元
  - **Complexity Score**：复杂度评分，1-10的归一化分数
  - **Operator Level**：操作符级别，L1-L5五级分类
  - **Decomposition Strategy**：任务分解策略（by_file/by_step/by_domain）
  - **Checkpoint**：检查点，保存任务执行状态用于恢复
  - **Parallel Group**：并行组，无依赖关系的可并行执行子任务集合
