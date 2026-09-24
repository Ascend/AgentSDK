# 概述

## 简介

本提案描述了在 Agent SDK 框架中，将黑盒 Agent 模式（VirtualAgentEngine）接入现有训推调框架的设计与实现。Agent 的多轮对话推理过程完全在框架外部执行——通过 HTTP API 将 prompt 发送给外部 Agent Service，Agent Service 完成多轮对话（含工具调用）后将轨迹数据存储在 TrajProxy 中，训练侧再从 TrajProxy 取回轨迹进行策略更新。Agent 作为独立 HTTP 服务外置，Agent SDK 训练框架只通过 API 调用和拉取轨迹，不感知内部实现和推理过程，因此称为"黑盒"。

## 动机

在 Agent SDK 框架中引入黑盒 Agent 模式的动机如下：

**外部 Agent 能力复用**：业界已有大量成熟的 Agent 框架和部署服务，具备复杂工具调用、多步推理、丰富的 prompt 策略等能力。Agent SDK 框架无需重新实现这些能力，通过 HTTP 接口即可复用已有 Agent 服务。

**灵活的拓扑部署**：Agent 服务可独立部署，与训练 NPU 节点解耦，允许 Agent 服务和训练服务独立扩缩容。

**轨迹数据统一管理**：外部 Agent 的所有对话记录统一写入 TrajProxy，训练侧批量拉取，抹平了时序依赖。

**支持自定义 Agent 奖励函数**：指定 `agent_name: proxy`，无需通过 `agents_mapping.py` 注册不同的 Agent 配置，无需修改核心训练逻辑。

## 提议方案

在现有训推调框架基础上，新增黑盒 Agent 接入模式，包含通信层（AgentProxyClient / TrajProxyClient）、引擎层（VirtualAgentEngineExecutionWrapper）和接入层（HybridAgentLoopManager / RolloutWorker），支持共卡和训推分离两种运行模式。

## 目标

整体目标如下：

- 建立黑盒 Agent 的接入规范，定义清晰的通信协议和数据流转路径
- 实现 AgentProxyClient → Agent Service → TrajProxyClient 的完整通信链路
- 支持两种运行模式：共卡模式（hybrid）和训推分离模式（one_step_off），通过 `work_mode` 切换
- 不改造外部 Agent Service 和 TrajProxy 的内部逻辑

非目标：

- 不涉及新的强化学习算法设计，仅改造训练流程中轨迹的处理

## 方案设计

### 总体方案

黑盒模式采用三层架构：

**通信层**：与外部黑盒 Agent 服务和 TrajProxy 的 HTTP 通信

- `AgentProxyClient`：将 prompt 封装为 HTTP 请求，发送到外部 Agent 服务
- `TrajProxyClient`：从 TrajProxy 服务拉取轨迹数据用于训练

**引擎层**：调度 Agent 轨迹生成全流程

- `VirtualAgentEngineExecutionWrapper`：编排发送请求 → 取回记录 → refine → reward 的完整流程

**接入层**：将黑盒 Agent 接入训练框架

- `HybridAgentLoopManager`：共卡模式下，训练和推理共享 NPU 卡
- `OneStepOffRolloutExecutor` + `RolloutWorker`：训推分离模式下，推理服务独立部署，训练和推理异步执行

### 技术选型

选择黑盒 Agent 接入模式的理由：

- 复用外部已有 Agent 能力，降低框架内部实现复杂度
- HTTP 协议作为通信方式，通用性强，与现有 Agent 服务兼容性好
- 通过 TrajProxy 统一管理轨迹数据，抹平时序依赖，降低对接复杂度
- 与现有共卡和训推分离模式兼容，对业务方透明

### 功能与性能设计

**通信过程**：

AgentProxyClient 向 Agent Service 发起请求：

```text
POST {agent_addr}/v1/chat/completions
{"model", "messages", "infer_url", "infer_params", ...}
```

Agent Service 收到请求后开始多轮对话，每轮通过 `infer_url` 将结果写入 TrajProxy。

TrajProxyClient 从 TrajProxy 取回轨迹：

```text
GET {traj_addr}/trajectory?session_id={session_id}
```

返回 `RequestRecord` 列表，每条记录对应 Agent 一轮对话。

**数据转换管道**：

`AgentProxyClient 发送请求 → Agent Service 多轮对话并写入 TrajProxy → TrajProxyClient 取回 RequestRecord → traj_refine_func 聚合为 Episode → calc_rewards 计算奖励 → traj_reward_func 计算轨迹奖励 → transform_trajectories_to_batch 转换为 DataProto`

**Agent 类型注册**：

无需通过 `agents_mapping.py` 注册 Agent，通过 `agent_name: proxy` 配置即可接入，支持自定义 `res_reward_func` 和 `traj_reward_func`。

### 数据模型

核心数据结构沿用框架现有的经验数据格式（prompt、response、reward 等字段）。

黑盒模式特有的数据层次为 `RequestRecord → Episode → Trajectory → Step`：

- **RequestRecord**（外部存储层）：session_id、messages、raw_request、raw_response、token_ids、response_ids、logprobs 等字段
- **Episode**：id、trajectories、metrics
- **Trajectory**：steps、reward
- **Step**：prompt_ids、response_ids、logprobs、reward、done

关键机制：AgentProxy 负责"黑盒"交互（LLM 推理 + 工具调用），TrajProxy 负责轨迹持久化，VAEE 负责拉取和聚合。Agent SDK 训练侧不参与 Agent 的工具调用和推理过程。

## 编程与调用设计

### 编程模型基本设计

开发约束：

- 需要启动外部 Agent 服务
- 需要启动 TrajProxy 服务
- 数据需符合框架现有格式规范

可验收设计：

- 提供示例配置文件
- 实现端到端的测试用例

## 接口定义与设计

### VirtualAgentEngineExecutionWrapper

接口描述：编排完整的黑盒轨迹生成流程，包括创建 Agent/Environment、发送请求到 Agent Service、取回并处理轨迹。

接口原型：

```python
class VirtualAgentEngineExecutionWrapper(BaseEngineWrapper):
    async def generate_trajectory(self, task, ...) -> Episode:
        # 1. Create Agent + Environment
        # 2. Send request to Agent Service
        # 3. Fetch + process trajectory
```

输入参数：

- task: AgentTask 实例，包含 prompt、task_id 等任务信息

返回参数：Episode 实例，包含完整的轨迹数据和指标

### AgentProxyClient

接口描述：与外部 Agent Service 通信，发送 chat completion 请求并接收 session_id。

接口原型：

```python
class AgentProxyClient:
    async def get_agent_response(self, prompt_messages, task_id, **kwargs) -> tuple[int, str]:
        """POST {agent_addr}/v1/chat/completions, returns (status_code, session_id)"""
```

输入参数：

- prompt_messages: 对话消息列表
- task_id: 任务标识
- kwargs: 其他请求参数（如 temperature、max_tokens 等）

返回参数：(status_code, session_id) 元组，status_code=0 表示成功

### TrajProxyClient

接口描述：从 TrajProxy 拉取轨迹数据。

接口原型：

```python
class TrajProxyClient:
    async def get_records_by_session(self, session_id) -> List[dict]:
        """GET {traj_addr}/trajectory?session_id={session_id}, return RequestRecord list"""
```

输入参数：

- session_id: 会话标识

返回参数：RequestRecord 列表，每条记录对应 Agent 一轮对话

### HybridAgentLoopManager

接口描述：共卡模式下的 Agent 管理器，在训练循环内同步调用黑盒 Agent 生成轨迹。

接口原型：

```python
class HybridAgentLoopManager(AgentLoopManager):
    def generate_sequences(self, prompts: DataProto) -> DataProto:
```

输入参数：

- prompts: DataProto 格式的 prompt 数据

返回参数：DataProto 格式的轨迹数据，包含 tokenized trajectories

### RolloutWorker / OneStepOffRolloutExecutor

接口描述：训推分离模式下的 rollout 执行器，负责异步消费 prompt、调用黑盒 Agent 生成轨迹、将经验数据放回队列。

接口原型：

```python
class RolloutWorker:

class OneStepOffRolloutExecutor:
    def fit(self):
        while iteration < self.train_iters:
            batch = self.queue_actor.pop_queue()
            trajectories = self.rollout_worker.generate_trajectories(batch_dict)
            self.put_prompts_experience(trajectories, ...)
```

输入参数：

- OneStepOffRolloutExecutor 通过构造函数注入 rollout_worker、train_iters、put_prompts_experience 等依赖

返回参数：无（经验数据通过回调写入队列）

### 外部 Agent 接入规范

外部 Agent 若需接入 Agent SDK 训推框架，需对接以下核心接口：

**Agent Service 接收请求**：

- 协议：HTTP POST
- 路径：`{agent_addr}/v1/chat/completions`
- 请求体：model、messages、infer_url、infer_params 等字段
- 返回：HTTP 200 表示 Agent 已接收并开始执行，返回 session_id

**Agent Service 写入轨迹**：

- Agent 收到请求后执行多轮对话（LLM 推理 + 工具调用），每轮通过 `infer_url` 将结果写入 TrajProxy
- 写入内容包含 messages、raw_request、raw_response、token_ids、response_ids、logprobs、timestamps 等

**TrajProxy 提供轨迹查询**：

- 协议：HTTP GET
- 路径：`{traj_addr}/trajectory?session_id={session_id}`
- 返回：RequestRecord 列表，须包含 session_id、messages、raw_request、raw_response、token_ids、response_ids、logprobs 等必需字段

**可选的轨迹处理回调**：

- `traj_refine_func`：将原始 RequestRecord 聚合为 Episode → Trajectory → Step 结构
- `res_reward_func`：对每一步的工具调用结果计算奖励和终止信号
- `traj_reward_func`：对整条轨迹计算最终奖励

## 文档设计

编程手册将包含以下内容：

- 配置文件说明
- API 参考文档
- 外部 Agent 接入指南
- 示例代码和使用案例

## 缺点和风险

1. **外部依赖风险**：黑盒 Agent 依赖外部 Agent Service 和 TrajProxy 的可用性，服务故障将阻塞训练。

2. **协议耦合**：AgentProxyClient 与 Agent Service 的 HTTP 通信协议需双方对齐，协议变更需同步。

3. **数据同步延迟**：TrajProxy 中轨迹数据的写入到读取存在时延，需要重试机制。

4. **调试困难**：黑盒内部不可见，轨迹异常难以定位。

## 应对措施

- AgentProxyClient 和 TrajProxyClient 均支持重试机制
- 提供 `traj_output_path` 配置，支持将轨迹数据写文件以便离线调试
- 提供 `traj_refine_func` 和 `res_reward_func` 配置接口，支持自定义轨迹处理和奖励
- 提供外部 Agent 接入规范文档，降低对接成本

## 现有技术

黑盒模式借鉴了以下技术：

- **Ray 分布式计算框架**：提供了高效的分布式任务调度和通信机制
- **Actor 模型**：用于实现组件间的异步通信

与其他项目的差异：

- 支持接入外部 Agent 服务，扩展框架适用范围
- 通过 TrajProxy 解耦轨迹生成与训练消费

## 验收标准

在Atlas A3系列产品上，Agent SDK 训推调框架支持 Qwen3-8B 模型的黑盒 Agent 接入。
训练数据集（MATH），相同核心训练超参数下，模型迭代一定步数后的 entropy、grad-norm 有稳定收敛下降趋势，以及 reward 曲线有稳定上升趋势。
硬件：Atlas A2系列产品、Atlas A3系列产品
OS：Ubuntu 22.04 LTS

## 其他补充说明

无
