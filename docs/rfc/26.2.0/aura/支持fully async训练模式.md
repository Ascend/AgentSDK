# 概述

## 简介

本提案描述了Agent SDK项目中fully async（完全异步）训练模式的设计与实现。该模式在one-step off-policy基础上进一步演进，通过rollout与training的完全解耦、流式数据传输、多步异步权重同步与新鲜度控制，实现了rollouter生成与trainer训练的时间重叠（overlap），显著缓解长尾样本带来的NPU空闲问题，提升整体训练吞吐。

## 动机

one-step off-policy模式通过一轮异步执行缓解了rollout与训练串行带来的资源浪费，但其强制使用一轮异步的数据，存在不够灵活、无法完全消除长尾影响等问题。在大规模Agentic RL训练中，rollout阶段的长尾样本（如多轮工具调用、长reasoning链）会持续造成部分NPU空闲，而one-step off-policy无法让rollout在trainer训练时继续生成新样本。

fully async模式借鉴verl、AReaL、StreamRL、AsyncFlow等系统的设计，通过分离架构实现异步、流式训练：rollout逐样本生成并流入队列，trainer逐样本消费并训练，两者通过队列解耦；同时通过staleness控制在吞吐与精度间取得平衡，通过trigger控制权重同步频率降低同步开销。在合理设置资源分配与参数同步频率的情况下，fully async能够显著提升训练效率，且训练效果无明显退化。

## 提议方案

在现有训推分离框架基础上，新增fully async训练模式，包含流式数据队列、多步异步权重同步、新鲜度控制，保持one-step off-policy模式行为不受影响。

## 目标

整体目标如下：

- 实现rollout与training的完全解耦，两者通过流式队列异步执行，时间上重叠
- 提供新鲜度控制(staleness_threshold)，允许使用旧参数生成的样本训练，在吞吐与精度间平衡，支持多步异步(0.x步到多步)
- 通过trigger_parameter_sync_step灵活控制权重同步频率
- 保证训练过程的稳定性和可重复性
- 不影响已有的one-step off-policy模式

非目标：

- 不涉及新的强化学习算法设计，仅关注训练流程的优化
- 不涉及PartialRollout（打断进行中的rollout并下次继续）特性
- 不涉及基于NCCL/NIXL的高效参数同步（当前采用文件IO），作为后续性能优化项

## 方案设计

### 总体方案

fully async模式采用分布式架构，主要包含以下组件：

- **Rollout组件**：负责流式生成经验数据，包括FullyAsyncRolloutExecutor和RolloutWorker
- **训练组件**：负责模型训练和权重更新，包括FullyAsyncTrainer
- **队列组件**：实现rollout与trainer之间单样本粒度的解耦，即SampleQueue(Ray Actor)
- **控制器组件**：协调rollout和训练过程，包括TrainController和RolloutController
- **权重管理组件**：协调rollout与trainer之间的权重版本同步，包括RolloutWeightManager

系统采用Ray作为分布式计算框架，通过Actor模型实现组件间的通信和任务调度。整体流程如下：

1. 初始化rollout worker、trainer和SampleQueue
2. Rollout持续逐样本生成经验数据，逐个放入SampleQueue
3. Trainer从SampleQueue逐样本获取经验数据，攒到required_samples后进行一次本地训练
4. Trainer训练trigger_parameter_sync_step轮后，触发一次权重同步到rollout
5. Rollout在staleness达上限或队列堆积时，查询是否有新权重，若有则加载并重置staleness；若无则等待重试。
6. 重复步骤2-5直到训练完成
7. 训练结束/异常时，trainer通知SampleQueue shutdown，rollout感知后停止生成

### 技术选型

选择fully async模式的理由：

- 能够充分利用分布式计算资源
- 与现有代码架构兼容性好
- 在one-step off-policy基础上进一步消除长尾样本导致的NPU空闲，提升训练吞吐
- 通过trigger_parameter_sync_step和staleness_threshold组合，灵活支持on-policy到多步异步的多种策略

### 功能与性能设计

**Rollout流程**：

- 初始化rollout worker和控制器
- 加载初始模型权重
- 流式生成经验数据，逐样本放入SampleQueue
- 每个样本入队前检查staleness和队列堆积，达上限时背压等待或加载新权重
- 样本携带生成时的权重版本，供trainer端做重要性采样修正

**训练端流程**：

- 初始化训练环境和数据加载器
- 从SampleQueue逐样本获取经验数据，攒到required_samples后训练
- 每 trigger_parameter_sync_step轮训练后，触发一次权重同步到rollout
- 训练结束/异常时，通知SampleQueue shutdown，信号贯穿调用链终止rollout

**权重更新机制**：

- Trainer端通过local_trigger_step计数，每trigger_parameter_sync_step轮训练才推一次权重
- Rollout端通过staleness_samples计数和队列堆积双条件判断是否加载新权重
- 权重更新后staleness重置为当前队列在途样本数

### 数据模型

经验数据结构：

- 包含prompt、response、reward等字段
- 支持自定义额外字段（通过dataset_additional_keys配置）
- 使用张量格式存储，支持高效的批处理操作

## 编程与调用设计

### 编程模型基本设计

开发约束：

- 需要配置Ray集群环境
- 模型文件需要符合指定格式
- 数据需要预处理为指定格式
- 需要合理设置staleness_threshold、trigger_parameter_sync_step等异步参数

可验收设计：

- 提供示例配置文件
- 实现端到端的测试用例

## 接口定义与设计

### FullyAsyncTrainer

接口描述：FullyAsyncTrainer是训练过程的主要执行器，负责从SampleQueue流式取数、训练、权重同步与终止信号触发。

接口原型：

```python
class FullyAsyncTrainer(SeparateRayPPOTrainer): 
    def init(self, config, controller, sample_queue=None, **kwargs):
        pass

    async def fit(self) -> None:
        pass
```

输入参数：

- config: 训练配置，包含trigger_parameter_sync_step、staleness_threshold等
- controller: TrainController实例，用于协调训练过程
- sample_queue: SampleQueue Actor引用，fully_async模式注入，one_step_off模式为None
- kwargs: 其他配置参数

返回参数：无

### SampleQueue

接口描述：SampleQueue是rollout与trainer之间的流式数据队列，以Ray Actor形式提供跨进程的单样本粒度生产/消费。

接口原型：

```python
@ray.remote(num_cpus=1, max_concurrency=20)
class SampleQueue:
    def init(self, max_queue_size: int = 1000): 
        pass

    async def put_sample(self, sample: Any) -> bool:
        pass

    async def get_sample(self) -> Any:
        pass

    async def get_queue_size(self) -> int:
        pass

    def shutdown(self) -> None:
        pass
```

输入参数：

- max_queue_size: 队列最大容量,超出时丢弃最旧样本

返回参数：无

### RolloutWorker(fully async相关接口)

接口描述：RolloutWorker在fully async模式下提供流式生成、权重更新、staleness控制等能力。

接口原型：

```python
class RolloutWorker:
    async def generate_sequences_fully_async(self, actual_batch_num=1):
        pass

    async def update_model_weights_fully_async(self, actual_batch_num=1):
        pass

    def set_fully_async_config(self, sample_queue, max_required_samples):
        pass

    async def _put_sample_streaming(self, outputs, rollout_metrics):
        pass

    async def _load_weights_fully_async(self, weights_version):
        pass
```

调用参考代码：

```python
sample_queue = SampleQueue.remote(max_queue_size=required_samples * 2)
rollout_worker.set_fully_async_config(sample_queue, max_required_samples)
trainer = FullyAsyncTrainer(config, controller, sample_queue=sample_queue)
await trainer.fit()
```

## 文档设计

编程手册将包含以下内容：

- 配置文件说明
- API参考文档
- 示例代码和使用案例

## 缺点和风险

1. **实现复杂度**:相比one-step off-policy，fully async模式的实现复杂度更高，涉及流式队列、staleness控制、权重版本管理、优雅终止等多组件协调和状态管理。

2. **数据一致性**:方案核心允许rollout和训练使用不同版本的模型权重，且允许使用stale样本训练，影响训练稳定性。staleness_threshold设置过大可能导致off-policy偏差累积，训练效果退化甚至发散。

## 应对措施

- 提供详细的文档以及参数配置指南，提升易用性
- 提供staleness_threshold与trigger_parameter_sync_step的调参建议，推荐staleness_threshold < 1

## 现有技术

fully async模式借鉴了以下技术：

- **verl fully_async_policy**：提供了fully async的核心设计参考，包括流式数据流、staleness控制、trigger权重同步、Rollout Importance Sampling等
- **Ray分布式计算框架**：提供了高效的分布式任务调度和通信机制
- **Actor模型**：用于实现组件间的异步通信
- **经验回放机制**：用于存储和重用经验数据

与其他项目的差异：

- 分离后端推理VLLM引擎，支持更多特性
- 在one-step off-policy基础上演进，保持模式隔离，不影响已有特性

## 验收标准

在Atlas A3系列产品上，Agent SDK训推调框架支持Qwen3-32B的训推分离 + fully async训练。
稳定性指标：在Math数据集上，相同核心训练超参数下，通过控制权在同步更新频率trigger_parameter_sync_step和样本新鲜度staleness_threshold，保证在一定步数后的entropy、grad_norm稳定收敛且有下降趋势，以及reward曲线有稳定上升趋势。
硬件:Atlas A2系列产品、Atlas A3系列产品
OS:Ubuntu 22.04 LTS

## 其他补充说明

无
