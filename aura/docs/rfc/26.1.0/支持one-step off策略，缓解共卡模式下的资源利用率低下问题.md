# 概述

## 简介

本提案描述了Agentic RL项目中one-step off-policy模式的设计与实现。该模式是一种高效的强化学习训练范式，通过分离的rollout和训练流程，实现了数据生成与模型更新的解耦，从而提高训练效率和资源利用率。

## 动机

在传统的强化学习训练中，rollout（经验生成）和训练（模型更新）通常是串行执行的，但它存在严重的效率问题， 模型更新必须等待生成阶段最长的输出完成。 在生成长尾样本的过程中，NPU 保持空闲，导致资源利用率严重不足。 样本生成中的长尾问题越严重，整体训练效率就越低导致计算资源利用率低下。one-step off-policy模式通过异步执行rollout和训练过程，允许模型在生成新经验的同时利用上一步生成的样本进行当前的训练，一定程度上降低了长尾样本在生成期间的NPU空闲时间。由于训练保证了永远使用上一步所生成的样本，因此命名为单步策略(one-step off policy)。目前单步策略并且在LLM RL上已经验证精度以及具备较高的收敛稳定性。在Agentic RL训推调框架上，支持Qwen3系列（Qwen3-4B/8B/32B/30B-A3B）训推分离+one step off policy训练模式。

## 提议方案

## 目标

整体目标如下：

- 实现rollout与训练的异步执行，提高计算资源利用率
- 提供灵活的配置选项，支持不同规模的训练任务
- 保证训练过程的稳定性和可重复性
- 非目标：不涉及新的强化学习算法设计，仅关注训练流程的优化

## 方案设计

### 总体方案

one-step off-policy模式采用分布式架构，主要包含以下组件：

- **Rollout组件**：负责生成经验数据，包括OneStepOffRollouter和OneStepOffRolloutExecutor
- **训练组件**：负责模型训练和更新，包括OneStepOffTrainExecutor和TrainDataLoader
- **控制器组件**：协调rollout和训练过程，包括TrainController和RolloutController
- **通信组件**：实现组件间的数据传输和状态同步

系统采用Ray作为分布式计算框架，通过Actor模型实现组件间的通信和任务调度。整体流程如下：

1. 初始化rollout worker和控制器
2. Rollout端准备就绪后，向训练端发送信号
3. Rollout端开始生成经验数据并存储到共享队列
4. 训练端从队列中获取经验数据进行训练
5. 训练端定期将更新后的模型权重发送给rollout端
6. 重复步骤3-5直到训练完成

### 技术选型

选择one-step off-policy模式的理由：

- 平衡了实现复杂度和训练效率
- 适合大语言模型的强化学习训练场景
- 能够充分利用分布式计算资源
- 与现有代码架构兼容性好

### 功能与性能设计

**Rollout流程**：

- 初始化rollout worker和控制器
- 加载初始模型权重
- 生成经验数据并进行预处理
- 将处理后的数据发送到训练端

**训练流程**：

- 初始化训练环境和数据加载器
- 从rollout端接收经验数据
- 执行模型训练和评估
- 定期更新模型权重到rollout端

**权重更新机制**：

- 训练端在完成一定迭代后，将更新后的权重发送给rollout端
- Rollout端接收新权重并更新本地模型
- 支持增量更新，减少数据传输量

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

可验收设计：

- 提供示例配置文件
- 实现端到端的测试用例

## 接口定义与设计

### OneStepOffRollouter

接口描述：OneStepOffRollouter是rollout过程的主要执行器，负责协调rollout worker生成经验数据

接口原型：

```python
class OneStepOffRollouter:
    def __init__(self, controller, rollout_worker, train_iters, padding_dict_to_tensor_dict, put_prompts_experience, **kwargs):
        pass
    
    def fit(self):
        pass
```

输入参数：

- controller: RolloutController实例，用于协调rollout过程
- rollout_worker: RolloutWorker实例，用于生成经验数据
- train_iters: 训练迭代次数
- padding_dict_to_tensor_dict: 数据预处理函数
- put_prompts_experience: 经验数据存储函数
- kwargs: 其他配置参数

返回参数：无

#### OneStepOffTrainExecutor

接口描述：OneStepOffTrainExecutor是训练过程的主要执行器，负责模型训练和权重更新

接口原型：

```python
class OneStepOffTrainExecutor:
    def __init__(self, controller, *args, **kwargs):
        pass
    
    def fit(self):
        pass
    
    def update_weights_to_rollout_unit(self, last_iteration: bool, iteration: int) -> None:
        pass
```

输入参数：

- controller: TrainController实例，用于协调训练过程
- args: 位置参数
- kwargs: 配置参数，包括validate_freq、test_before_train、update_weights_interval等

返回参数：无

调用参考代码：

```python
# 初始化训练执行器
executor = OneStepOffTrainExecutor(controller, **train_config)
# 开始训练
executor.fit()
```

## 文档设计

编程手册将包含以下内容：

- 配置文件说明
- API参考文档
- 示例代码和使用案例

## 缺点和风险

1. **实现复杂度**：相比同步训练模式，one-step off-policy模式的实现复杂度更高，需要更多的组件协调和状态管理。

2. **数据一致性**：方案核心会导致rollout和训练使用不同版本的模型权重，影响训练稳定性。

## 应对措施

- 提供详细的文档以及参数配置指南，提升易用性
- 提供模型最佳实践，在该配置下训练效率提升，且精度无明显降低。

## 现有技术

one-step off-policy模式借鉴了以下技术：

- **Ray分布式计算框架**：提供了高效的分布式任务调度和通信机制
- **Actor模型**：用于实现组件间的异步通信
- **经验回放机制**：用于存储和重用经验数据

与其他项目的差异：

- 专门针对大Agent智能体训练训练进行了优化
- 分离后端推理VLLM引擎，支持更多特性

## 验收标准

在A3服务器上，Agentic RL训推调框架支持 Qwen3系列模型（Qwen3-4B/8B/32B/30B-A3B）的 训推分离+one step off policy训练。
在训练数据集（[https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset],  Agent采用DeepSWE Agent[https://huggingface.co/agentica-org/DeepSWE-Preview], 精度指标：持平GPU（reward曲线收敛稳定的指标 或者 任务成功率pass@1）,1epoch的端到端训练时长较 共卡+on policy模式 降低20%以上
硬件：Ascend  A2/A3
OS：Ubuntu 22.04 LTS

## 其他补充说明

无
