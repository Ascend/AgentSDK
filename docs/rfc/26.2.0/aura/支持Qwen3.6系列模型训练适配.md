# 概述

## 简介

本提案描述了Agent SDK项目中Qwen3.6系列模型（27B/35B）训练适配的设计与实现。该工作通过模型结构适配、配置项扩展和算子兼容性验证，使现有训推调框架能够稳定支持Qwen3.6系列大语言模型的强化学习训练，扩大框架可支持的模型范围。

## 动机

随着Qwen3.6系列模型的发布，27B和35B规格在推理质量与训练成本之间提供了新的平衡点，成为Agentic RL场景下的重要候选基座。当前框架已完成Qwen3系列（Qwen3-4B/8B/32B/30B-A3B）的适配，但Qwen3.6在网络结构、分词器、归一化层及部分算子实现上存在差异，直接复用现有配置会导致加载失败或训练精度异常。为避免业务方重复适配工作，并保证训练精度和资源利用率，需要在框架侧统一完成Qwen3.6系列（27B/35B）的模型适配。

## 提议方案

在现有训推调框架基础上，新增Qwen3.6-27B和Qwen3.6-35B两款模型的自适应配置，包含模型结构注册、分词器配置、权重映射规则、超参默认值和精度对齐策略，不引入新的对外接口。

## 目标

整体目标如下：

- 支持Qwen3.6-27B和Qwen3.6-35B在Agent SDK训推调框架下的端到端训练
- 复用现有训练接口和流程，不新增对外API
- 保证训练精度与Qwen官方实现一致，loss曲线和reward曲线收敛稳定
- 提供开箱即用的参考配置和最佳实践

非目标：

- 不涉及模型结构本身的修改
- 不涉及新的强化学习算法设计

## 方案设计

### 总体方案

Qwen3.6系列适配主要包含以下工作：

- **模型结构适配**：注册Qwen3.6模型结构，处理与Qwen3在RMSNorm、旋转位置编码（RoPE）、注意力机制上的差异
- **分词器适配**：兼容Qwen3.6分词器的新增token和特殊token处理
- **权重映射**：建立官方权重字段到框架内部字段的双向映射规则，支持权重加载和增量更新
- **配置项扩展**：在模型配置中新增Qwen3.6-27B和Qwen3.6-35B的默认超参模板
- **算子兼容性验证**：验证关键算子（FlashAttention、MoE路由等）在Qwen3.6下的数值稳定性

整体流程如下：

1. 下载Qwen3.6官方权重和分词器
2. 使用框架提供的权重转换脚本生成框架内部格式
3. 加载适配后的模型配置，启动训练
4. 训练过程中复用现有的rollout和训练流程
5. 通过精度对齐用例验证适配正确性

### 技术选型

选择在现有框架上做适配而非新增独立模块的理由：

- 复用已有的分布式训练、rollout、权重同步逻辑，降低维护成本
- 与Qwen3系列适配方案保持一致，便于后续版本演进
- 对业务方透明，无需修改训练脚本

### 功能与性能设计

**模型加载流程**：

- 通过配置文件指定模型类型为Qwen3.6
- 框架根据模型类型自动选择对应的模型结构和权重映射规则
- 支持从本地路径权重

**训练流程**：

- 复用现有的训练执行器和数据加载器
- 根据模型规格（27B/35B）提供不同的参考启动脚本，包含可用的batch size、学习率、梯度累积步数

### 数据模型

适配工作不引入新的数据结构，复用现有的经验数据格式：

- prompt、response、reward等字段保持不变
- 分词器内部处理Qwen3.6新增token，对外接口透明
- 支持张量格式存储和批处理操作

## 编程与调用设计

### 编程模型基本设计

开发约束：

- 需要准备Qwen3.6官方模型权重和分词器文件
- NPU显存需要满足27B/35B模型的训练需求
- 数据格式需符合框架现有规范

可验收设计：

- 提供Qwen3.6-27B和Qwen3.6-35B的示例配置文件

## 接口定义与设计

本次适配不新增任何对外Python接口，模型切换完全通过配置文件实现。参照仓库 [aura/configs/README.md](https://gitcode.com/Ascend/AgentSDK/blob/master/aura/configs/README.md) 的命名规范，为Qwen3.6-27B和Qwen3.6-35B分别新增共卡模式、分离模式的训练配置文件，以及分离模式下的推理配置文件，业务方在 [base.conf](https://gitcode.com/Ascend/AgentSDK/blob/master/aura/configs/base.conf) 中指定对应的train/infer配置文件名即可一键拉起，无需修改任何训练代码。

### 新增配置文件清单

| 模式 | 配置文件路径 | 说明 |
| --- | --- | --- |
| 训推共卡 | `aura/configs/train/verl_train_hybrid_A3_t16_qwen3_27b_math_fsdp.yaml` | Qwen3.6-27B 共卡训练配置 |
| 训推共卡 | `aura/configs/train/verl_train_hybrid_A3_t16_qwen3_35b_math_fsdp.yaml` | Qwen3.6-35B 共卡训练配置 |
| 训推分离 | `aura/configs/train/verl_train_async_A3_t16_qwen3_27b_math_fsdp.yaml` | Qwen3.6-27B 分离训练配置 |
| 训推分离 | `aura/configs/train/verl_train_async_A3_t16_qwen3_35b_math_fsdp.yaml` | Qwen3.6-35B 分离训练配置 |
| 训推分离 | `aura/configs/infer/vllm_infer_i16_qwen3_27b.yaml` | Qwen3.6-27B 分离推理配置 |
| 训推分离 | `aura/configs/infer/vllm_infer_i16_qwen3_35b.yaml` | Qwen3.6-35B 分离推理配置 |

### 共卡模式关键配置项

参照 [verl_train_hybrid_A3_t16_qwen3_4b_math_fsdp.yaml](https://gitcode.com/Ascend/AgentSDK/blob/master/aura/configs/train/verl_train_hybrid_A3_t16_qwen3_4b_math_fsdp.yaml) 模板，业务方需修改以下参数：

- `hydra.searchpath`：指向当前代码仓中 `aura/configs/train/verl_conf` 目录的绝对路径
- `verl_conf.data.train_files` / `verl_conf.data.val_files`：共卡模式使用parquet格式数据集路径
- `verl_conf.actor_rollout_ref.model.path`：Qwen3.6模型权重路径
- `train_instances.0.executor_kwargs.rollout_config.llm_tokenizer_path`：Qwen3.6 tokenizer路径
- `verl_conf.actor_rollout_ref.rollout.tensor_model_parallel_size` / `data_parallel_size`：按卡数调整，相乘结果需等于总卡数
- `infer_instances.0.name` / `model_name`：模型名称，需与权重目录名一致

### 分离模式关键配置项

训练侧参照 [verl_train_async_A3_t16_qwen3_4b_math_fsdp.yaml](https://gitcode.com/Ascend/AgentSDK/blob/master/aura/configs/train/verl_train_async_A3_t16_qwen3_4b_math_fsdp.yaml) 模板，推理侧参照 [vllm_infer_i16_qwen3_4b.yaml](https://gitcode.com/Ascend/AgentSDK/blob/master/aura/configs/infer/vllm_infer_i16_qwen3_4b.yaml) 模板，业务方需修改以下参数：

训练配置：

- `hydra.searchpath`：同共卡模式
- `verl_conf.extras.data_loader.train_data_path`：分离模式使用megatron bin格式数据集路径，末尾需加 `/rl`
- `verl_conf.actor_rollout_ref.model.path`：Qwen3.6模型权重路径
- `train_instances.0.executor_kwargs.rollout_config.llm_tokenizer_path`：Qwen3.6 tokenizer路径
- `train_instances.0.executor_kwargs.work_mode`：固定为 `one_step_off`

推理配置：

- `infer_model_path`：Qwen3.6模型权重路径
- `infer_model_name`：模型名称，需与训练侧 `infer_instances.0.name` 一致
- `enable_expert_parallel`：Qwen3.6-27B/35B为Dense模型，配置为 `false`
- `tensor_parallel_size` / `data_parallel_size`：按卡数调整，相乘结果需等于推理总卡数

## 文档设计

编程手册将包含以下内容：

- Qwen3.6-27B和Qwen3.6-35B的配置文件说明
- 权重转换和加载步骤
- 常见问题排查指南

## 缺点和风险

1. **显存压力**：35B模型在单机训练下显存占用较高，可能需要多机分布式训练或开启重计算等，增加配置复杂度。

2. **算子兼容性**：Qwen3.6的部分算子实现与Qwen3存在差异，在NPU上的算子库可能需要补充适配，存在精度或性能风险。

3. **权重版本管理**：Qwen3.6官方权重可能存在多个版本（Base/Instruct等），权重映射规则需要明确支持的版本范围，避免误用。

## 应对措施

- 提供分规格的显存占用评估表和推荐硬件配置，指导业务方选择合适的部署方案
- 提供最佳实践配置，在该配置下训练效率和精度均能满足验收标准

## 现有技术

Qwen3.6系列适配借鉴了以下技术：

- **Qwen3系列适配方案**：提供了模型结构注册、权重映射的成熟实现，可直接复用大部分逻辑
- **框架现有的模型抽象层**：通过模型类型注册机制实现模型的无侵入式扩展
- **官方模型实现**：参考Qwen3.6官方仓库的网络结构定义和默认超参

与其他模型适配工作的差异：

- Qwen3.6在归一化层和位置编码上存在细节调整，需要针对性处理
- 27B和35B两个规格的默认超参差异较大，需要分别调优

## 验收标准

在Atlas A3系列产品上，Agent SDK训推调框架支持Qwen3.6系列模型（Qwen3.6-27B/35B）的端到端训练。
训练数据集 [MATH](https://www.modelscope.cn/datasets/AI-ModelScope/gsm8k) ，entropy、grad-norm有稳定收敛或下降趋势，reward曲线收敛稳定持平参考实现。
硬件：Atlas A2系列产品、Atlas A3系列产品    
OS：Ubuntu 22.04 LTS

## 其他补充说明

无
