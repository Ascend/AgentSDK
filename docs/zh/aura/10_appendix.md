# 附录

## 环境变量使用

Aura 在运行过程中可能会使用到以下环境变量。

|环境变量名称|描述|
|--|--|
|LOCAL_RANK|torch分布式训练设置，用来描述当前线程在当前节点上的rank信息，取值范围为[0, 8)。|
|RANK|torch分布式训练设置，用来描述当前线程在所有节点上的rank信息，取值范围为[0, 8)。|

Aura 在启动时会使用白名单校验环境变量，只有以下环境变量将会被保留。

|环境变量名称|描述|
|--|--|
|ASCEND_WORK_PATH|归一CANN运行中过程中生成文件的位置。|
|ASCEND_AICPU_PATH|ascend-toolkit的AI CPU的安装路径。|
|ASCEND_HOME_PATH|同ASCEND_TOOLKIT_HOME，代表CANN-toolkit软件安装后文件存储路径。|
|ASCEND_OPP_PATH|算子库根目录。|
|ASCEND_TOOLKIT_HOME|CANN-toolkit软件包安装后文件存储路径。|
|ASDOPS_LOG_LEVEL|算子库日志级别。|
|ASDOPS_LOG_PATH|算子库日志保存路径。|
|ASDOPS_LOG_TO_BOOST_TYPE|加速库日志目录名称。|
|ASDOPS_LOG_TO_FILE|算子库日志是否输出到文件。|
|ASDOPS_LOG_TO_FILE_FLUSH|日志写文件是否刷新。|
|ASDOPS_LOG_TO_STDOUT|算子库日志是否输出到控制台。|
|ATB_COMPARE_TILING_EVERY_KERNEL|每个Kernel运行后，比较运行前和后的NPU上tiling内容是否变化，一般用于检查是否发生tiling内存踩踏。|
|ATB_DEVICE_TILING_BUFFER_BLOCK_NUM|Context内部DeviceTilingBuffer块数，数量与OP并行的最大并行数有关，通常使用默认值，不建议修改。|
|ATB_HOME_PATH|nnal软件包安装后文件存储路径。|
|ATB_HOST_TILING_BUFFER_BLOCK_NUM|Context内部HostTilingBuffer块数，数量与OP并行的最大并行数有关，通常使用默认值，不建议修改。|
|ATB_MATMUL_SHUFFLE_K_ENABLE|Shuffle-K使能，矩阵乘的结果矩阵不同位置计算时的累加序一致/不一致。会影响matmul算子内部累加序。|
|ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT|全局kernelCache的槽位数。槽位数增加：<li>增加cache命中率，但降低检索效率。</li><li>槽位数减少：提高检索效率，但降低cache命中率。</li>|
|ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT|本地kernelCache的槽位数。<li>槽位数增加时：增加cache命中率，但降低检索效率。</li><li>槽位数减少时：提高检索效率，但降低cache命中率。</li>|
|ATB_OPSRUNNER_SETUP_CACHE_ENABLE|是否开启ATB的SetupCache功能。该功能在检测到operation的输入和输出tensor未发生变化时会跳过setup的大部分流程，进而提升调度侧性能。默认开启，以进行性能加速。|
|ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE|用于问题定位，确定报错所在的kernel。当变量配置为1时，每个Kernel的Execute结束时就做流同步。|
|ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE|用于问题定位，确定报错所在的Operation。当变量配置为1时，每个Operation的Execute时就做同步。|
|ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE|用于问题定位，确定报错所在的runner。当变量配置为1时，每个Runner的Execute时就做流同步。|
|ATB_SHARE_MEMORY_NAME_SUFFIX|共享内存命名后缀，多用户同时使用通信算子时，需通过设置该值进行共享内存的区分。|
|ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE|workspace内存分配算法选择。根据环境变量配置不同，ATB会选择不同的算法去计算workspace大小与workspace分配，用户可通过选择不同算法自行测试workspace分配情况。|
|ATB_WORKSPACE_MEM_ALLOC_GLOBAL|是否使用全局中间tensor内存分配算法。开启后会对中间tensor内存进行大小计算与分配。|
|HOME|当前用户的主目录路径|
|LCCL_DETERMINISTIC|LCCL确定性AllReduce（保序加）是否开启。需注意，开启功能在rankSize<=8时生效。开启后会有如下影响：<li>影响部分通信算子性能。</li><li>影响lccl通信算子的累加序。</li>|
|LD_LIBRARY_PATH|动态链接库搜索路径（Linux 专用）。|
|PATH|可执行文件搜索路径。|
|PYTHONPATH|Python 模块搜索路径。|
|TOOLCHAIN_HOME|toolkit工具链安装路径。|

> [!NOTE] 说明
>
>- Aura 的运行会使用到开源软件，相关开源软件会使用的环境变量请参考对应软件说明。
>- Aura 依赖CANN，运行CANN的过程中，会生成kernel\_meta等文件夹，Aura 不具有转储和删除这些文件的功能，用户可参考《CANN 环境变量参考》中的"安装配置相关" \> "落盘文件配置" \> "ASCEND\_WORK\_PATH"章节，使用环境变量进行文件统一管理。

## 支持的推理后端

Aura 支持以下推理后端：

|推理后端|描述|
|--|--|
|vllm-ascend|基于vLLM框架的昇腾NPU适配版本，提供高性能的大模型推理能力。|

## 支持的训练后端

Aura 支持以下训练后端：

|训练后端|描述|
|--|--|
|verl|可验证的强化学习训练框架，支持PPO、GRPO等强化学习算法。|

## 支持的Agent后端

Aura 支持以下Agent后端：

|Agent后端|描述|
|--|--|
|rLLM|基于强化学习的大语言模型Agent框架，支持工具调用和多轮对话。|

## 支持的模型列表

Aura 支持以下模型：

|模型名称|参数规模|描述|
|--|--|--|
|Qwen3-4B|4B|Qwen3系列轻量模型，适合资源受限场景的快速训练和推理。|
|Qwen3-8B|8B|Qwen3系列中等规模模型，平衡性能与资源消耗。|
|Qwen3-14B|14B|Qwen3系列中大型模型，具备强大的复杂推理与数学逻辑能力。|
|Qwen3-32B|32B|Qwen3系列通用大模型，具有强大的语言理解和生成能力。|
|Qwen3-30B-A3B|30B|Qwen3系列MoE模型，兼顾大模型性能与推理效率。|
