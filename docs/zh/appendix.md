# 附录<a name="ZH-CN_TOPIC_0000002459355044"></a>

## 软件中包含的公网地址<a name="ZH-CN_TOPIC_0000002459355068"></a>

Agent SDK的安装包中的网址安装结束后会被清除，并不会访问，不会造成风险。

Agent SDK本手册中存在的公开网址和邮箱地址，SDK本身不会访问，不会造成风险。

更多公网地址请参考[AgentSDK 公网地址.xlsx](./resource/AgentSDK_公网地址_0000002516443057.xlsx)。

## 环境变量使用<a name="ZH-CN_TOPIC_0000002504002753"></a>

Agent SDK在运行过程中可能会使用到以下环境变量。

|环境变量名称|描述|
|--|--|
|LOCAL_RANK|torch分布式训练设置，用来描述当前线程在当前节点上的rank信息，取值范围为[0, 8)。|
|RANK|torch分布式训练设置，用来描述当前线程在所有节点上的rank信息，取值范围为[0, 8)。|

Agent SDK在启动时会使用白名单校验环境变量，只有以下环境变量将会被保留。

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
|TOOLCHAIN_HOME|tookit工具链安装路径。|

> [!NOTE] 说明
>
>- Agent SDK的运行会使用到开源软件，相关开源软件会使用的环境变量请参考对应软件说明。
>- Agent SDK依赖CANN，运行CANN的过程中，会生成kernel\_meta等文件夹，Agent SDK不具有转储和删除这些文件的功能，用户可参考《CANN 环境变量参考》中的“安装配置相关”\>“落盘文件配置”\>“ASCEND\_WORK\_PATH”章节使用环境变量进行文件统一管理**。**
