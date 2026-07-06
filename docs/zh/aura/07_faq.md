# FAQ<a name="faq_main"></a>

## 提示网络不可用<a name="faq_001"></a>

**问题现象<a name="faq_001_phenomenon"></a>**

启动 Aura 后，出现如下报错信息。

```text
eth0: error fetching interface information: Device not found
[ERROR] [ 26-05-21 10:47:56 ] can not get IP from eth0
```

**原因分析<a name="faq_001_analysis"></a>**

系统默认通过环境变量 DEFAULT_SOCKET_IFNAME（默认值为 eth0）来获取本地IP。当前报错是因为在 ifconfig 中无法找到名为 eth0 的虚拟网桥，导致无法解析出正确的本地IP地址。

**解决方案<a name="faq_001_solution"></a>**

请检查您容器内的网络配置，并将环境变量 DEFAULT_SOCKET_IFNAME 的值修改为当前环境中实际存在的、且包含正确本地IP的虚拟网桥名称（例如 br0 或其他自定义网桥名）。

具体示例如下：

1. 执行 ifconfig 命令，查看网络配置：

    ```shell
    ifconfig
    ```

2. 假设得到打印信息（部分）为：

    ```text
    docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255

    enp189s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 192.168.0.1  netmask 255.255.0.0  broadcast 192.168.255.255

    enp189s0f1: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
            inet 192.168.100.100  netmask 255.255.255.0  broadcast 192.168.100.255
    ```

3. 假设本地IP为 192.168.0.1，那么指向本地IP对应虚拟网桥的值即为 enp189s0f0 ，即需要执行：

    ```shell
    export DEFAULT_SOCKET_IFNAME=enp189s0f0
    ```

## 提示换行符不识别<a name="faq_002"></a>

**问题现象<a name="faq_002_phenomenon"></a>**

启动 Aura 后，出现如下报错信息。

```text
$'\r': command not found
```

**原因分析<a name="faq_002_analysis"></a>**

shell脚本在不同操作系统间的换行符可能不同，导致解析错误。

**解决方案<a name="faq_002_solution"></a>**

将所有shell脚本和配置文件转换为Unix格式。

```shell
# 转换所有shell脚本为Unix格式
find /path/to/AgentSDK -type f -name "*.sh" -exec dos2unix {} +
```

## IP地址或端口已被绑定<a name="faq_003"></a>

**问题现象<a name="faq_003_phenomenon"></a>**

启动 Aura 后，出现如下报错信息：

```text
RuntimeError: createHCCLCommOrigin:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:2314 HCCL function error: HcclGetRootInfo(&hcclID)，error code is 7
ERR02200 DIST call hccl api failed
Failed to bind the IP port. Reason: The IP address and port have been bound already.
```

**原因分析<a name="faq_003_analysis"></a>**

HCCL端口被其他进程占用，导致出现IP地址和端口已经被绑定的HCCL错误。

**解决方案<a name="faq_003_solution"></a>**

设置环境变量，更换HCCL端口：

```shell
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60100
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
```

## 提示uid不一致<a name="faq_004"></a>

**问题现象<a name="faq_004_phenomenon"></a>**

共卡模式，完成训练和推理时，在最后一步验证阶段出现如下报错信息：

```text
AssertionError: 'uid' in tensor_dict1 and tensor_dict2 are not same object..
```

**原因分析<a name="faq_004_analysis"></a>**

yaml文件中参数test_freq参数设置不正确

**解决方案<a name="faq_004_solution"></a>**

将yaml文件中参数test_freq设置为-1，即可禁用验证阶段。

## 提示运行triton时，utils文件部分字段找不到<a name="faq_005"></a>

**问题现象<a name="faq_005_phenomenon"></a>**

启动脚本后，triton/backends/ascend/utils.py报错找不到RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE：

```text
/tmp/tmpaxgs3vq5/npu_utils.cpp:324:3: error: could not convert '{{"LOW_POWER_TIMEOUT", RT_LIMIT_TYPE_LOW_POWER_TIMEOUT}, {"WARP_STACK_SIZE", <expression error>}, {"DVG_WARP_STACK_SIZE", RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE}, {"STACK_SIZE", RT_LIMIT_TYPE_STACK_SIZE}}' from '<brace-enclosed initializer list>' to 'const std::unordered_map<std::__cxx11::basic_string<char>, tagRtLimitType>' [repeated 3x across cluster]
```

**原因分析<a name="faq_005_analysis"></a>**

CANN8.5.1存在该文件，而CANN9.0.0已去除该文件，可通过下面的指令搜索该文件的位置。

```text
grep -R RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE /usr/local/Ascend/ascend-toolkit/latest/include/experiment/runtime/runtime
```

triton-ascend 3.2.0 存在该问题，可以打patch(../third_party/patch/triton-ascend.patch)进行修复，triton-ascend对应PR：<https://gitcode.com/Ascend/triton-ascend/pull/1525/diffs>

**解决方案<a name="faq_005_solution"></a>**

对triton-ascend 3.2.0该部分进行补丁修改。

## 提示缺少模块pkg_resources<a name="faq_006"></a>

**问题现象<a name="faq_006_phenomenon"></a>**

启动脚本后，出现如下报错信息：

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

**原因分析<a name="faq_006_analysis"></a>**

setuptools版本比较新，新版本去除了该模块。

**解决方案<a name="faq_006_solution"></a>**

安装setuptools 80.10.2版本。

## 提示配置文件多了strict字段<a name="faq_007"></a>

**问题现象<a name="faq_007_phenomenon"></a>**

启动脚本后，出现如下报错信息：

```text
TypeError("CheckpointConfig.__init__() got an unexpected keyword argument 'strict'")
```

**原因分析<a name="faq_007_analysis"></a>**

配置文件会读取根目录下的verl，根目录下的verl版本不对，建议切换到指定commit版本。

**解决方案<a name="faq_007_solution"></a>**

将根目录下的verl版本切换到指定commit版本。

```text
git clone https://github.com/verl-project/verl.git
git checkout e9972368aa6a6078eacd7f0678bdfdd0196ce7b5
```

## 提示vllm缺少fused_moe.runner模块<a name="faq_008"></a>

**问题现象<a name="faq_008_phenomenon"></a>**

启动脚本后，出现如下报错信息：

```text
ModuleNotFoundError: No module named 'vllm.model_executor.layers.fused_moe.runner'
```

**原因分析<a name="faq_008_analysis"></a>**

vllm 0.16.0rc1版本没有这个runner目录，一直到0.16.1rc0版本才有，建议直接切换到指定commit版本。

vllm和vllm-ascend版本对应链接可参考：https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html

**解决方案<a name="faq_008_solution"></a>**

将vllm切换到对应版本。

```text
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 4034c3d32
VLLM_TARGET_DEVICE=empty pip install -v -e.

git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout fe4cad24e
export COMPILE_CUSTOM_KERNELS=1 && pip install -v -e .
```

## 提示缺少libtorch_cuda.so文件<a name="faq_009"></a>

**问题现象<a name="faq_009_phenomenon"></a>**

启动脚本后，出现如下报错信息：

```text
OSError: libtorch_cuda.so: cannot open shared object file: No such file or directory.
```

**原因分析<a name="faq_009_analysis"></a>**

调用了cuda版的torch，在昇腾平台上，要么调用torch_npu，要么调用cpu版torch，所以需要卸载torch，安装cpu版torch。

**解决方案<a name="faq_009_solution"></a>**

安装cpu版的torch。

```text
pip uninstall -y torch torchvision torchaudio
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
```

## 提示triton缺少language.target_info模块<a name="faq_010"></a>

**问题现象<a name="faq_010_phenomenon"></a>**

启动脚本后，出现如下报错信息：

```text
Failed to import Triton kernels. Please make sure your triton version is compatible. Error: No module named 'triton.language.target_info'
```

**原因分析<a name="faq_010_analysis"></a>**

这个是vllm在判断当前环境是不是cuda环境，如果不是cuda环境就会抛一个ERROR，然后进入ascend-vllm的流程。

**解决方案<a name="faq_010_solution"></a>**

vllm内部代码逻辑，无需解决，仅为打印提示，不会中断主流程，建议排查其他错误原因。

## 编译安装vllm-ascend时，提示缺少文件moe_distribute_comm_ctx.h<a name="faq_011"></a>

**问题现象<a name="faq_011_phenomenon"></a>**

启动脚本后，出现如下报错信息：

```text
fatal error: 'moe_distribute_comm_ctx.h' file not found
```

**原因分析<a name="faq_011_analysis"></a>**

vllm-ascend对于a3服务器，在cmake编译算子时，缺少头文件路径，需补充环境变量，从而将该文件路径添加到CMakeList的文件路径中。

**解决方案<a name="faq_011_solution"></a>**

添加环境变量，添加cmake编译算子时包含的头文件路径：

```text
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/inc/kernel:${CPLUS_INCLUDE_PATH}
```

## 提示http连接失败<a name="faq_012"></a>

**问题现象<a name="faq_012_phenomenon"></a>**

启动脚本后，出现如下报错信息：

```text
httpcore.ConnectError: All connection attempts failed
```

**原因分析<a name="faq_012_analysis"></a>**

环境中存在http代理，导致InferExecutor 调用vLLM接口时网络连接失败。

**解决方案<a name="faq_012_solution"></a>**

通过 `export` 指令查看所有环境变量，将代理相关的环境变量 `unset` 掉。

```text
unset http_proxy
unset https_proxy
```

## 提示未配置host index<a name="faq_013"></a>

**问题现象<a name="faq_013_phenomenon"></a>**

启动脚本后，出现如下报错信息：

```text
You should exec "export VC_TASK_INDEX={0|1|2...}" to configure the host index
```

**原因分析<a name="faq_013_analysis"></a>**

未正确设置hosts.conf。

**解决方案<a name="faq_013_solution"></a>**

根据单机或多机的IP地址，正确配置hosts.conf：

```shell
# host,index,train_master_index,infer_master_index(可选)
# 如果单机训练+推理共部署, 则需要配置infer_master_index

# [单机训练+推理]
# 配置例子1：单机，训推共节点部署, 方便本地调测
# host,index,train_master_index,infer_master_index(可选)
# 192.168.0.1,0,1,1

# [多机训练+推理]
# 配置例子2：双机, 训推分离, 分节点部署
# host,index,train_master_index,infer_master_index(可选)
192.168.0.1,0,0
192.168.0.2,1,1
```

## 提示ray启动失败<a name="faq_014"></a>

**问题现象<a name="faq_014_phenomenon"></a>**

启动脚本后，出现如下报错信息：

```text
ConnectionError: Could not find any running Ray instance. Please specify the one to connect to by setting the --address flag or RAY_ADDRESS environment variable.
```

**原因分析<a name="faq_014_analysis"></a>**

环境问题导致ray启动失败。

**解决方案<a name="faq_014_solution"></a>**

尝试手动起ray，判断是否是环境存在问题，执行下面的指令前需保证指令使用的端口没有被占用：

```shell
ray start --head --port=7894 --dashboard-port=7895
```

如果ray存在报错，需排查环境问题，下面提供一些排查方向：

1. 保证ray使用的端口没有被占用
2. 保证有足够的文件描述符供raylet进程使用
3. 机器上其他容器是否有ray进程正在运行
