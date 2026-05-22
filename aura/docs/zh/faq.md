# FAQ<a name="ZH-CN_TOPIC_0000002503313713"></a>

## 提示网络不可用<a name="ZH-CN_TOPIC_0000002470393856"></a>

**问题现象<a name="section108091832161719"></a>**

启动Agent SDK后，出现如下报错信息。

```text
eth0: error fetching interface information: Device not found
[ERROR] [ 26-05-21 10:47:56 ] can not get IP from eth0
```

**原因分析<a name="section19494183319186"></a>**

系统默认通过环境变量 DEFAULT_SOCKET_IFNAME（默认值为 eth0）来获取本地IP。当前报错是因为在 ifconfig 中无法找到名为 eth0 的虚拟网桥，导致无法解析出正确的本地IP地址。

**解决方案<a name="section137992561914"></a>**

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

## 提示换行符不识别<a name="ZH-CN_TOPIC_0000002470393856"></a>

**问题现象<a name="section108091832161719"></a>**

启动Agent SDK后，出现如下报错信息。

```text
$'\r': command not found
```

**原因分析<a name="section19494183319186"></a>**

shell脚本在不同操作系统间的换行符可能不同，导致解析错误。

**解决方案<a name="section137992561914"></a>**

将所有shell脚本和配置文件转换为Unix格式。

```shell
# 转换所有shell脚本为Unix格式
find /path/to/AgentSDK -type f -name "*.sh" -exec dos2unix {} +
```

## IP地址或端口已被绑定<a name="ZH-CN_TOPIC_0000002470393856"></a>

**问题现象<a name="section108091832161719"></a>**

启动 Agent SDK后，出现如下报错信息：

```text
RuntimeError: createHCCLCommOrigin:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:2314 HCCL function error: HcclGetRootInfo(&hcclID)，error code is 7
ERR02200 DIST call hccl api failed
Failed to bind the IP port. Reason: The IP address and port have been bound already.
```

**原因分析<a name="section19494183319186"></a>**

HCCL端口被其他进程占用，导致出现IP地址和端口已经被绑定的HCCL错误。

**解决方案<a name="section137992561914"></a>**

设置环境变量，更换HCCL端口：

```shell
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60100
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
```

## 提示uid不一致<a name="ZH-CN_TOPIC_0000002470393856"></a>

**问题现象<a name="section108091832161719"></a>**

共卡模式，完成训练和推理时，在最后一步验证阶段出现如下报错信息：

```text
AssertionError: 'uid' in tensor_dict1 and tensor_dict2 are not same object..
```

**原因分析<a name="section19494183319186"></a>**

yaml文件中参数test_freq参数设置不正确

**解决方案<a name="section137992561914"></a>**

将yaml文件中参数test_freq设置为-1，即可禁用验证阶段。
