# FAQ<a name="ZH-CN_TOPIC_0000002503313713"></a>

## 提示网络不可用<a name="ZH-CN_TOPIC_0000002470393856"></a>

**问题现象<a name="section108091832161719"></a>**

启动Agent SDK后，出现如下报错信息。

```text
...
socket.gaierror: [Errno -3] Temporary failure in name resolution
...
```

**原因分析<a name="section19494183319186"></a>**

获取本地ip是通过socket.gethostbyname(socket.gethostname())获取的，在容器中运行时，需要修改hostname为容器hostname。

**解决方案<a name="section137992561914"></a>**

查看hostname，并通过设置/etc/hosts解决问题。

```shell
# 查看hostname
hostname
# 修改/etc/hosts
echo "127.0.0.1 $(hostname)" >> /etc/hosts
```

## 提示换行符不识别<a name="ZH-CN_TOPIC_0000002470393856"></a>

**问题现象<a name="section108091832161719"></a>**

启动Agent SDK后，出现如下报错信息。

```text
...
$'\r': command not found
...
```

**原因分析<a name="section19494183319186"></a>**

shell脚本在不同操作系统间的换行符可能不同，导致解析错误。

**解决方案<a name="section137992561914"></a>**

将所有shell脚本和配置文件转换为Unix格式。

```shell
# 转换所有shell脚本为Unix格式
find /path/to/AgentSDK -type f -name "*.sh" -exec dos2unix {} +
```

## Tensor 尺寸不一致<a name="ZH-CN_TOPIC_0000002470393856"></a>

**问题现象<a name="section108091832161719"></a>**

启动 Agent SDK ，训练一定步数后，出现如下报错信息：

```text
RuntimeError: The size of tensor a(4096) must match the size of tensor b (3747) at non-singleton dimension 1.
```

**原因分析<a name="section19494183319186"></a>**

未对 verl 应用补丁（patch），导致张量尺寸计算错误。

**解决方案<a name="section137992561914"></a>**

下载并应用 rllm 提供的 verl 补丁文件：

```shell
# 下载补丁文件
wget -P /verl https://raw.githubusercontent.com/rllm-org/rllm/b5b9760fc3d7208d368f21dcd3e12f4f7eddfdc7/rllm/experimental/fully_async/verl_dp_actor.patch

# 应用补丁
cd /verl
patch -p1 < /verl/verl_dp_actor.patch
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
