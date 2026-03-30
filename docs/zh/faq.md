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

可能是由于mindspeed-rl在解析节点IP时出现错误。

**解决方案<a name="section137992561914"></a>**

查看hostname，并通过设置/etc/hosts解决问题。

```shell
# 查看hostname
hostname
# 修改/etc/hosts
# 在127.0.0.1 localhost后面增加当前设备的hostname即可
```
