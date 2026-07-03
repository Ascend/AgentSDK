# 安全声明<a name="ZH-CN_TOPIC_0000002459514612"></a>

## 公网地址声明

Agent SDK 服务启动中存在的[公网地址](resource/AgentSDK_公网地址_0000002516443057.xlsx)并不会访问，不会造成风险。

## 文件权限控制

使用 API 读取文件时，用户需要保证该文件的 owner 必须为自己，且权限不高于 640，避免发生提权等安全问题。
外部下载的软件代码或程序可能存在风险，功能的安全性需由用户保证。

## 通信矩阵

目前 Agent SDK 提供分布式训练能力，支持在单机和多机场景下进行训练，需要进行网络通信。其中PyTorch需要使用TCP进行通信，
torch_npu使用CANN中HCCL在NPU设备间通信，通信端口见[AgentSDK 2026.1.0 通信矩阵](resource/AgentSDK%2026.1.0%20通信矩阵.xlsx)信息。
用户需要注意并保障节点间通信网络安全，可以使用iptables等方式消减安全风险， 可参考[通信安全加固](#通信安全加固)进行网络安全加固。

## 通信安全加固

Agent SDK 分布式训练服务需要在设备间进行通信，通信开启的端口默认为全0侦听，为了降低安全风险，建议用户针对此场景进行安全加固，
如使用iptables配置防火墙，在运行分布式训练开始前限制外部对分布式训练使用端口的访问，在运行分布式训练结束后清理防火墙规则。

1. 防火墙规则设定和移除参考脚本模板
    - 防火墙规则设定，可参考如下脚本：

    ```bash
    #!/bin/bash
    set -x

    # 要限制的端口号
    port={端口号}

    # 清除旧规则
    iptables -D INPUT -p tcp -j {规则名}
    iptables -F {规则名}
    iptables -X {规则名}

    # 创建新的规则链
    iptables -t filter -N {规则名}

    # 在多机场景下设定白名单，允许其他节点访问主节点的侦听端口
    # 在规则链中添加允许特定IP地址范围的规则
    iptables -t filter -A {规则名} -i eth0 -p tcp --dport $port -s {允许外部访问的IP} -j ACCEPT

    # 屏蔽外部地址访问分布式训练端口
    # 在PORT-LIMIT-RULE规则链中添加拒绝其他IP地址的规则
    iptables -t filter -A {规则名} -i {要限制的网卡名} -p tcp --dport $port -j DROP

    # 将流量传递给规则链
    iptables -I INPUT -p tcp -j {规则名}
    ```

    - 防火墙规则移除，可参考如下脚本：

    ```bash
    #!/bin/bash
    set -x
    # 清除规则
    iptables -D INPUT -p tcp -j {规则名}
    iptables -F {规则名}
    iptables -X {规则名}
    ```

2. 防火墙规则设定和移除参考脚本示例
    1. 针对特定端口设定防火墙，脚本中端口号为要限制的端口，在 Agent SDK 分布式训练中端口号请参考[通信矩阵信息](resource/AgentSDK%2026.1.0%20通信矩阵.xlsx)；
   要限制的网卡名为服务器用于分布式通信使用的网卡，允许的外部访问的IP为分布式训练服务器的IP地址。网卡和服务器IP可以通过ifconfig查看，
   如下文回显的eth0为网卡名，192.168.1.1为服务器IP地址：

        ```bash
        # ifconfig
        eth0
            inet addr:192.168.1.1 Bcast:192.168.1.255 Mask:255.255.255.0
            inet6 addr: fe80::230:64ee:ef1a:c1a/64 Scope:Link
        ```

    2. 假定服务器主节点地址192.168.1.1，另一台需要进行分布式训练的服务器为192.168.1.2，训练端口为4002。
        - 防火墙规则设定，可参考如下脚本：

        ```bash
        #!/bin/bash
        set -x

        # 设定侦听的端口
        port=4002

        # 清除旧规则
        iptables -D INPUT -p tcp -j PORT-LIMIT-RULE
        iptables -F PORT-LIMIT-RULE
        iptables -X PORT-LIMIT-RULE

        # 创建新的PORT-LIMIT-RULE规则链
        iptables -t filter -N PORT-LIMIT-RULE

        # 在多机场景下设定白名单，允许192.168.1.2访问主节点
        # 在PORT-LIMIT-RULE规则链中添加允许特定IP地址范围的规则
        iptables -t filter -A PORT-LIMIT-RULE -i eth0 -p tcp --dport $port -s 192.168.1.2 -j ACCEPT

        # 屏蔽外部地址访问分布式训练端口
        # 在PORT-LIMIT-RULE规则链中添加拒绝其他IP地址的规则
        iptables -t filter -A PORT-LIMIT-RULE -i eth0 -p tcp --dport $port -j DROP

        # 将流量传递给PORT-LIMIT-RULE规则链
        iptables -I INPUT -p tcp -j PORT-LIMIT-RULE
        ```

        - 防火墙规则移除，可参考如下脚本：

        ```bash
        #!/bin/bash
        set -x
        # 清除规则
        iptables -D INPUT -p tcp -j PORT-LIMIT-RULE
        iptables -F PORT-LIMIT-RULE
        iptables -X PORT-LIMIT-RULE
        ```
