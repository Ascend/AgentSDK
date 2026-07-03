# Aura

> [English](./OVERVIEW.md) | 中文

## 1.快速参考

- 从哪里获取帮助
    - [issue 反馈](https://gitcode.com/Ascend/AgentSDK/issues)
    - [AgentSDK 代码](https://gitcode.com/Ascend/AgentSDK/aura)
    - [Aura 快速启动文档](https://gitcode.com/Ascend/AgentSDK/blob/master/aura/docs/zh/quick_start.md)
    - [社区](https://www.hiascend.com/)

## 2.Aura简介

Aura(Agentic Ultra-fast Reinforcement Architecture) 是一个面向基础模型的训推调一体化框架，能够基于任务轨迹和奖励信号持续优化基础模型，通过强化学习等优化方法，使模型通过后训练，逐步具备规划、工具使用和长程决策等 Agent 化能力。

Aura 通过统一抽象接口兼容多种训练引擎、推理引擎与 Agent 框架，支持用户灵活接入自定义模型与工具链，帮助开发者快速构建、训练与部署自己的 Agent。

## 3.支持的 Tags 及 Dockerfile 链接

### 3.1 Tag 规范

Tag 遵循以下格式：

```bash
<AgentSDK版本>-<芯片系列>-<操作系统>-<python版本>
```

| 字段           | 示例值                             | 说明            |
|--------------|---------------------------------|---------------|
| `AgentSDK版本` | `26.1.0`                        | Agent SDK 版本号 |
| `芯片系列`       | `910`、`a3`、`atlas 800`          | 目标芯片系列        |
| `操作系统`       | `ubuntu22.04`, `openeuler24.03` | 基础操作系统        |
| `python版本`   | `py3.11`                        | Python 版本     |

### 3.2 CANN 9.0.0 + 26.1.0 Agent SDK镜像

| Tag                                 | Dockerfile                                                                              | 镜像内容                |
|-------------------------------------|-----------------------------------------------------------------------------------------|---------------------|
| `26.1.0-910b-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/AgentSDK/docker/aura/Dockerfile.910b.ubuntu)    | toolkit + Agent SDK |
| `26.1.0-a3-ubuntu22.04-py3.11`      | [Dockerfile](https://gitcode.com/Ascend/AgentSDK/docker/aura/Dockerfile.a3.ubuntu)      | toolkit + Agent SDK |
| `26.1.0-910b-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/AgentSDK/docker/aura/Dockerfile.910b.openeuler) | toolkit + Agent SDK |
| `26.1.0-a3-openeuler24.03-py3.11`   | [Dockerfile](https://gitcode.com/Ascend/AgentSDK/docker/aura/Dockerfile.a3.openeuler)   | toolkit + Agent SDK |

## 4.快速开始

### 4.1 前置要求

#### 4.1.1 安装驱动

- 主机上必须安装与容器内 CANN 版本兼容的NPU 驱动。请参阅 [CANN 兼容性矩阵](https://www.hiascend.com/document) 了解驱动与 CANN 版本的对应关系。
- docker版本要求：docker版本建议不低于24.0.x。

### 4.2 运行 Aura 容器

#### 手动挂载设备

- 设备挂载 ：通过 --device 参数将宿主机的设备文件映射到容器中，确保容器能够访问指定的硬件资源。/dev/davinci为NPU加速卡（按需挂载），/dev/davinci_manager, /dev/devmm_svm， /dev/hisi_hdc为NPU管理设备（全部挂载）。

- 驱动与工具链挂载 ：将宿主机上的驱动文件和工具链目录（如 /usr/local/Ascend/driver 和 /usr/local/bin/npu-smi）以只读方式挂载到容器中，保证容器内的运行环境与宿主机一致。 以下样例代码中，/dev/davinci1 表示挂载 1 号设备。

```bash
docker run --name your_container_name \
    --hostname agent \
    --network host \
    -it -d --shm-size=500g \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci8 --device=/dev/davinci9 \
    --device=/dev/davinci10 --device=/dev/davinci11 \
    --device=/dev/davinci12 --device=/dev/davinci13 \
    --device=/dev/davinci14 --device=/dev/davinci15 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -v /usr/local/sbin:/usr/local/sbin \
    your_image_name:your_image_tag  \
    sleep infinity
```

> [!NOTE] 说明
>
> 1. 根据 NPU 数量的不同，挂载不同数量的设备 ID。例如： Atlas A3 有 16 个 NPU，需挂载 16 个设备 ID，每个设备 ID 对应一个 NPU。
> 2. 镜像内默认工作目录为 /home/work，因此不建议挂载整个 /home 目录，以避免覆盖容器内默认工作空间或引发权限冲突。

### 4.3 快速启动用例

快速启动参考：[Qwen3-4B 共卡模式快速拉起指南](../../docs/aura/zh/models/qwen3-4b_quick_start/qwen3-4b-hybrid.md)

---

## 5. 支持的硬件

| 芯片系列      | 产品示例          | 架构             |
|-----------|---------------|----------------|
| Atlas 910 | Atlas 800I A2 | ARM64 / x86_64 |
| Atlas A3  | Atlas 800I A3 | ARM64 / x86_64 |

---

## 6. 许可证

查看这些镜像中包含的 CANN 和 Mind 系列软件的[许可证信息](https://github.com/Ascend/cann-container-image/blob/main/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
