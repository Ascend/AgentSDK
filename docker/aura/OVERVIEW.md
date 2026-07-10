# Agent SDK

> English | [中文](./OVERVIEW.zh.md)

## 1.Quick Reference

- Where to get help

    - [Issue Feedback](https://gitcode.com/Ascend/AgentSDK/issues)
    - [AgentSDK Code](https://gitcode.com/Ascend/AgentSDK/tree/master/aura)
    - [Aura Quick-start Documentation](../../docs/zh/aura/03_quick_start.md)
    - [Community](https://www.hiascend.com/en/)

## 2.Aura Overview

Aura (Agentic Ultra-fast Reinforcement Architecture) is an integrated training-inference-tuning framework for foundation models. It continuously improves foundation models based on task trajectories and reward signals, and through reinforcement learning and other optimization methods, enables the model—via post-training—to progressively acquire agent-like capabilities such as planning, tool use, and long-horizon decision-making.

Through a unified abstraction interface, Aura is compatible with multiple training engines, inference engines, and agent frameworks. It supports flexible integration of custom models and toolchains, helping developers quickly build, train, and deploy their own agents.

## 3.Supported Tags and Dockerfile Links

### 3.1 Tag Naming Convention

Tags follow this pattern:

```bash
<AgentSDK_version>-<CANN_version>-<pytorch_version>-<chip_series>-<os>-<python_version>
```

| Field              | Example Values                  | Description           |
|--------------------|---------------------------------|-----------------------|
| `AgentSDK_version` | `26.1.0`                        | AgentSDK version      |
| `chip_series`      | `910`, `a3`, `atlas 800`        | Target chip family    |
| `os`               | `ubuntu22.04`, `openeuler24.03` | Base operating system |
| `python_version`   | `py3.11`                        | Python version        |

### 3.2 CANN 9.0.0 + 26.1.0 Agent SDK Image

| Tag                                 | Dockerfile                                                                              | 镜像内容                |
|-------------------------------------|-----------------------------------------------------------------------------------------|---------------------|
| `26.1.0-910b-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/AgentSDK/docker/aura/Dockerfile.910b.ubuntu)    | toolkit + Agent SDK |
| `26.1.0-a3-ubuntu22.04-py3.11`      | [Dockerfile](https://gitcode.com/Ascend/AgentSDK/docker/aura/Dockerfile.a3.ubuntu)      | toolkit + Agent SDK |
| `26.1.0-910b-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/AgentSDK/docker/aura/Dockerfile.910b.openeuler) | toolkit + Agent SDK |
| `26.1.0-a3-openeuler24.03-py3.11`   | [Dockerfile](https://gitcode.com/Ascend/AgentSDK/docker/aura/Dockerfile.a3.openeuler)   | toolkit + Agent SDK |

## 4.Quick Start

### 4.1 Prerequisites

#### 4.1.1 Install Driver

- An Atlas NPU driver compatible with the container's CANN version must be installed on the host. See the [CANN Compatibility Matrix](https://www.hiascend.com/document) for driver ↔ CANN version mapping.
- Docker version requirement: Docker version should not be lower than 24.0.x.

### 4.2 Running a Aura Container

#### Mount Devices Manually

- Device Mounting: Map host device files to the container using the --device parameter to ensure the container can access specified hardware resources. /dev/davinci is the NPU accelerator card (mount as needed), while /dev/davinci_manager, /dev/devmm_svm, and /dev/hisi_hdc are NPU management devices (mount all).

- Driver and Toolchain Mounting: Mount driver files and toolchain directories (such as /usr/local/Ascend/driver and /usr/local/bin/npu-smi) from the host to the container in read-only mode to ensure consistent runtime environment. In the example code below, /dev/davinci1 represents mounting device 1.

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

> [!NOTE] NOTE
>
> 1. Depending on the number of NPUs, mount a different number of device IDs. For example, Atlas A3 has 16 NPUs, so 16 device IDs need to be mounted, with each device ID corresponding to one NPU.
> 2. The default working directory inside the container is /home/work. Therefore, it is not recommended to mount the entire /home directory, as this may overwrite the default workspace inside the container or cause permission conflicts.

### 4.3 Quick-start Demo

Quick start reference: [Qwen3-4B hybrid mode quick start](../../docs/zh/aura/models/qwen3-4b_quick_start/qwen3-4b-hybrid.md)

---

## 5. Supported Hardware

| Chip Series | Product Examples | Architecture   |
|-------------|------------------|----------------|
| Atlas 910   | Atlas 800I A2    | ARM64 / x86_64 |
| Atlas A3    | Atlas 800I A3    | ARM64 / x86_64 |

---

## 6. License

View the [license information](https://github.com/Ascend/cann-container-image/blob/main/LICENSE) for CANN and MindSeries software included in these images.

As with all container images, the pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
