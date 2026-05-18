# 1. 相关版本和依赖参考

| 依赖               | 版本      | 获取                                                                                                                                           |
|:-----------------|:--------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| NPU 驱动和固件        | 25.2.0  | 在[昇腾社区](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.2.RC1&driver=Ascend+HDK+25.2.0)选择对应版本和型号的run包 |
| CANN 算子包         | 8.2.RC1 |                                                                                                                                              |
| python           | 3.10    |                                                                                                                                              |
| torch/torch_npu  | 2.5.1   | 到[昇腾社区](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)选择对应的版本进行下载                              |
| apex             | 0.1     | 参考[安装APEX模块](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0009.html)                                    |
| ray              | 2.42.1  |                                                                                                                                              |
| vllm/vllm-ascend | 0.9.1   |                                                                                                                                              |
| mindspeed-rl     | 2.1.0   |                                                                                                                                              |

参考：
[MindSpeed 安装指南](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/zh/install_guide.md)

# 2. 镜像构建说明

## 2.1 准备相关依赖包

在当前目录下：

1. 创建cann目录并将cann算子包（kernels/toolkit/nnal）拷贝到cann目录下
2. 创建miniconda目录并将miniconda安装shell脚本拷贝到miniconda目录
3. 创建PTA目录，并将torch/torch-npu的whl包拷贝到PTA目录

## 2.2 运行

准备好相关安装包后，直接运行build_docker_image.sh

```bash
bash build_docker_image
```

# 3. 构建流程说明

## 3.1 Dockerfile镜像构建

**具体操作请参考Dockerfile文件**

- 写入版本匹配的yun源并安装`"Development Tools"`等必要组件
- 构建miniconda3的python环境
- 提前准备相关cann包并执行安装
    - cann包可直接到[昇腾社区]搜索下载
- 提前准备torch/torch_npu whl包并执行安装
    - whl包可以到[昇腾社区](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)
      选择对应的版本进行下载
- 拷贝相关脚本进入容器：1）init.sh 用于在docker run时写入DNS和环境变量到bashrc文件；2）vllm/mindspeed 安装脚本

## 3.2 vllm/MindSpeed 安装

vllm/MindSpeed的安装需要在容器交互模式下执行，无法通过Dockerfile一步构建完成，所以在Dockerfile构建的初版镜像后，需要进入到容器内再执行相关安装脚本

**具体参考 [install_vllm_mindspeed.sh](build_docker_image.sh)**

1. 配置代理：会提示输入工号和密码
2. 在华为镜像源下安装vllm/vllm-ascend
3. 在国内其他镜像源下安装MindSpeed相关依赖（推荐阿里源，注意是否存在中途下载速度慢退出的问题）
4. 安装apex，参考[安装APEX模块](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0009.html)
5. 取消代理，删除tmp相关文件
