ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-ubuntu22.04-py3.11

FROM ${BASE_IMAGE}
ARG SOC_VERSION=ascend910_9391

ENV SOC_VERSION=${SOC_VERSION}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y net-tools dos2unix ca-certificates curl wget && \
    update-ca-certificates || true

RUN mkdir -p /home/work

WORKDIR /home/work

RUN git clone https://gitcode.com/Ascend/AgentSDK.git && \
    rm -rf /home/work/AgentSDK/aura/tests && \
    bash /home/work/AgentSDK/aura/dockers/env/build_repos.sh

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip config set global.trusted-host mirrors.aliyun.com

RUN cd /home/work/vllm-ascend && \
    git checkout fe4cad24e && \
    export COMPILE_CUSTOM_KERNELS=1 && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    pip install -v -e .

RUN pip install -e /home/work/MindSpeed && \
    pip install -e /home/work/Megatron-LM && \
    pip uninstall -y triton && \
    pip install mbridge

RUN cd /verl && \
    git checkout e9972368aa6a6078eacd7f0678bdfdd0196ce7b5 && \
    pip install -r requirements-npu.txt && \
    pip install -v -e .

RUN cd /home/work/transformers && \
    git checkout cc7ab9be508ce6ed3637bba9e50367b29b742dc6 && \
    pip install -v -e .

RUN bash /home/work/AgentSDK/aura/dockers/env/build_common.sh
RUN bash /home/work/AgentSDK/aura/dockers/patch/patch_triton_ascend.sh
RUN bash /home/work/AgentSDK/aura/dockers/patch/patch_vllm_ascend.sh

RUN pip install uv && \
    mkdir -p /home/work/model_env && \
    uv venv /home/work/model_env/qwen3_moe && \
    cd /home/work/model_env/qwen3_moe && \
    source bin/activate && \
    hash -r && \
    bash /home/work/AgentSDK/aura/dockers/env/build_qwen3_moe_env.sh && \
    deactivate

WORKDIR /home/work/AgentSDK/aura
