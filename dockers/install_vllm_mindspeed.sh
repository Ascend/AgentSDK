#!/bin/bash
git config --global http.sslVerify false

#########  install vllm/vllm-ascend
git clone -b releases/v0.9.1 https://github.com/vllm-project/vllm.git
cd vllm
git checkout b6553be1bc75f046b00046a4ad7576364d03c835
VLLM_TARGET_DEVICE=empty pip install .
cd ..

git clone -b v0.9.1-dev https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -r requirements.txt
pip install -e .
cd ..

######### install mindspeed_rl
# pip installation below requires configuring other domestic mirror sources; otherwise, some packages may not be found.
# Note that some mirror sources may have slower download speeds, which could result in unsuccessful installations.
# However, subsequent commands will continue to execute, and it may be necessary to switch to other mirror sources.

git clone https://gitee.com/ascend/MindSpeed-RL.git -b 2.1.0

git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout ca70c1338f1b3d1ce46a0ea426e5779ae1312e2e
pip install -r requirements.txt --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
\cp -rf mindspeed ../MindSpeed-RL/
cd ..

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
\cp -rf megatron ../MindSpeed-RL/
cd ..

git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout fe7d93c5b6dd36043203e6080e2d2566604e4860
\cp -rf mindspeed_llm ../MindSpeed-RL/
cd ..

cd ./MindSpeed-RL
pip install -r requirements.txt --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install antlr4-python3-runtime==4.7.2 --no-deps --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install datasets hydra-core loguru sentence_transformers vertexai --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/

# copy mindspeed/megatron/mindspeed_llm to current python package path
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp -r megatron $SITE_PACKAGES/
cp -r mindspeed $SITE_PACKAGES/
cp -r mindspeed_llm $SITE_PACKAGES/

######### install apex
git clone -b master https://gitee.com/ascend/apex.git
cd apex
bash scripts/build.sh --python=3.10
cd apex/dist/
pip install *.whl

######### clear tmp modified file
rm -rf /tmp/*
rm -f /tmp/.hosts_modified
rm -f /tmp/.bashrc_modified