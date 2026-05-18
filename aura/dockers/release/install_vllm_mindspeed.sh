#!/bin/bash
read -p "proxy user name: " proxy_user
read -s -p "proxy passward " proxy_pass

echo ""

git config --global --unset http.proxy
git config --global --unset https.proxy
git config --global http.proxy "http://${proxy_user}:${proxy_pass}@proxycn.huawei.com:8080"
git config --global https.proxy "http://${proxy_user}:${proxy_pass}@proxycn.huawei.com:8080"
git config --global http.sslVerify false
git config --global https.sslVerify false

pip show torch

export http_proxy="http://${proxy_user}:${proxy_pass}@proxycn.huawei.com:8080"
export https_proxy="http://${proxy_user}:${proxy_pass}@proxycn.huawei.com:8080"

python -m pip install --upgrade pip
pip install cmake>=3.26.1 --no-deps --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install setuptools_scm --no-deps --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install decorator --no-deps --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install absl-py --no-deps --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install ml-dtypes --no-deps --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install tornado --no-deps --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/

export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/aarch64-linux/devlib:$ASCEND_HOME/ascend-toolkit/latest/aarch64-linux/devlib/linux/aarch64:$ASCEND_HOME/driver/lib64/driver:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/ascend-toolkit/latest/python/site-packages:$PYTHONPATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh

######### install vllm/vllm-ascend
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout b5ee1e3261d9edf94d76ba8b437ebdef7ac599ea
VLLM_TARGET_DEVICE=empty pip install . --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
cd ..

git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout e7409e95ee73fb3bb7bf8b23f26c16620ed94543
pip install -r requirements.txt --no-build-isolation --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install -e . --no-build-isolation
cd ..

######### install mindspeed_rl
# pip installation below requires configuring other domestic mirror sources; otherwise, some packages may not be found.
# Note that some mirror sources may have slower download speeds, which could result in unsuccessful installations.
# However, subsequent commands will continue to execute, and it may be necessary to switch to other mirror sources.

export http_proxy="http://${proxy_user}:${proxy_pass}@proxycn.huawei.com:8080"
export https_proxy="http://${proxy_user}:${proxy_pass}@proxycn.huawei.com:8080"


git clone https://gitee.com/ascend/MindSpeed-RL.git -b 2.1.0

git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout ca70c1338f1b3d1ce46a0ea426e5779ae1312e2e
pip install -r requirements.txt --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/ --no-build-isolation
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
pip install -r requirements.txt --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/ --no-build-isolation
pip install antlr4-python3-runtime==4.7.2 --no-deps --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install datasets hydra-core loguru sentence_transformers vertexai --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/

# copy mindspeed/megatron/mindspeed_llm to current python package path
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp -r megatron $SITE_PACKAGES/
cp -r mindspeed $SITE_PACKAGES/
cp -r mindspeed_llm $SITE_PACKAGES/

echo "==========================install other packages============================"

pip install tensordict --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install tokenizers==0.21.1 --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install transformers==4.55.2 --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install latex2sympy2 --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install word2number --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install codetiming --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install mathruler --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install pylatexenc --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install tensorboard --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install click==8.2.1 --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/

pip install aiohttp_cors --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install opencensus --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/

echo "=====================install apex=============================="

pip install torch==2.7.1 --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
cd /opt/packages/PTA
pip install torch_npu-*.whl

pip install torchvision==0.22.1 --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/

cd -

######### install apex
git clone -b master https://gitee.com/ascend/apex.git
cd apex
bash scripts/build.sh --python=3.10
cd apex/dist/
pip install *.whl

######### unset git proxy
git config --global --unset http.proxy
git config --global --unset https.proxy
#rm -rf /tmp/*
rm -f /tmp/.hosts_modified
rm -f /tmp/.bashrc_modified
