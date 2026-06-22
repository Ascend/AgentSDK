#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
# 三方件下载脚本

root_dir=$(realpath $(dirname $0))
third_party_dir=${root_dir}/third_party

rm -rf ${root_dir}/tmp
mkdir -p ${root_dir}/tmp

requirements_files=(
  "requirements_with_msrl_vllm.txt"
  "requirements_with_verl_vllm.txt"
)

echo "support following requirements:"
number=0
for item in "${requirements_files[@]}"; do
  ((number++))
  echo "    ${number}. ${item}"
done
read -p "chose one requirements file (1-${number}): " requirements_no

index=$((requirements_no-1))
requirements_file=${requirements_files[${index}]}

echo "selected: ${requirements_file}"

dos2unix ${root_dir}/third_party/${requirements_file}

git config --global --unset http.proxy
git config --global --unset https.proxy
git config --global http.sslVerify false
git config --global https.sslVerify false

function check_succeed()
{
  name=$1
  id=$2
  checkout_flag=$(git branch | grep ${id} | wc -l)
  if [[ ${checkout_flag} != "1" ]]; then
    echo -e "\e[40;31;1m[ERROR]: \e[m${name} checkout to ${id} failed, please use 'dos2unix ${requirements_file}' to change format..."
    exit 1
  fi
}

function download_pacakge_succeed()
{
  name=$1
  echo -e "\e[40;32;1mdownload ${name} src code succeed\e[m"
}

function download_mindspeed_rl_src_code()
{
  commit_id=$(cat ${root_dir}/third_party/${requirements_file} | grep mindspeed_rl | awk -F'==' '{print $2}')
  echo "start download mindspeed_rl src code, version: ${commit_id}"

  mkdir -p ${root_dir}/tmp/mindspeed_rl
  cd ${root_dir}/tmp/mindspeed_rl
  git clone https://github.com/Ascend/MindSpeed-RL.git
  cd MindSpeed-RL
  git branch
  git checkout ${commit_id}
  git branch
  short_commit_id=${commit_id:0:5}
  check_succeed mindspeed_rl ${short_commit_id}
  cd ${root_dir}/tmp/mindspeed_rl
  cp -rf MindSpeed-RL ${third_party_dir}/rl/mindspeed_rl
  download_pacakge_succeed mindspeed_rl
}

function download_megatron_src_code()
{
  branch_id=$(cat ${root_dir}/third_party/${requirements_file} | grep megatron | awk -F'==' '{print $2}')
  echo "start download megatron src code, version: ${branch_id}"

  mkdir -p ${root_dir}/tmp/megatron
  cd ${root_dir}/tmp/megatron
  git clone -b core_r0.8.0 https://github.com/NVIDIA/Megatron-LM.git
  cd Megatron-LM
  git branch
  git checkout ${branch_id}
  git branch
  check_succeed megatron ${branch_id}
  cd ${root_dir}/tmp/megatron
  cp -rf Megatron-LM ${third_party_dir}/rl/megatron
  download_pacakge_succeed megatron
}

function download_mindspeed_src_code()
{
  commit_id=$(cat ${root_dir}/third_party/${requirements_file} | grep mindspeed | grep -v mindspeed_rl | grep -v mindspeed_llm | awk -F'==' '{print $2}')
  echo "start download mindspeed src code, version: ${commit_id}"

  mkdir -p ${root_dir}/tmp/mindspeed
  cd ${root_dir}/tmp/mindspeed
  git clone https://github.com/Ascend/MindSpeed.git
  cd MindSpeed
  git branch
  git checkout ${commit_id}
  git branch
  short_commit_id=${commit_id:0:5}
  check_succeed mindspeed ${short_commit_id}
  cd ${root_dir}/tmp/mindspeed
  cp -rf MindSpeed ${third_party_dir}/rl/mindspeed
  download_pacakge_succeed mindspeed
}

function download_mindspeed_llm_src_code()
{
  commit_id=$(cat ${root_dir}/third_party/${requirements_file} | grep mindspeed_llm | awk -F'==' '{print $2}')
  echo "start download mindspeed_llm src code, version: ${commit_id}"

  mkdir -p ${root_dir}/tmp/mindspeed_llm
  cd ${root_dir}/tmp/mindspeed_llm
  git clone https://github.com/Ascend/MindSpeed-LLM.git
  cd MindSpeed-LLM
  git branch
  git checkout ${commit_id}
  git branch
  short_commit_id=${commit_id:0:5}
  check_succeed mindspeed_llm ${short_commit_id}
  cd ${root_dir}/tmp/mindspeed_llm
  cp -rf MindSpeed-LLM ${third_party_dir}/rl/mindspeed_llm
  download_pacakge_succeed mindspeed_llm

  echo "patch for mindspeed_llm..."
  sed -i "/            raise AssertionError('Context parallelism is forbidden when use variable seq lengths.')/c\            print('WARNING: Context parallelism is forbidden when use variable seq lengths.')" ${third_party_dir}/rl/mindspeed_llm/mindspeed_llm/training/arguments.py
}

function download_rllm_src_code()
{
  branch_id=$(cat ${root_dir}/third_party/${requirements_file} | grep rllm | awk -F'==' '{print $2}')
  echo "start download rllm src code, version: ${branch_id}"

  mkdir -p ${root_dir}/tmp/rllm
  cd ${root_dir}/tmp/rllm
  git clone -b main https://github.com/agentica-project/rllm.git
  cd rllm
  git branch
  git checkout -b ${branch_id}
  git branch
  check_succeed rllm ${branch_id}
  cd ${root_dir}/tmp/rllm
  cp -rf rllm ${third_party_dir}/agent_engine/rllm
  download_pacakge_succeed rllm
}

function clean_old_srcs()
{
  echo "start clean old srcs"
  cd ${third_party_dir}/rl
  rm -rf $(ls -lrt ${third_party_dir}/rl/ | grep -v __init__.py)

  cd ${third_party_dir}/infer
  rm -rf $(ls -lrt ${third_party_dir}/infer/ | grep -v __init__.py)

  cd ${third_party_dir}/agent_engine
  rm -rf $(ls -lrt ${third_party_dir}/agent_engine | grep -v __init__.py)
  cd -
  echo -e "\e[40;32;1mclean old srcs succeed\e[m"
}

clean_old_srcs
download_mindspeed_rl_src_code
download_megatron_src_code
download_mindspeed_src_code
download_mindspeed_llm_src_code
download_rllm_src_code

rm -rf ${root_dir}/tmp

echo -e "\e[40;32;1mDownload third party source code succeed\e[m"
