#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# TrajProxy download

file_dir=$(realpath $(dirname $0))
root_dir=${file_dir}
traj_proxy_dir=${file_dir}/app

requirements_file="requirements_traj_proxy.txt"

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

function download_package_succeed()
{
  name=$1
  echo -e "\e[40;32;1mdownload ${name} src code succeed\e[m"
}

function download_traj_proxy_src_code()
{
  commit_id=$(cat ${root_dir}/${requirements_file} | grep TrajProxy | awk -F'==' '{print $2}')
  echo "start download TrajProxy src code, version: ${commit_id}"

  mkdir -p ${root_dir}/tmp/trajproxy
  cd ${root_dir}/tmp/trajproxy
  git clone -b main https://github.com/infzo/TrajProxy.git
  cd TrajProxy
  git branch
  git checkout ${commit_id}
  git branch
  short_commit_id=${commit_id:0:5}
  check_succeed TrajProxy ${short_commit_id}

  mkdir -p ${traj_proxy_dir}
  cd ${root_dir}/tmp/trajproxy
  cp -rf TrajProxy/* ${traj_proxy_dir}/
  download_package_succeed TrajProxy
}

function clean_old_traj_proxy_srcs()
{
  echo "start clean old TrajProxy srcs"
  rm -fr ${traj_proxy_dir}

  echo -e "\e[40;32;1mclean old TrajProxy srcs succeed\e[m"
}

rm -rf ${root_dir}/tmp
mkdir -p ${root_dir}/tmp

dos2unix ${root_dir}/${requirements_file}

clean_old_traj_proxy_srcs
download_traj_proxy_src_code

rm -rf ${root_dir}/tmp
