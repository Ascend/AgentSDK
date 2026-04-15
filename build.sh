#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
# 构建入口启动脚本

#set -ex

me=$(basename $0)
root_dir=$(realpath $(dirname $0))
cd ${root_dir}

shared_path=${root_dir}/shared
pack_path=${shared_path}/pack

function show_help()
{
    echo "usage: ${me} <make>"
}

function file_exists()
{
    if [ ! -f $1 ] ; then
        echo -e "Check \e[40;37;1m$1\e[m|[ \e[40;31;1mFAIL\e[m ]" | awk -F"|" '{printf "%-160s%s\n", $1, $2}'
        exit 1
    else
        echo -e "Check \e[40;37;1m$1\e[m|[ \e[40;32;1mDONE\e[m ]" | awk -F"|" '{printf "%-160s%s\n", $1, $2}'
    fi
}

function check_tar_file()
{
    file_exists "${pack_path}/AgenticRL-linux.tar.gz"
}

function package()
{
    mkdir -p ${shared_path}
    cd ${shared_path}
    mkdir -p ${pack_path}

    cp -rf ${root_dir}/agentic_rl ${pack_path}
    cp -rf ${root_dir}/agents ${pack_path}
    cp -rf ${root_dir}/third_party ${pack_path}
    cp -rf ${root_dir}/logs ${pack_path}

    cd ${pack_path}
    tar -zcf AgenticRL-linux.tar.gz *
    check_tar_file

    echo "build & package AgenticRL succeed!!!"
}

if [ ! -d "${shared_path}" ]; then
    mkdir -p ${shared_path}
fi

if [ x$1 != x ]; then
    if [ $1 == "make" ]; then
        rm -rf ${shared_path}/*
        package
        exit 0
    else
        show_help
        exit 1
    fi
else
    show_help
    exit 1
fi
