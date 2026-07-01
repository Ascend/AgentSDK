#!/bin/bash
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

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
    file_exists "${pack_path}/aura-linux.tar.gz"
}

function package()
{
    mkdir -p ${shared_path}
    cd ${shared_path}
    mkdir -p ${pack_path}

    cp -rf ${root_dir}/aura/aura ${pack_path}
    cp -rf ${root_dir}/aura/aura/agents ${pack_path}
    cp -rf ${root_dir}/aura/aura/third_party ${pack_path}
    cp -rf ${root_dir}/aura/aura/logs ${pack_path}

    cd ${pack_path}
    tar -zcf aura-linux.tar.gz *
    check_tar_file

    echo "build & package aura succeed!!!"
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
