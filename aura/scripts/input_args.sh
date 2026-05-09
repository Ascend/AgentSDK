#!/bin/bash

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

me=$(basename $0)

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --node_type)
            if [[ -n "$2" && "$2" != --* ]]; then
                node_type="$2"
                shift
            else
                echo -e "\e[40;31;1m[ERROR]:\e[m --node_type should input node type, such as master or worker"
                exit 1
            fi
            ;;
        --train_ip)
            if [[ -n "$2" && "$2" != --* ]]; then
                train_ip="$2"
                shift
            else
                echo -e "\e[40;31;1m[ERROR]:\e[m --train_ip should input train ip"
                exit 1
            fi
            ;;
        --rollout_ip)
            if [[ -n "$2" && "$2" != --* ]]; then
                rollout_ip="$2"
                shift
            else
                echo -e "\e[40;31;1m[ERROR]:\e[m --rollout_ip should input rollout ip"
                exit 1
            fi
            ;;
        --task_type)
            if [[ -n "$2" && "$2" != --* ]]; then
                task_type="$2"
                shift
            else
                echo -e "\e[40;31;1m[ERROR]:\e[m --task_type should input task type, such as train or rollout"
                exit 1
            fi
            ;;
        --help)
            echo -e "\e[40;32;1musage:\e[m"
            echo -e "\e[40;32;1m${me}\e[m"
            echo -e "\e[40;32;1m    [--node_type <master|worker>]\e[m"
            echo -e "\e[40;32;1m    [--train_ip <IP>]\e[m"
            echo -e "\e[40;32;1m    [--rollout_ip <IP>]\e[m"
            echo -e "\e[40;32;1m    [--task_type <train|rollout>]\e[m"
            exit 0
            ;;
        *)
            echo -e "\e[40;31;1munknown arg: $1\e[m"
            exit 1
            ;;
    esac
    shift
done
