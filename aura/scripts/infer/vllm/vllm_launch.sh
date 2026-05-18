#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

vllm_dir=$(realpath "$(dirname "$0")")
infer_dir=$(realpath "$(dirname "$vllm_dir")")
scripts_dir=$(realpath "$(dirname "$infer_dir")")

# Default values
export DEFAULT_SOCKET_IFNAME=${DEFAULT_SOCKET_IFNAME:-"eth0"}
VLLM_PORT=20012
PROXY_PORT=8080

# 初始值, 下面会根据实际配置自动修改
PREFILL_INSTANCE_COUNT=1
DECODE_INSTANCE_COUNT=3
PREFILL_CARDS_PER_INSTANCE=32
DECODE_CARDS_PER_INSTANCE=32
NODE_CARDS_COUNT=8
SOCKET_IFNAME="$DEFAULT_SOCKET_IFNAME"

# Argument parsing function
function parse_arguments()
{
    local OPTIONS="p:d:s:e:c:i:"
    local LONGOPTS="prefill-instances:,decode-instances:,prefill-cards-per-instance:,decode-cards-per-instance:,node-cards:,socket-ifname:"

    local TEMP
    TEMP=$(getopt -o $OPTIONS --longoptions $LONGOPTS -n 'start_vllm_cluster.sh' -- "$@")
    if [ $? != 0 ]; then
        echo "Error: Failed to parse arguments. Use --help for usage." >&2
        exit 1
    fi

    eval set -- "$TEMP"
    while true; do
        case "$1" in
            -p|--prefill-instances)
                PREFILL_INSTANCE_COUNT=$2; shift 2 ;;
            -d|--decode-instances)
                DECODE_INSTANCE_COUNT=$2; shift 2 ;;
            -s|--prefill-cards-per-instance)
                PREFILL_CARDS_PER_INSTANCE=$2; shift 2 ;;
            -e|--decode-cards-per-instance)
                DECODE_CARDS_PER_INSTANCE=$2; shift 2 ;;
            -c|--node-cards)
                NODE_CARDS_COUNT=$2; shift 2 ;;
            -i|--socket-ifname)
                SOCKET_IFNAME=$2; shift 2 ;;
            --)
                shift; break ;;
            *)
                echo "Error: Unknown argument '$1'"
                exit 1 ;;
        esac
    done

    if [ -z "$PREFILL_INSTANCE_COUNT" ] || [ -z "$DECODE_INSTANCE_COUNT" ] || \
       [ -z "$PREFILL_CARDS_PER_INSTANCE" ] || [ -z "$DECODE_CARDS_PER_INSTANCE" ] || \
       [ -z "$NODE_CARDS_COUNT" ]; then
        echo "Error: Missing required arguments"
        exit 1
    fi
    export PREFILL_TOTAL_CARDS=$((PREFILL_INSTANCE_COUNT * PREFILL_CARDS_PER_INSTANCE))
    export DECODE_TOTAL_CARDS=$((DECODE_INSTANCE_COUNT * DECODE_CARDS_PER_INSTANCE))
    export TOTAL_REQUIRED_CARDS=$((PREFILL_TOTAL_CARDS + DECODE_TOTAL_CARDS))
}

function validate_parameters()
{
    echo "   - Prefill instance count: ${PREFILL_INSTANCE_COUNT}"
    echo "   - Decode instance count: ${DECODE_INSTANCE_COUNT}"
    echo "   - Cards per Prefill instance: ${PREFILL_CARDS_PER_INSTANCE}"
    echo "   - Cards per Decode instance: ${DECODE_CARDS_PER_INSTANCE}"
    echo "   - Cards per node: ${NODE_CARDS_COUNT}"
    echo "   - Network interface name: ${SOCKET_IFNAME}"
    echo "   - Total required cards: ${TOTAL_REQUIRED_CARDS}"

    local TOTAL_AVAILABLE_CARDS=$((NODE_COUNT * NODE_CARDS_COUNT))
    if [ "$TOTAL_AVAILABLE_CARDS" -lt "$TOTAL_REQUIRED_CARDS" ]; then
        echo "Error: Insufficient total cards in cluster!"
        exit 1
    fi
    echo "   - Validation passed: Available cards (${TOTAL_AVAILABLE_CARDS}) >= Required cards (${TOTAL_REQUIRED_CARDS})"
}

function get_node_ips()
{
    if [ -z "$VC_TASK_HOSTS" ] || [ -z "$THIS_POD_IP" ] || [ -z "$VC_TASK_INDEX" ]; then
        echo "Error: Missing critical environment variables"
        exit 1
    fi

    local TASK_HOSTS_DOMAINS=${VC_TASK_HOSTS//,/ }
    local first_element=$(echo "$TASK_HOSTS_DOMAINS" | awk '{print $1}')
    NODE_IP_LIST=""

    if [[ "$first_element" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        NODE_IP_LIST=$TASK_HOSTS_DOMAINS
    else
        for domain in ${TASK_HOSTS_DOMAINS}; do
            local IP_ADDRESS=$(python -c "import socket; print(socket.gethostbyname('$domain'))")
            if [ $? -ne 0 ] || [ -z "$IP_ADDRESS" ]; then
                echo "Error: Failed to resolve domain $domain"
                exit 1
            fi
            NODE_IP_LIST="${NODE_IP_LIST} ${IP_ADDRESS}"
        done
    fi

    NODE_IP_ARRAY=($NODE_IP_LIST)
    NODE_COUNT=${#NODE_IP_ARRAY[@]}

    echo "   - Node IP list: ${NODE_IP_LIST}"
    echo "   - Total nodes: ${NODE_COUNT}"
}

# Generate ranktable.json and initialize hccn.conf
function generate_ranktable()
{
    local LOCAL_RANKTABLE_SCRIPT="${scripts_dir}/infer/vllm/gen_ranktable.sh"
    local LOCAL_RANKTABLE_PYTHON="${scripts_dir}/infer/vllm/gen_ranktable.py"
    local TARGET_RANKTABLE_SCRIPT="/gen_ranktable.sh"
    local TARGET_RANKTABLE_PYTHON="/gen_ranktable.py"

    if [ -f "$LOCAL_RANKTABLE_SCRIPT" ] && [ ${USE_PD} -eq 1 ]; then
        local CURRENT_DIR=$(pwd)
        cp "$LOCAL_RANKTABLE_SCRIPT" "$TARGET_RANKTABLE_SCRIPT" || { cd "$CURRENT_DIR"; exit 1; }

        if [ -f "$LOCAL_RANKTABLE_PYTHON" ]; then
            cp "$LOCAL_RANKTABLE_PYTHON" "$TARGET_RANKTABLE_PYTHON" || { rm -f "$TARGET_RANKTABLE_SCRIPT"; cd "$CURRENT_DIR"; exit 1; }
        fi

        cd /
        rm -f /ranktable.json

        local PREFILL_TOTAL_CARDS=$((PREFILL_TOTAL_CARDS < 8 ? 8 : PREFILL_TOTAL_CARDS))
        local DECODE_TOTAL_CARDS=$((DECODE_TOTAL_CARDS < 8 ? 8 : DECODE_TOTAL_CARDS))
        local IPS_ARG=$(echo "${NODE_IP_LIST}" | sed 's/  */ /g')
        # 打印即将执行的命令 (在 / 目录下执行 /gen_ranktable.sh)
        echo "--- 🔑 KEY COMMAND: RANKTABLE GENERATION (Executed in /) ---"
        echo "sh $TARGET_RANKTABLE_SCRIPT \\"
        echo "    --ips ${IPS_ARG} \\"
        echo "    --prefill-device-cnt ${PREFILL_TOTAL_CARDS} \\"
        echo "    --decode-device-cnt ${DECODE_TOTAL_CARDS} \\"
        echo "    --network-card-name ${SOCKET_IFNAME}"
        echo "------------------------------------------------------------"

        sh "$TARGET_RANKTABLE_SCRIPT" \
            --ips ${IPS_ARG} \
            --prefill-device-cnt ${PREFILL_TOTAL_CARDS} \
            --decode-device-cnt ${DECODE_TOTAL_CARDS} \
            --network-card-name ${SOCKET_IFNAME}

        setup_hccn_conf $THIS_POD_IP /ranktable.json || true

        rm -f "$TARGET_RANKTABLE_SCRIPT" "$TARGET_RANKTABLE_PYTHON"
        cd "$CURRENT_DIR"
    else
        echo "PD hybrid mode, skip generate rank table"
    fi
}

function setup_hccn_conf()
{
    local target_server_id=$1
    local ranktable_path=${2:-"ranktable.json"}
    local hccn_file="/etc/hccn.conf"
    local netmask="255.255.0.0"

    if [ -f "$hccn_file" ]; then return 0; fi

    if [ ! -f "$ranktable_path" ]; then
        echo "[ERROR] Ranktable file not found: $ranktable_path"
        return 1
    fi

    local devices_info
    devices_info=$(python3 -c "
import json, sys
try:
    with open('$ranktable_path', 'r') as f:
        data = json.load(f)
    all_devs = data.get('prefill_device_list', []) + data.get('decode_device_list', [])
    found = []
    for d in all_devs:
        if d.get('server_id') == '$target_server_id':
            ip = d.get('device_hccn_ip') or d.get('device_ip')
            dev_id = d.get('device_id')
            if ip and dev_id is not None:
                found.append(f'{dev_id} {ip}')
    found.sort(key=lambda x: int(x.split()[0]))
    print('\n'.join(found))
except Exception as e:
    sys.exit(1)
" 2>/dev/null)

    if [ -z "$devices_info" ]; then
        echo "[WARN] No device info found for server_id $target_server_id in ranktable."
        return 1
    fi

    [ ! -f "$hccn_file" ] && touch "$hccn_file"
    true > "$hccn_file"

    for i in {0..7}; do
        echo "tls_ca_recovered_$i=1" >> "$hccn_file"
    done

    while read -r did dip; do
        {
            echo "address_$did=$dip"
            echo "netmask_$did=$netmask"
            echo "send_arp_status_$did=1"
            echo "tls_enable_$did=0"
        } >> "$hccn_file"
    done <<< "$devices_info"

    echo "[SUCCESS] $hccn_file has been generated using device_hccn_ip."
}

function launch_vllm_instances()
{
    local NODE_IP_ARRAY_INDEX=$VC_TASK_INDEX
    local ROLE=""
    local INSTANCE_INDEX=0
    local MASTER_ADDR=""
    local local_node_rank=0
    local ENGINE_ID=0

    local PREFILL_NODES_PER_INSTANCE=$(( (PREFILL_CARDS_PER_INSTANCE + NODE_CARDS_COUNT - 1) / NODE_CARDS_COUNT ))
    local DECODE_NODES_PER_INSTANCE=$(( (DECODE_CARDS_PER_INSTANCE + NODE_CARDS_COUNT - 1) / NODE_CARDS_COUNT ))
    local PREFILL_TOTAL_NODES=$((PREFILL_INSTANCE_COUNT * PREFILL_NODES_PER_INSTANCE))
    local DECODE_TOTAL_NODES=$((DECODE_INSTANCE_COUNT * DECODE_NODES_PER_INSTANCE))
    local TOTAL_USED_NODES=$((PREFILL_TOTAL_NODES + DECODE_TOTAL_NODES))

    if [ "$NODE_IP_ARRAY_INDEX" -lt "$PREFILL_TOTAL_NODES" ]; then
        ROLE="prefill"
        INSTANCE_INDEX=$((NODE_IP_ARRAY_INDEX / PREFILL_NODES_PER_INSTANCE))
        local MASTER_NODE_INDEX=$((INSTANCE_INDEX * PREFILL_NODES_PER_INSTANCE))
        MASTER_ADDR=${NODE_IP_ARRAY[MASTER_NODE_INDEX]}
        local_node_rank=$((NODE_IP_ARRAY_INDEX - MASTER_NODE_INDEX))
        ENGINE_ID=${INSTANCE_INDEX}
    elif [ "$NODE_IP_ARRAY_INDEX" -lt "$TOTAL_USED_NODES" ]; then
        ROLE="decode"
        local DECODE_INDEX_OFFSET=$((NODE_IP_ARRAY_INDEX - PREFILL_TOTAL_NODES))
        INSTANCE_INDEX=$((DECODE_INDEX_OFFSET / DECODE_NODES_PER_INSTANCE))
        local MASTER_NODE_INDEX=$((PREFILL_TOTAL_NODES + INSTANCE_INDEX * DECODE_NODES_PER_INSTANCE))
        MASTER_ADDR=${NODE_IP_ARRAY[MASTER_NODE_INDEX]}
        local_node_rank=$((NODE_IP_ARRAY_INDEX - MASTER_NODE_INDEX))
        ENGINE_ID=$((INSTANCE_INDEX + PREFILL_INSTANCE_COUNT))
    else
        echo "Info: Current node is not assigned to any VLLM instance. Exiting."
        exit 0
    fi

    local HOST=$THIS_POD_IP
    local PORT=$VLLM_PORT

    echo "   - Node role (ROLE): ${ROLE} HOST=${HOST}"
    echo "   - Node ENGINE_ID: ${ENGINE_ID}"
    echo "   - Instance Master address (MASTER_ADDR): ${MASTER_ADDR}"
    echo "   - Local node rank within instance (local_node_rank): ${local_node_rank}"

    # Launch proxy (only on first node)
    if [ "$VC_TASK_INDEX" -eq 0 ]; then
        launch_proxy_server
    fi

    # Launch vLLM instance
    launch_vllm_service "$HOST" "$PORT" "$MASTER_ADDR" "$local_node_rank" "$ROLE" "$ENGINE_ID"
}

function launch_proxy_server()
{
    local PREFILLER_HOSTS_LIST=()
    local DECODER_HOSTS_LIST=()
    local i

    for i in $(seq 0 $((PREFILL_INSTANCE_COUNT - 1))); do
        local NODE_INDEX=$((i * PREFILL_NODES_PER_INSTANCE))
        PREFILLER_HOSTS_LIST+=("${NODE_IP_ARRAY[NODE_INDEX]}")
    done
    local PREFILLER_HOSTS_ARG=$(IFS=' '; echo "${PREFILLER_HOSTS_LIST[*]}")
    local PREFILLER_PORTS_ARG=$(seq 1 $PREFILL_INSTANCE_COUNT | xargs -I {} echo -n "$VLLM_PORT ")

    for i in $(seq 0 $((DECODE_INSTANCE_COUNT - 1))); do
        local NODE_INDEX=$((PREFILL_TOTAL_NODES + i * DECODE_NODES_PER_INSTANCE))
        DECODER_HOSTS_LIST+=("${NODE_IP_ARRAY[NODE_INDEX]}")
    done
    local DECODER_HOSTS_ARG=$(IFS=' '; echo "${DECODER_HOSTS_LIST[*]}")
    local DECODER_PORTS_ARG=$(seq 1 $DECODE_INSTANCE_COUNT | xargs -I {} echo -n "$VLLM_PORT ")

    local PROXY_SCRIPT="${scripts_dir}/infer/vllm/load_balance_proxy_server.py"
    if [ -f "$PROXY_SCRIPT" ]; then
        echo "   - Starting proxy in background: ${PROXY_SCRIPT}"
        if [ ${USE_PD} -eq 1 ]; then
          python $PROXY_SCRIPT \
              --host 0.0.0.0 \
              --port ${PROXY_PORT} \
              --prefiller-hosts ${PREFILLER_HOSTS_ARG} \
              --prefiller-ports ${PREFILLER_PORTS_ARG} \
              --decoder-hosts ${DECODER_HOSTS_ARG} \
              --decoder-ports ${DECODER_PORTS_ARG} &
        else
          python $PROXY_SCRIPT \
              --host 0.0.0.0 \
              --port ${PROXY_PORT} \
              --prefiller-hosts ${PREFILLER_HOSTS_ARG} \
              --prefiller-ports ${PREFILLER_PORTS_ARG} &
        fi
        local PROXY_PID=$!
        echo $PROXY_PID > /tmp/proxy_server.pid
    else
        echo "Warning: Load balance proxy script not found. Skipping proxy launch."
    fi
}

function launch_vllm_service()
{
    local HOST=$1
    local PORT=$2
    local MASTER_ADDR=$3
    local local_node_rank=$4
    local ROLE=$5
    local ENGINE_ID=$6

    if [ -f "${scripts_dir}/infer/vllm/vllm_serve.sh" ]; then
        bash ${scripts_dir}/infer/vllm/vllm_serve.sh \
            --host ${HOST} \
            --port ${PORT} \
            --master_addr ${MASTER_ADDR} \
            --local_node_rank ${local_node_rank} \
            --role ${ROLE} \
            --engine_id ${ENGINE_ID}
        if [ $? -ne 0 ]; then
            echo "Error: vLLM service failed to start or exited abnormally."
            exit 1
        fi
        echo "Info: vLLM service exited normally."
    else
        echo "Error: vLLM startup script not found."
        exit 1
    fi
}

function main()
{
    parse_arguments "$@"
    get_node_ips
    validate_parameters
    generate_ranktable
    launch_vllm_instances
}

# Execute main function
main "$@"
