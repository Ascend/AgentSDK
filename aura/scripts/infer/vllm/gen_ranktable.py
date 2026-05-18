# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
Description:
Author: Aura Team
"""

import argparse
import json
import os
from enum import Enum
import torch_npu
import torch.distributed as dist


class AscendDeviceType(Enum):
    ASCEND_A2 = 0
    ASCEND_A3 = 1
    ASCEND_310P = 2
    ASCEND_A5 = 3


parser = argparse.ArgumentParser(
    description="Arguments of rank table generator",
)
parser.add_argument("--local-host", type=str, required=True, help="local ip")
parser.add_argument("--prefill-device-cnt", type=int, required=True, help="number of prefill devices")
parser.add_argument("--decode-device-cnt", type=int, required=True, help="number of decode devices")
parser.add_argument("--local-device-ids", type=str, required=False, help="local device ids")
args = parser.parse_args()
local_host = args.local_host
prefill_device_cnt = args.prefill_device_cnt
decode_device_cnt = args.decode_device_cnt

print("enter py")

hccn_tool_path = os.environ.get("HCCN_TOOL_PATH", "/usr/local/Ascend/driver/tools/hccn_tool")
master_addr = os.environ.get("MASTER_ADDR")
master_port = os.environ.get("MASTER_PORT")
rank = os.environ.get("RANK")
local_rank = os.environ.get("LOCAL_RANK")
# This variable is set by torchrun,
# and is different from WORLD_SIZE in gen_rank_table.sh.
world_size = os.environ.get("WORLD_SIZE")


def get_soc_version():
    cur_device_type = AscendDeviceType.ASCEND_A2
    soc_version = torch_npu.npu.get_soc_version()
    if 220 <= soc_version <= 225:
        cur_device_type = AscendDeviceType.ASCEND_A2
    elif 250 <= soc_version <= 255:
        cur_device_type = AscendDeviceType.ASCEND_A3
    elif 200 <= soc_version <= 205:
        cur_device_type = AscendDeviceType.ASCEND_310P
    elif soc_version == 260:
        cur_device_type = AscendDeviceType.ASCEND_A5
    return cur_device_type


soc_info = get_soc_version()


def get_cmd_stdout(cmd):
    import subprocess

    return subprocess.run(cmd, capture_output=True, shell=False).stdout.decode("utf-8").strip()


print(f"local_host: {local_host}")
print("gen ranktable.json")

num_cards = get_cmd_stdout("npu-smi info -l | grep \"Total Count\"").split(":")[1].strip()
num_cards = int(num_cards)
chips_per_card = get_cmd_stdout("npu-smi info -l | grep \"Chip Count\"").split("\n")[0].split(":")[1].strip()
chips_per_card = int(chips_per_card)

if args.local_device_ids:
    local_device_ids = args.local_device_ids.split(',')
else:
    local_device_ids = []
    for card_id in range(num_cards):
        for chip_id in range(chips_per_card):
            device_id = card_id * chips_per_card + chip_id
            local_device_ids.append(device_id)

# generate local device list for local rank 0, and gather it to all ranks
local_device_list: list[dict[str, str]] = list()
if local_rank == "0":
    super_pod_id = "0"
    for idx in range(len(local_device_ids)):
        device_id = local_device_ids[idx]
        chip_id = device_id % chips_per_card
        card_id = device_id // chips_per_card
        if soc_info == AscendDeviceType.ASCEND_A3:
            device_ip = get_cmd_stdout(f"{hccn_tool_path} -i {device_id} -vnic -g | grep ipaddr").split(":")[1].strip()
            device_ip2 = get_cmd_stdout(f"{hccn_tool_path} -i {device_id} -ip -g | grep ipaddr").split(":")[1].strip()
            super_device_id = (
                get_cmd_stdout(f"npu-smi info -t spod-info -i {card_id} -c {chip_id} | grep SDID").split(":")[1].strip()
            )
            super_pod_id = (
                get_cmd_stdout(f"npu-smi info -t spod-info -i {card_id} -c {chip_id} | grep \"Super Pod ID\"")
                .split(":")[1]
                .strip()
            )
        else:
            device_ip = get_cmd_stdout(f"{hccn_tool_path} -i {device_id} -ip -g | grep ipaddr").split(":")[1].strip()
            device_ip2 = device_ip

        device_info = {
            "server_id": local_host,
            "device_id": str(device_id),
            "device_ip": str(device_ip),
            "device_hccn_ip": str(device_ip2),
        }
        if soc_info == AscendDeviceType.ASCEND_A3:
            device_info.update({"super_pod_id": str(super_pod_id), "super_device_id": str(super_device_id)})
        local_device_list.append(device_info)

dist.init_process_group(backend=dist.Backend.GLOO)
global_device_list = [None] * dist.get_world_size()
dist.all_gather_object(global_device_list, local_device_list)
global_device_list = [device_info for device_list in global_device_list for device_info in device_list]
cnt = 1
for device_info in global_device_list:
    device_info["cluster_id"] = str(cnt)
    cnt += 1

total_required_devices = prefill_device_cnt + decode_device_cnt
if total_required_devices > len(global_device_list):
    raise ValueError(
        "prefill_device_cnt + decode_device_cnt must be less than or equal to the number of all devices in cluster"
    )

ranktable = {
    "version": "1.2",
    "server_count": str(world_size),
    "prefill_device_list": global_device_list[:prefill_device_cnt],
    "decode_device_list": global_device_list[prefill_device_cnt : prefill_device_cnt + decode_device_cnt],
    "status": "completed",
}

if local_rank == '0':
    with open("ranktable.json", "w") as f:
        json.dump(ranktable, f, indent=4)

    print("gen ranktable.json done")
