#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
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

import subprocess

def get_value_from_lines(lines, key):
    """Extract the value of a specified key from multiple lines of text."""
    for line in lines:
        line = " ".join(line.split())
        if key in line:
            return line.split(":")[-1].strip()
    return ""

def get_chip_type():
    """Get the chip model via the npu-smi command."""
    try:
        # Get NPU ID
        npu_info_lines = subprocess.check_output(["npu-smi", "info", "-l"]).decode().strip().split("\n")
        npu_id = int(get_value_from_lines(npu_info_lines, "NPU ID"))

        # Get board information
        board_info_lines = subprocess.check_output(
            ["npu-smi", "info", "-t", "board", "-i", str(npu_id)]
        ).decode().strip().split("\n")

        # Extract chip name
        chip_name = get_value_from_lines(board_info_lines, "Chip Name")

        # If Chip Name is not found, try adding -c 0 parameter (applicable to A2/A3/310P)
        if not chip_name:
            chip_info_lines = subprocess.check_output(
                ["npu-smi", "info", "-t", "board", "-i", str(npu_id), "-c", "0"]
            ).decode().strip().split("\n")
        else:
            chip_info_lines = board_info_lines

        # Extract required fields
        chip_name = get_value_from_lines(chip_info_lines, "Chip Name")
        chip_type = get_value_from_lines(chip_info_lines, "Chip Type")
        npu_name = get_value_from_lines(chip_info_lines, "NPU Name")

        # Compose SOC_VERSION based on different chips
        if "310" in chip_name:
            return (chip_type + chip_name).lower()
        elif "910" in chip_name:
            if chip_type:
                return (chip_type + chip_name).lower()
            else:
                return (chip_name + "_" + npu_name).lower()
        elif "950" in chip_name:
            return (chip_name + "_" + npu_name).lower()
        else:
            raise ValueError(f"Unrecognized chip name: {chip_name}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get chip information: {e}")
    except FileNotFoundError:
        raise RuntimeError("npu-smi command not found, please check if the NPU driver is properly installed.")

# Only print the value of SOC_VERSION
print(get_chip_type())
