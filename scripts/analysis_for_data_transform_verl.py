#!/usr/bin/env python3
# coding=utf-8

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

import json
import os
import glob
import numpy as np
from json import JSONDecoder

# ==========================================
# 1. 基础工具函数
# ==========================================

def parse_multiple_jsons(content):
    """解析堆叠的JSON对象"""
    decoder = JSONDecoder()
    content = content.strip()
    objs = []
    while content:
        try:
            obj, idx = decoder.raw_decode(content)
            objs.append(obj)
            content = content[idx:].lstrip()
        except Exception as e:
            print(f"解析警告: 在剩余内容中停止解析 - {str(e)}")
            break
    return objs

def safe_parse_json_list(val):
    """
    尝试将输入转换为列表。
    支持输入为: List, 或 String形式的List ("[1, 2]")
    """
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('[') and val.endswith(']'):
            try:
                return json.loads(val)
            except:
                pass
    return []

def normalize_and_strip_padding(val, pad_token=151643):
    """
    核心修复逻辑：
    1. 确保转为 int list
    2. 移除末尾所有的 pad_token
    3. 转为 tuple 以便哈希
    """
    # 1. 解析并确保是列表
    data_list = safe_parse_json_list(val)
    if not data_list:
        # 如果解析失败或者是空，返回原始值的字符串形式作为fallback
        return str(val)

    # 2. 确保元素是int (防止一个是 "100" 一个是 100)
    try:
        data_list = [int(x) for x in data_list]
    except:
        return tuple(data_list)

    # 3. 移除末尾的 Padding (151643)
    # 使用 while 循环从后往前删
    while data_list and data_list[-1] == pad_token:
        data_list.pop()

    return tuple(data_list)

def compare_arrays_detailed(arr1, arr2, field_name, rtol=1e-5):
    """
    增强版对比函数：提供详细的错误定位
    """
    if arr1 is None or arr2 is None:
        return False, f"存在None值: Source1={arr1 is None}, Source2={arr2 is None}"

    try:
        a1 = np.array(arr1)
        a2 = np.array(arr2)
    except Exception as e:
        return False, f"数组转换失败: {str(e)}"

    # 1. 形状检查
    if a1.shape != a2.shape:
        # 尝试打印形状差异
        return False, f"形状不匹配: Out={a1.shape} vs Verl={a2.shape}"

    # 2. 数值检查
    is_float = np.issubdtype(a1.dtype, np.floating) or np.issubdtype(a2.dtype, np.floating)

    if is_float:
        # 浮点数对比
        if np.allclose(a1, a2, rtol=rtol, equal_nan=True):
            return True, "Match"
        else:
            diff = np.abs(a1 - a2)
            max_diff = np.nanmax(diff)
            # 找到第一个超出误差的索引
            bad_indices = np.where(diff > (rtol + 1e-8))[0] # 简单处理1D/2D
            first_idx = bad_indices[0] if len(bad_indices) > 0 else "Unknown"

            detail_msg = f"浮点差异 (Max: {max_diff:.6f}). "
            if first_idx != "Unknown":
                # 获取具体数值
                v1 = a1.flatten()[first_idx] if a1.ndim > 0 else a1
                v2 = a2.flatten()[first_idx] if a2.ndim > 0 else a2
                detail_msg += f"首个差异在索引 [{first_idx}]: {v1} vs {v2}"
            return False, detail_msg
    else:
        # 整数/对象对比
        if np.array_equal(a1, a2):
            return True, "Match"
        else:
            # 找到具体哪里不一样
            # 展平以便统一处理索引
            flat1 = a1.flatten()
            flat2 = a2.flatten()
            mismatch_indices = np.where(flat1 != flat2)[0]

            count = len(mismatch_indices)
            first_idx = mismatch_indices[0] if count > 0 else 0
            v1 = flat1[first_idx]
            v2 = flat2[first_idx]

            return False, (f"值不匹配 (共 {count} 处). "
                           f"首个差异在索引 [{first_idx}]: Out={v1} vs Verl={v2}")

# ==========================================
# 2. 数据加载器
# ==========================================

def load_out_json_data(file_path):
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    raw_objs = parse_multiple_jsons(content)
    data_map = {}

    for obj in raw_objs:
        iteration = str(obj.get("iteration", "0"))
        outputs = obj.get("outputs", {})

        if iteration not in data_map:
            data_map[iteration] = {}

        # 预处理所有字段
        processed_outputs = {}
        for k, v in outputs.items():
            processed_outputs[k] = safe_parse_json_list(v)

        if 'prompts' not in processed_outputs or 'responses' not in processed_outputs:
            continue

        prompts_list = processed_outputs['prompts']
        responses_list = processed_outputs['responses']
        batch_size = len(prompts_list)

        for i in range(batch_size):
            try:
                # === 生成主键 (Strip Padding) ===
                # out.json 里的 prompts/responses 可能是 Token IDs 的列表
                key_p = normalize_and_strip_padding(prompts_list[i])
                key_r = normalize_and_strip_padding(responses_list[i])
                key = (key_p, key_r)

                record = {
                    "source": "out.json",
                    # 安全获取并处理可能的字符串嵌套
                    "attention_mask": safe_parse_json_list(processed_outputs.get('attention_mask', [])[i]),
                    "response_mask": safe_parse_json_list(processed_outputs.get('response_mask', [])[i]),
                    "position_ids": safe_parse_json_list(processed_outputs.get('position_ids', [])[i]),
                    "rm_scores": safe_parse_json_list(processed_outputs.get('rm_scores', [])[i]),
                    "token_level_rewards": safe_parse_json_list(processed_outputs.get('token_level_rewards', [])[i])
                }
                data_map[iteration][key] = record
            except IndexError:
                continue

    return data_map

def load_verl_data(verl_dir):
    print(f"Loading Verl data from {verl_dir}...")
    data_map = {}

    files = glob.glob(os.path.join(verl_dir, "*.jsonl"))

    for f_path in files:
        filename = os.path.basename(f_path)
        try:
            verl_iter_num = int(filename.split('.')[0])
            target_iter = str(verl_iter_num - 1)
        except ValueError:
            continue

        if target_iter not in data_map:
            data_map[target_iter] = {}

        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    row = json.loads(line)

                    # === 生成主键 (Strip Padding) ===
                    # Verl 里的 raw_input/raw_output 本身就是 ID list
                    key_p = normalize_and_strip_padding(row.get("raw_input"))
                    key_r = normalize_and_strip_padding(row.get("raw_output"))
                    key = (key_p, key_r)

                    if key in data_map[target_iter]:
                        continue

                    record = {
                        "source": "verl",
                        "attention_mask": row.get("attention_mask"),
                        "response_mask": row.get("response_mask"),
                        "position_ids": row.get("position_ids"),
                        "rm_scores": row.get("rm_scores"),
                        "token_level_rewards": row.get("token_level_rewards")
                    }
                    data_map[target_iter][key] = record

                except json.JSONDecodeError:
                    pass

    return data_map

# ==========================================
# 3. 对比逻辑
# ==========================================

def run_comparison(out_path, verl_dir):
    out_data = load_out_json_data(out_path)
    verl_data = load_verl_data(verl_dir)

    fields_to_check = [
        "attention_mask",
        "response_mask",
        "position_ids",
        "rm_scores",
        "token_level_rewards"
    ]

    total_checked = 0
    total_mismatches = 0

    sorted_iters = sorted(out_data.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for iter_id in sorted_iters:
        print(f"\n{'='*20} Iteration {iter_id} {'='*20}")

        if iter_id not in verl_data:
            print(f"Warning: Iteration {iter_id} missing in Verl data.")
            continue

        iter_out = out_data[iter_id]
        iter_verl = verl_data[iter_id]

        print(f"Keys in Out.json: {len(iter_out)}")
        print(f"Keys in Verl    : {len(iter_verl)}")

        # 计算交集和差集，方便调试
        keys_out = set(iter_out.keys())
        keys_verl = set(iter_verl.keys())
        common_keys = keys_out & keys_verl
        missing_in_verl = keys_out - keys_verl

        print(f"Matched Keys    : {len(common_keys)}")

        if len(missing_in_verl) > 0:
            print(f"MISSING: {len(missing_in_verl)} trajectories from out.json not found in Verl.")
            # 打印一个丢失的Key的示例，查看长度信息
            sample_key = list(missing_in_verl)[0]
            print(f"  -> Sample Missing Key Lengths: Prompt={len(sample_key[0])}, Response={len(sample_key[1])}")
            print(f"  -> Sample Response Tail: {sample_key[1][-10:]}") # 看看尾部是否还有特殊token

        for key in common_keys:
            rec_out = iter_out[key]
            rec_verl = iter_verl[key]
            total_checked += 1

            for field in fields_to_check:
                val_out = rec_out.get(field)
                val_verl = rec_verl.get(field)

                # 使用增强版对比函数
                match, msg = compare_arrays_detailed(val_out, val_verl, field)

                if not match:
                    total_mismatches += 1
                    # 打印精简但有用的报错
                    print(f"\n[MISMATCH] Iter {iter_id} | Field: {field}")
                    print(f"  Key ID (short): ...{str(key[1])[-50:]}") # 只打印Response Key的最后一部分作为标识
                    print(f"  Error Detail  : {msg}")

    print("\n" + "="*30)
    print(f"Comparison Finished.")
    print(f"Total Trajectories Checked: {total_checked}")
    print(f"Total Fields Mismatched   : {total_mismatches}")

# ==========================================
# 4. 执行入口
# ==========================================

if __name__ == "__main__":
    # 配置路径
    OUT_JSON_PATH = "/opt/DPC/models/z00943413/AgenticRL_5.0/AgenticRL_verl/origin_rollout/rollout_outputs_20260210_070021.json"
    VERL_LOG_DIR = "/opt/DPC/models/z00943413/AgenticRL_5.0/AgenticRL_verl/rollout_path_0210_v2"

    if os.path.exists(OUT_JSON_PATH) and os.path.exists(VERL_LOG_DIR):
        run_comparison(OUT_JSON_PATH, VERL_LOG_DIR)
    else:
        print("请修改脚本中的 OUT_JSON_PATH 和 VERL_LOG_DIR 为实际路径")
        print("未找到文件，无法执行对比。")