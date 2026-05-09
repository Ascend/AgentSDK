#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import pandas as pd
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime


def analyze_logs(log_file_path):
    # 1. Iterative indicator data (including all required indicators)
    iteration_metrics = [
        "timing/all", "timing/rollout", "timing/reference_model", "timing/update",
        "traj/llm_time_max", "traj/env_time_max",
        "response_length/max", "prompt_length/max"
    ]
    iter_metric_values = defaultdict(list)
    max_completed_iter = 0
    total_iterations = 0

    # 2. Data structure of other indicators
    app_total_metrics = ["total_llm_time", "total_env_time"]
    global_app_values = defaultdict(list)
    all_tpot_values = []
    tpot_step_values = defaultdict(list)
    tpot_details = []

    # 3. Detailed data structure of iteration
    iteration_details = []

    # 4. Configuration Information Data Structure
    config_info = {}
    target_config_keys = [
        'global_batch_size', 'seq_length', 'mini_batch_size',
        'max_num_seqs', 'max_model_len', 'max_num_batched_tokens',
        'gpu_memory_utilization', 'enable_prefix_caching'
    ]

    iter_progress_pattern = re.compile(r'iteration:\s*(\d+)\s*/\s*(\d+)')
    metric_pattern = re.compile(
        r'((?:timing|response_length|prompt_length|traj|grpo|tokens|vllm)/[^:\s]+)\s*:\s*([\d.-]+)'
    )
    traj_time_pattern = re.compile(
        r'traj/(llm_time|env_time)_(max|mean|min):\s*([\d.]+)'
    )
    config_pattern = re.compile(r"'([^']+)':\s*([^,}]+)")
    nested_config_pattern = re.compile(r"'([^']+)':\s*([^,}]+)")
    app_total_pattern = re.compile(r'appID:([0-9a-f-]+).*?total_llm_time:([\d.]+).*?total_env_time:([\d.]+)')
    llm_step_pattern = re.compile(r'appID:([0-9a-f-]+).*?step_idx:(\d+).*?llm_time:\s*([\d.]+)')
    response_step_pattern = re.compile(r'appID:([0-9a-f-]+).*?step_idx:(\d+).*?response_length:(\d+)')

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"错误：日志文件 {log_file_path} 不存在")
        return
    except UnicodeDecodeError:
        print(f"错误：日志文件编码不是UTF-8")
        return
    
    for match in iter_progress_pattern.finditer(log_content):
        completed = int(match.group(1))
        total = int(match.group(2))
        if completed > max_completed_iter:
            max_completed_iter = completed
        if total_iterations == 0:
            total_iterations = total
    
    for line in log_content.split('\n'):
        if iter_progress_pattern.search(line):
            iter_match = iter_progress_pattern.search(line)
            if iter_match:
                completed = int(iter_match.group(1))
                iteration_data = {"iteration": completed}

            metrics = metric_pattern.findall(line)
            for match in metrics:
                if len(match) != 2:
                    print(f"警告：跳过格式不匹配的指标：{match}")
                    continue
                metric_full, value_str = match
                if metric_full in iteration_metrics:
                    try:
                        value = float(value_str)
                        iter_metric_values[metric_full].append(value)
                        if metric_full in ["response_length/max", "prompt_length/max", "traj/llm_time_max"]:
                            iteration_data[metric_full] = value
                    except ValueError:
                        continue

            if len(iteration_data) > 1:
                iteration_details.append(iteration_data)
        
        traj_matches = traj_time_pattern.findall(line)
        for time_type, stat_type, value_str in traj_matches:
            metric_full = f"traj/{time_type}_{stat_type}"
            if metric_full in iteration_metrics:
                try:
                    value = float(value_str)
                    iter_metric_values[metric_full].append(value)

                    if metric_full == "traj/llm_time_max":
                        iter_match_in_line = re.search(r'iteration\s+(\d+)', line)
                        if iter_match_in_line:
                            iter_num = int(iter_match_in_line.group(1))
                            found = False
                            for record in iteration_details:
                                if record.get("iteration") == iter_num:
                                    record[metric_full] = value
                                    found = True
                                    break

                            if not found:
                                new_record = {"iteration": iter_num, metric_full: value}
                                iteration_details.append(new_record)
                except ValueError:
                    continue
        
        if "model':" in line or "megatron_training':" in line or "actor_config':" in line or "generate_config':" in line or "rl_config':" in line:
            config_matches = config_pattern.findall(line)
            for key, value in config_matches:
                clean_value = value.strip().strip("'\"")
                if clean_value and clean_value not in ['{', '}'] and key in target_config_keys:
                    config_info[key] = clean_value
    
    for app_id, llm_str, env_str in app_total_pattern.findall(log_content):
        try:
            global_app_values["total_llm_time"].append(float(llm_str))
            global_app_values["total_env_time"].append(float(env_str))
        except ValueError:
            continue
    
    llm_step_map = defaultdict(dict)
    for app_id, step_str, time_str in llm_step_pattern.findall(log_content):
        try:
            step_idx = int(step_str)
            llm_step_map[app_id][step_idx] = float(time_str)
        except (ValueError, TypeError):
            continue
    
    for app_id, step_str, length_str in response_step_pattern.findall(log_content):
        try:
            step_idx = int(step_str)
            response_length = int(length_str)
            if response_length == 0:
                continue
            if app_id in llm_step_map and step_idx in llm_step_map[app_id]:
                llm_time = llm_step_map[app_id][step_idx]
                tpot = llm_time / response_length
                tpot_details.append({
                    "appID": app_id, "step_idx": step_idx,
                    "llm_time_sec": round(llm_time, 6),
                    "response_length_tokens": response_length,
                    "tpot_sec_per_token": round(tpot, 6)
                })
                tpot_step_values[step_idx].append(tpot)
                all_tpot_values.append(tpot)
        except (ValueError, TypeError):
            continue
    
    has_valid_data = (any(iter_metric_values.values()) or
                      any(global_app_values.values()) or
                      bool(all_tpot_values) or
                      max_completed_iter > 0)
    if not has_valid_data:
        print("警告：没有有效数据用于分析")
        return
    
    iter_stats = {}
    for metric in iteration_metrics:
        values = iter_metric_values[metric]
        if values:
            iter_stats[metric] = {
                "avg": round(statistics.mean(values), 6),
                "max": round(max(values), 6),
                "min": round(min(values), 6),
                "cnt": len(values)
            }

    app_global_stats = {}
    for metric in app_total_metrics:
        values = global_app_values[metric]
        if values:
            app_global_stats[metric] = {
                "avg": round(statistics.mean(values), 6),
                "max": round(max(values), 6),
                "min": round(min(values), 6),
                "cnt": len(values)
            }
    
    tpot_global_stats = None
    if all_tpot_values:
        tpot_global_stats = {
            "avg": round(statistics.mean(all_tpot_values), 6),
            "max": round(max(all_tpot_values), 6),
            "min": round(min(all_tpot_values), 6),
            "cnt": len(all_tpot_values)
        }
    
    print("\n==== 配置信息 ====")
    if config_info:
        print("关键配置参数")
        for key in target_config_keys:
            if key in config_info:
                print(f"{key}: {config_info[key]}")
        if len(config_info) == 0:
            print("没有找到指定的配置参数")
    else:
        print("没有找到配置信息")
    
    print("\n==== 迭代进度信息 ====")
    print(f"最大已完成迭代轮数：{max_completed_iter}")
    print(f"总迭代轮数：{total_iterations if total_iterations > 0 else '未知'}")

    print("\n==== iteration粒度时间统计（秒） ====")
    if iter_stats:
        print(f"{'指标':<25} | 平均值       | 最大值        | 最小值        | 总样本数")
        print("-" * 85)
        for metric in iteration_metrics:
            if metric in iter_stats and (metric.startswith("timing/") or metric.startswith("traj/")):
                s = iter_stats[metric]
                print(f"{metric:<25} | {s['avg']:<12} | {s['max']:<12} | {s['min']:<12} | {s['cnt']}")
    else:
        print("没有有效迭代时间数据")
    
    print("\n==== iteration粒度长度统计（tokens) ====")
    if iter_stats:
        print(f"{'指标':<25} | 平均值       | 最大值        | 最小值        | 总样本数")
        print("-" * 85)
        for metric in iteration_metrics:
            if metric in iter_stats and (metric.startswith("response_length/") or metric.startswith("prompt_length/")):
                s = iter_stats[metric]
                print(f"{metric:<25} | {s['avg']:<12} | {s['max']:<12} | {s['min']:<12} | {s['cnt']}")
    else:
        print("没有有效长度数据")

    print("\n==== 单个step粒度TPOT时间统计（秒/token）=====")
    if tpot_global_stats:
        print(f"{'指标':<18} | 平均值       | 最大值        | 最小值        | 总样本数")
        print("-" * 75)
        print(f"{'TPOT':<18} | {tpot_global_stats['avg']:<12} | {tpot_global_stats['max']:<12} | {tpot_global_stats['min']:<12} | {tpot_global_stats['cnt']}")
    else:
        print("没有有效TPOT数据")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f"log_analysis_{timestamp}.xlsx"

    with pd.ExcelWriter(excel_file, engine='auto') as writer:
        if config_info:
            config_df = pd.DataFrame([
                {"配置项": key, "值": value} for key, value in config_info.items()
            ])
            config_df.to_excel(writer, sheet_name="配置信息", index=False)
        
        pd.DataFrame([{
            "最大已完成迭代轮数": max_completed_iter,
            "总迭代轮数": total_iterations if total_iterations > 0 else "未知"
        }]).to_excel(writer, sheet_name="迭代进度", index=False)

        if iter_stats:
            time_stats = [(m, s) for m, s in iter_stats.items() if m.startswith("timing/")]
            if time_stats:
                time_df = pd.DataFrame([
                    {"指标": m, "平均值": s["avg"], "最大值": s["max"], "最小值": s["min"], "总样本数": s["cnt"]}
                    for m, s in time_stats
                ])
                time_df.to_excel(writer, sheet_name="迭代时间统计", index=False)
        
        if iter_stats:
            length_stats = [(m, s) for m, s in iter_stats.items() if m.startswith("response_length/") or m.startswith("prompt_length/")]
            if length_stats:
                length_df = pd.DataFrame([
                    {"指标": m, "平均值": s["avg"], "最大值": s["max"], "最小值": s["min"], "总样本数": s["cnt"]}
                    for m, s in length_stats
                ])
                length_df.to_excel(writer, sheet_name="迭代长度统计", index=False)
        
        if iter_stats:
            other_stats = [(m, s) for m, s in iter_stats.items() if not (m.startswith("timing/") or m.startswith("response_length/") or m.startswith("prompt_length/"))]
            if other_stats:
                other_df = pd.DataFrame([
                    {"指标": m, "平均值": s["avg"], "最大值": s["max"], "最小值": s["min"], "总样本数": s["cnt"]}
                    for m, s in other_stats
                ])
                other_df.to_excel(writer, sheet_name="迭代其他指标统计", index=False)

        if all_tpot_values:
            pd.DataFrame([{
                "指标": "TPOT（秒/token）",
                "平均值": tpot_global_stats["avg"],
                "最大值": tpot_global_stats["max"],
                "最小值": tpot_global_stats["min"],
                "总样本数": tpot_global_stats["cnt"]
            }]).to_excel(writer, sheet_name="TPOT全局统计", index=False)
        
            step_stats = []
            for step_idx in sorted(tpot_step_values.keys()):
                values = tpot_step_values[step_idx]
                step_stats.append({
                    "step_idx": step_idx,
                    "平均值": round(statistics.mean(values), 6),
                    "最大值": round(max(values), 6),
                    "最小值": round(min(values), 6),
                    "样本数": len(values)
                })
            pd.DataFrame(step_stats).to_excel(writer, sheet_name="TPOT按step统计", index=False)
            pd.DataFrame(tpot_details).to_excel(writer, sheet_name="TPOT详细数据", index=False)
        
        if iteration_details:
            all_keys = set()
            for record in iteration_details:
                all_keys.update(record.keys())
        
            for record in iteration_details:
                for key in all_keys:
                    if key not in record:
                        record[key] = None
            
            iteration_details.sort(key=lambda x: x.get("iteration", 0))

            iter_df = pd.DataFrame(iteration_details)
            iter_df.to_excel(writer, sheet_name="iteration详细数据", index=False)
    
    print(f"\n✅️ 分析结果已保存到：{excel_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法：python log_analysis_tool.py <日志文件路径>")
        sys.exit(1)
    log_file = sys.argv[1]
    analyze_logs(log_file)