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


import glob
import json
import math
import os
import pandas as pd
import re
import sys
from collections import defaultdict
from datetime import datetime


def convert_tick(tick):
    return datetime.fromtimestamp(tick)


def build_timeline(data):
    timeline = []

    for addr, inst in data['schedulers'].items():
        for req_id in inst['requests']:
            if data['request'].get(req_id) is None:
                print(f"req: {req_id} not exist in data")
                continue
            req = data['request'][req_id]
            timeline.append({
                "address": addr,
                "start": convert_tick(req['schedule_tick']),
                "end": convert_tick(req['finish_tick']),
                "prompt_tokens": req['prompt_len'],
                "output_tokens": req['output_len']
            })
    return pd.DataFrame(timeline)


def plot_scheduler_load(df, file_path):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.figure(figsize=(20, 12))

    ax1 = plt.subplot(3, 1, 1)
    groups = df.groupby('address')
    for name, group in groups:
        time_points = sorted(set(group['start']) | set(group['end']))
        counts = [((group['start'] <= t) & (t < group['end'])).sum() for t in time_points]
        ax1.step(time_points, counts, where='post', label=name, linewidth=1.5)
    
    ax1.set_title('Requests-Num@vLLM')
    ax1.set_ylabel('Request Count')
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    ax2 = plt.subplot(3, 1, 2)
    for name, group in groups:
        events = []
        for _, row in group.iterrows():
            events.append((row['start'], 'start', row['prompt_tokens']))
            events.append((row['end'], 'end', row['prompt_tokens']))
        events.sort()
    
        time_points, token_rates = [], []
        current_tokens = 0
        for t, typ, tokens in events:
            time_points.append(t)
            token_rates.append(current_tokens)
            current_tokens += tokens if typ == 'start' else -tokens
        
        ax2.fill_between(time_points, token_rates, alpha=0.2, label=name)
        ax2.plot(time_points, token_rates, linewidth=0.5)
    
    ax2.set_title('Prompt-Tokens@vLLM')
    ax2.set_ylabel('Prompt Tokens')
    ax2.set_xlabel('Time')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    ax3 = plt.subplot(3, 1, 3)
    for name, group in groups:
        events = []
        for _, row in group.iterrows():
            events.append((row['start'], 'start', row['output_tokens']))
            events.append((row['end'], 'end', row['output_tokens']))
        events.sort()
    
        time_points, token_rates = [], []
        current_tokens = 0
        for t, typ, tokens in events:
            time_points.append(t)
            token_rates.append(current_tokens)
            current_tokens += tokens if typ == 'start' else -tokens
        
        ax3.fill_between(time_points, token_rates, alpha=0.2, label=name)
        ax3.plot(time_points, token_rates, linewidth=0.5)
    
    ax3.set_title('Output-Tokens@vLLM')
    ax3.set_ylabel('Output Tokens')
    ax3.set_xlabel('Time')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S'))
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def merge_vllm_sched_stats(vllm_sched_paths, iteration_id):
    merged_requests = {}
    for vllm_path in vllm_sched_paths:
        with open(vllm_path) as f:
            data = json.load(f)
            for req_id, req_data in data.get("request", {}).items():
                merged_requests[req_id] = req_data
    return merged_requests


def analysis_vllm_stats(app_data_path):
    FILE_PATTERN = os.path.join(app_data_path, "vllm_schedule_*.json")
    all_requests_data = []
    stats_counters = {
        "total_files": 0,
        "total_requests": 0,
        "processed_requests": 0,
        "skipped_missing_ticks": 0,
        "skipped_invalid_calc": 0,
        "skipped_extreme_values": 0
    }

    MAX_REASONABLE_TIME_DIFF = 3600.0
    MAX_REASONABLE_LEN = 128000
    for file_path in glob.glob(FILE_PATTERN):
        stats_counters["total_files"] += 1
        if stats_counters["total_files"] % 50 == 0:
            print(f"已处理 {stats_counters['total_files']} 个文件...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                requests_in_file = data.get("request", {})
                stats_counters["total_requests"] += len(requests_in_file)

                for req_id, req_details in requests_in_file.items():
                    add_tick = req_details.get("add_tick")
                    schedule_tick = req_details.get("schedule_tick")
                    prefill_done_tick = req_details.get("prefill_done_tick")
                    finish_tick = req_details.get("finish_tick")

                    if None in (add_tick, schedule_tick, prefill_done_tick, finish_tick):
                        stats_counters["skipped_missing_ticks"] += 1
                        continue
                    
                    ticks = [add_tick, schedule_tick, prefill_done_tick, finish_tick]
                    if not all(isinstance(t, (int, float)) and not (math.isnan(t) or math.isinf(t)) for t in ticks):
                        stats_counters["skipped_invalid_calc"] += 1
                        continue
                    
                    if not (add_tick <= schedule_tick <= prefill_done_tick <= finish_tick):
                        pass
                    
                    prompt_len = req_details.get("prompt_len")
                    output_len = req_details.get("output_len")
                    try:
                        sched = schedule_tick - add_tick
                        ttft = prefill_done_tick - schedule_tick
                        tpot = (finish_tick - prefill_done_tick) / output_len
                    except Exception as calc_error:
                        stats_counters["skipped_invalid_calc"] += 1
                        continue
                    
                    calc_results = [sched, ttft, tpot]
                    if not all(isinstance(res, (int, float)) and not (math.isnan(res) or math.isinf(res)) for res in calc_results):
                        stats_counters["skipped_invalid_calc"] += 1
                        continue

                    if abs(tpot) > 5.0:
                        stats_counters["skipped_extreme_values"] += 1
                        continue

                    if any(abs(res) > MAX_REASONABLE_TIME_DIFF for res in calc_results):
                        stats_counters["skipped_extreme_values"] += 1
                        continue

                    lens = [prompt_len, output_len]
                    if not all(isinstance(l, int) and 0 <= l <= MAX_REASONABLE_LEN for l in lens):
                        stats_counters["skipped_extreme_values"] += 1
                        continue
                    
                    request_summary = {
                        "file": os.path.basename(file_path),
                        "request_id": req_id,
                        "prompt_len": prompt_len,
                        "output_len": output_len,
                        "SCHED": sched,
                        "TTFT": ttft,
                        "TPOT": tpot,
                    }
                    all_requests_data.append(request_summary)
                    stats_counters["processed_requests"] += 1
        except json.JSONDecodeError as e:
            print(f"警告：无法解析文件 {file_path}: {e}")
        except FileNotFoundError:
            print(f"警告：找不到文件 {file_path}")
        except Exception as e:
            print(f"警告：处理文件 {file_path} 时发生未知错误: {e}")
    
    print("\n--- 数据处理统计 ---")
    for key, value in stats_counters.items():
        print(f"  {key}: {value}")
    
    if not all_requests_data:
        print("\n未找到任何有效请求数据。请检查目录路径、文件内容和过滤条件。")
    else:
        print(f"\n成功处理并用于统计的请求数量: {len(all_requests_data)}")

        metrics_to_analyze = ["prompt_len", "output_len", "SCHED", "TTFT", "TPOT"]

        stats_data = defaultdict(list)
        for req in all_requests_data:
            for metric in metrics_to_analyze:
                stats_data[metric].append(req[metric])
        
        print("\n--- 所有有效请求的统计摘要 ---")
        for metric in metrics_to_analyze:
            values = stats_data[metric]
            if values:
                avg_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                print(f"{metric}:")
                print(f"  平均值（Average）: {avg_val:.6f}")
                print(f"  最小值（Min）: {min_val:.6f}")
                print(f"  最大值（Max）: {max_val:.6f}")
            else:
                print(f"{metric}: 没有可用数据进行统计。")
            print("-" * 30)


if __name__ == "__main__":
    app_data_path = sys.argv[1]
    analysis_vllm_stats(app_data_path)
    app_stats_path = os.path.join(app_data_path, "app_stats_iter_*.json")
    json_files = sorted(glob.glob(app_stats_path),
                        key=lambda x: int(re.search(r'iter_(\d+)', x).group(1)))
    combined_data = {
        "schedulers": {},
        "request": {},
    }
    pattern = re.compile(r'app_stats_iter_(\d+)_')

    file_path = "./"
    for app_json in json_files:
        match = pattern.search(app_json)
        if not match: continue

        iteration_id = match.group(1)
        print(f"Processing iteration {iteration_id}")
        file_path = os.path.dirname(app_json)
        vllm_files = glob.glob(f"{file_path}//vllm_schedule_{iteration_id}_*.json")

        with open(app_json) as f:
            app_stats = json.load(f)
            for addr, sched in app_stats["schedulers"].items():
                if addr not in combined_data["schedulers"]:
                    combined_data["schedulers"][addr] = {"requests": []}
                combined_data["schedulers"][addr]["requests"].extend(sched["requests"])
        
        requests = merge_vllm_sched_stats(vllm_files, iteration_id)
        combined_data["request"].update(requests)
    
    full_timeline = build_timeline(combined_data)
    full_timeline.sort_values('start', inplace=True)
    plot_scheduler_load(full_timeline, f"{file_path}//combined_scheduler_load.png")
    print("Combined visualization saved to combined_scheduler_load.png")