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
from collections import OrderedDict
import sys
import re
import glob


def merge_vllm_sched_stats(vllm_sched_paths):
    merged_requests = {}
    for vllm_path in vllm_sched_paths:
        with open(vllm_path) as f:
            data = json.load(f)
            requests = data.get('request', {})
            for request_id, request_data in requests.items():
                if request_id in merged_requests:
                    merged_requests[request_id].update(request_data)
                else:
                    merged_requests[request_id] = request_data
    return merged_requests


def convert_to_chrome_tracing(iteration_id, app_stats_path, vllm_sched_paths, pid):
    with open(app_stats_path) as f:
        app_stats = json.load(f)
    vllm_sched = merge_vllm_sched_stats(vllm_sched_paths)

    events = []
    events.append({
        "name": "process_name",
        "ph": "M",
        "pid": pid,
        "args": {"name": f"Iteration {iteration_id}"}
    })

    request_map = OrderedDict()
    for app_id, app_data in app_stats['applications'].items():
        sorted_requests = sorted(app_data['requests'], key=lambda x: x['step_idx'])
        for req in sorted_requests:
            key = (app_id, req['step_idx'])
            request_map[key] = {
                'route_start': req['route_tick'],
                'vllm_start': req['vllm_start_tick'],
                'env_start': req['env_start_tick'],
                'env_end': req['env_end_tick'],
                'request_id': req['request_id'],
                'address': req['address'],
                'prompt_len': req['prompt_len'],
                'terminal_reason': req['terminal_reason']
            }

    CATEGORY_COLORS = {
        "Route": "#FF6B6B",
        "vLLM-Schedule": "#4ECDC4",
        "vLLM-Prefill": "#45B7D1",
        "vLLM-Decode": "#334D5C",
        "Env-Execute": "#96CEB4",
    }    

    current_tid = 1000
    app_tid_map = {}

    for (app_id, step_idx), step_info in request_map.items():
        if app_id not in app_tid_map:
            app_tid_map[app_id] = current_tid
            current_tid += 1
        
        tid = app_tid_map[app_id]
        if vllm_sched.get(step_info['request_id']) is None:
            print(f"request {step_info['request_id']} is not exist")
            continue
    
        sched_data = vllm_sched.get(step_info['request_id'], {})

        vllm_attr = {'prompt_len': sched_data['prompt_len'], 'output_len': sched_data['output_len'], 'address': step_info['address']}

        phases = [
            ("Route",
            step_info['route_start'],
            sched_data.get('add_tick', step_info['route_start'] + 1),
            {'address': step_info['address'], 'prompt_len': step_info['prompt_len']}
            ),

            ("vLLM-Schedule",
            sched_data.get('add_tick', 0),
            sched_data.get('schedule_tick', 0),
            vllm_attr
            ),

            ("vLLM-Prefill",
            sched_data.get('schedule_tick', 0),
            sched_data.get('prefill_done_tick', 0),
            vllm_attr,
            ),

            ("vLLM-Decode",
            sched_data.get('prefill_done_tick', 0),
            sched_data.get('finish_tick', 0),
            vllm_attr,
            ),

            ("Env-Execute",
            step_info['env_start'],
            step_info['env_end'],
            {'terminal_reason': step_info['terminal_reason']}
            )
        ]

        for phase in phases:
            name, start, end, attr = phase
            if start <= 0 or end <= 0 or start >= end:
                continue

            events.append({
                "name": name,
                "cat": "STAGES",
                "ph": "X",
                "ts": start * 1e6,
                "dur": (end - start) * 1e6,
                "pid": pid,
                "tid": tid,
                "args": {
                    "app": app_id[:8],
                    "step": step_idx,
                    "phase": name,
                    "color": CATEGORY_COLORS[name],
                    "attrs": attr,
                }
            })

    for app_id, tid in app_tid_map.items():
        events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": tid,
            "args": {"name": f"App: {app_id}"}
        })
        events.append({
            "name": "thread_sort_index",
            "ph": "M",
            "pid": pid,
            "tid": tid,
            "args": {"sort_index": tid}
        })

    return events


def get_iteration_id(app_file: str):
    return re.findall(r"iter_(\d+)", app_file)[0]


def single_iteration_parse():
    app_json = sys.argv[1]
    vllm_sched_files = sys.argv[2]
    iteration_id = get_iteration_id(app_json)
    output_json = "trajectory_trace_iter_" + iteration_id + ".json"
    events = convert_to_chrome_tracing(
        iteration_id, 
        app_json, 
        glob.glob(vllm_sched_files)
    )
    with open(output_json, 'w') as f:
        json.dump(events, f, indent=2)


def multi_iteration_parse():
    file_iter_pairs = []
    for app_json in glob.glob("app_stats_iter_*.json"):
        match = re.search(r'iter_(\d+)', app_json)
        if match:
            iteration_id = int(match.group(1))
            file_iter_pairs.append((iteration_id, app_json))
    
    file_iter_pairs.sort(key=lambda x: x[0])

    all_events = []
    pid_counter = 1

    for iteration_id, app_json in file_iter_pairs:
        print(f"Processing iteration {iteration_id} (PID:{pid_counter})")
        vllm_files = glob.glob(f"vllm_schedule_{iteration_id}_*.json")
        if not vllm_files:
            print(f"No vLLM files found for iteration {iteration_id}")
            continue
        
        try:
            events = convert_to_chrome_tracing(
                iteration_id=str(iteration_id), 
                app_stats_path=app_json, 
                vllm_sched_paths=vllm_files,
                pid=pid_counter
            )
            all_events.extend(events)
            pid_counter += 1
        except Exception as e:
            print(f"Error processing iteration {iteration_id}: {str(e)}")
            continue
    
    all_events.sort(key=lambda x: x.get('ts', 0))

    with open("trajectory_tracing.json", 'w') as f:
        json.dump(all_events, f, indent=2)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        single_iteration_parse()
    else:
        multi_iteration_parse()
