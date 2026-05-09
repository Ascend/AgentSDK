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


import argparse
import asyncio
import numpy as np
import os
import pandas as pd
import re
import sys
from functools import partial
from typing import Dict, Any
from typing import Dict, List, Union, Tuple

"""
Parameters:
- appid: Application ID
- response_length_tokens: Number of response tokens
- llm_time_sec: LLM processing time (seconds)
- tpot_sec_per_token: Processing time per token (seconds/token)
"""
def argumentParse():
    parser = argparse.ArgumentParser(
        description="LLM性能分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python %(prog)s --appid myapp --tokens 1000 --time 5.2 --tpot 0.005
  python %(prog)s -a test_app -t 500 -l 2.5 -p 0.003
        """
    )

    # Add arguments
    parser.add_argument(
        "--appid",
        type=str,
        required=False,
        help="应用程序ID"
    )

    parser.add_argument(
        "--response_length_tokens",
        type=int,
        required=False,
        help="响应token数量",
        dest="response_length_tokens"
    )

    parser.add_argument(
        "--llm_time_sec",
        type=float,
        required=False,
        help="LLM处理时间（秒）",
        dest="llm_time_sec"
    )

    parser.add_argument(
        "--tpot_sec_per_token",
        type=float,
        required=False,
        help="每个token的处理时间（秒/token）",
        dest="tpot_sec_per_token"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="log存放的路径",
        dest="data_path"
    )

    parser.add_argument(
        "--rollout_log_analysis_file",
        type=str,
        required=False,
        help="rollout log_analysis表格存放的路径",
        dest="rollout_log_analysis_file"
    )

    parser.add_argument(
        "--rollout_logs_file",
        type=str,
        required=False,
        help="rollout logs.txt存放的路径",
        dest="rollout_logs_file"
    )

    parser.add_argument(
        "--is_profiling",
        type=bool,
        required=False,
        help="是否开启了profiling采集",
        dest="is_profiling",
        default=False
    )

    global _is_profiling
    _is_profiling = parser.parse_args().is_profiling

    # Parse arguments
    return parser.parse_args()


# Parse the time consumption of each rollout iteration from rollout logs
def parse_rollout_time(text_content):
    """
    Parse logs using a simple method
    """
    matches = []

    # Process text line by line
    lines = text_content.split('\n')
    for line in lines:
        # Check if the line contains handle_full_batch_trajectories and specific patterns
        if 'handle_full_batch_trajectories' in line and '===rollout iteration:' in line and 'timing/rollout :' in line:
            # Extract iteration
            iteration_match = re.search(r'===rollout iteration:\s*(\d+)', line)
            # Extract timing
            timing_match = re.search(r'timing/rollout\s*:\s*([\d\.]+)', line)

            if iteration_match and timing_match:
                iteration = int(iteration_match.group(1))
                timing = float(timing_match.group(1))
                matches.append({
                    'iteration': iteration,
                    'rollout_time(s)': timing
                })

    return matches

def parse_e2e_rollout_time_from_log_file(log_file: str, data_path: str):
    """
    Parse rollout data from log file
    """
    # Check if the file exists
    if not os.path.exists(log_file):
        print(f"错误: 文件 {log_file} 不存在")
        return None

    # Read file content
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except UnicodeDecodeError:
        # Try other encodings
        with open(log_file, 'r', encoding='gbk') as f:
            text_content = f.read()

    matches = parse_rollout_time(text_content)

    if not matches:
        print("未找到匹配的rollout数据")
        return None

    print(f"找到 {len(matches)} 条匹配的rollout数据")

    # Create DataFrame
    df = pd.DataFrame(matches)
    df = df.sort_values('iteration').reset_index(drop=True)

    # Calculate summary statistics
    total_time = df['rollout_time(s)'].sum()
    avg_time = df['rollout_time(s)'].mean()
    max_time = df['rollout_time(s)'].max()
    min_time = df['rollout_time(s)'].min()

    # Add summary row
    summary_row = pd.DataFrame({
        'iteration': ['SUMMARY'],
        'rollout_time(s)': [total_time]
    })

    df_with_summary = pd.concat([df, summary_row], ignore_index=True)

    # Export to Excel
    output_file = f"{data_path}/{os.path.basename(data_path)}_e2e_rollout_time_analysis.xlsx"
    df_with_summary.to_excel(output_file, index=False)

    print(f"\n数据已导出到: {output_file}")
    print("\n解析结果:")
    print(df_with_summary.to_string())

    return total_time


def filter_worker_pid(filename):
    # Match all content from the beginning to before the timestamp
    pattern = r'^(.+?)-\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.[a-zA-Z0-9]+$'
    result = re.search(pattern, filename)

    if result:
        extracted = result.group(1)
        return extracted  # Output: 10.50.89.104 IntegratedWorker pid=43131
    return filename

def load_vllm_stats_file_to_dict(data_path):
    """
    Traverse all CSV files in the specified folder, read them as DataFrames and store them in a dictionary

    Parameters:
    data_path (str): Folder path containing CSV files

    Returns:
    dict: Key is filename, value is the corresponding DataFrame
    """
    file_dict = {}

    # Check if the path exists
    if not os.path.exists(data_path):
        print(f"错误: 路径 '{data_path}' 不存在")
        return file_dict

    # Supported file extensions
    supported_extensions = ['.csv', '.xlsx', '.xls']

    # Traverse all files in the folder
    for filename in os.listdir(data_path):
        if "IntegratedWorker" not in filename:
            continue
        # Get file extension
        file_ext = os.path.splitext(filename)[1].lower()

        # Check if it is a supported file type
        if file_ext in supported_extensions:
            # Build full file path
            file_path = os.path.join(data_path, filename)

            try:
                # Select reading method according to file type
                if file_ext == '.csv':
                    df = pd.read_csv(file_path)
                    file_type = "CSV"
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                    file_type = "Excel"
                else:
                    continue  # Will not be executed theoretically

                # Store DataFrame in the dictionary with filename as key
                worker_pid = filter_worker_pid(filename)
                if worker_pid in file_dict:
                    # Always try vertical concatenation
                    combined_df = pd.concat([file_dict[worker_pid], df], ignore_index=True, sort=False)
                    file_dict[worker_pid] = combined_df
                    print(f"已追加: {filename} 到已存在的键 '{worker_pid}' (合并后包含 {len(combined_df)} 行)")
                else:
                    file_dict[worker_pid] = df
                    print(f"已加载: {filename} ({file_type}, 包含 {len(df)} 行, {len(df.columns)} 列)")

            except Exception as e:
                print(f"错误: 无法读取文件 {filename} - {e}")

    # Print summary information
    if file_dict:
        print(f"\n总共加载了 {len(file_dict)} 个文件")
    else:
        print("未找到任何CSV或Excel文件")

    return file_dict

def write_appid_statistic_to_file(results, filename):
    # Create DataFrame
    df_data = []

    for key, value in results.items():
        if isinstance(value, list):
            # If it is a list, put key in the first column, and each element of the list in the following columns in turn
            row_data = [key] + value
        else:
            # If it is not a list, put key in the first column and value in the second column
            row_data = [key, value]
        df_data.append(row_data)

    # Find the longest row to set the number of columns
    max_len = max(len(row) for row in df_data)

    # Ensure all rows have the same length (fill with None)
    for row in df_data:
        if len(row) < max_len:
            row.extend([None] * (max_len - len(row)))

    # Create DataFrame
    df = pd.DataFrame(df_data)

    # Write to Excel file
    df.to_excel(filename, index=False, header=False)

    print(f"数据已写入 {filename}")

def load_topts_from_rollout_log_to_df(file_path):
    # Read the specified sheet in the Excel file
    sheet_name = 'TPOT详细数据'
    # Read data
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'，请确认文件路径是否正确")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")

def load_rollout_e2etime_from_rollout_log_to_df(file_path):
    # Read the specified sheet in the Excel file
    sheet_name = '迭代时间统计'
    # Read data
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        # print("load_rollout_e2etime_from_rollout_log_to_df df:",df)
        return df['最小值'].sum()
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'，请确认文件路径是否正确")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")

def parse_profiling_data_robust(worker_df: pd.DataFrame) -> Dict[str, str]:
    """
    Parse data where is_profiling is TRUE from the DataFrame
    Return a mapping dictionary from (ip, pid, step_id) to (with_prefill, is_dummy_run)
    Can handle mixed types of strings and booleans
    """
    # Precompile regular expressions
    IP_PATTERN = re.compile(r'^([\d\.]+)')
    PID_PATTERN = re.compile(r'pid=(\d+)')
    STEP_ID_PATTERN = re.compile(r'/(\d+)$')

    # Unify the type of the is_profiling column: convert to lowercase string for comparison
    # This is much faster than calling the to_bool function row by row
    if worker_df['is_profiling'].dtype == 'object':
        is_profiling_mask = worker_df['is_profiling'].astype(str).str.lower().eq('true')
    else:
        is_profiling_mask = worker_df['is_profiling'].astype(bool)

    # Filter rows that need to be processed
    profiling_df = worker_df[is_profiling_mask].copy()

    if profiling_df.empty:
        return {}

    # Vectorized extraction of information from title
    titles = profiling_df['title'].astype(str)

    # Extract IP - using vectorized operation
    ip_matches = titles.str.extract(IP_PATTERN, expand=False)
    ips = ip_matches.fillna('unknown')

    # Extract PID - using vectorized operation
    pid_matches = titles.str.extract(PID_PATTERN, expand=False)
    pids = pid_matches.fillna('unknown')

    # Extract step_id - using vectorized operation
    step_id_matches = titles.str.extract(STEP_ID_PATTERN, expand=False)
    step_ids = step_id_matches.fillna('unknown')

    # Convert with_prefill and is_dummy_run to boolean values
    # Using vectorized operation
    def vectorized_to_bool(series: pd.Series) -> pd.Series:
        if series.dtype == 'object':
            return series.astype(str).str.lower().eq('true')
        return series.astype(bool)

    with_prefill = vectorized_to_bool(profiling_df['with_prefill'])
    is_dummy_run = vectorized_to_bool(profiling_df['is_dummy_run'])

    # Build result dictionary - using zip and loop (faster than iterrows)
    result = {}
    for ip, pid, step_id, wp, idr in zip(ips, pids, step_ids, with_prefill, is_dummy_run):
        key = f"{ip}_{pid}_{step_id}"
        result[key] = f"{wp}_{idr}"
        # key = (ip, pid, step_id)
        # result[key] = (wp, idr)

    return result

def analyze_profiling_step_info(
        vllm_stats_file_dict: dict,
        data_path: str
):
    if not _is_profiling:
        return

    all_profiling_steps_dict = {}
    for worker in vllm_stats_file_dict:
        worker_dict = vllm_stats_file_dict[worker]
        profiling_steps_dict = parse_profiling_data_robust(worker_dict)
        all_profiling_steps_dict.update(profiling_steps_dict)

    last_dir = os.path.basename(data_path)
    result_file = f"{data_path}/{last_dir}_profiling_steps_info.csv"
    df = pd.DataFrame(list(all_profiling_steps_dict.items()), columns=['Key_Tuple', 'Value_Tuple'])
    df.to_csv(result_file, index=False)
    print(f"数据已成功输出到 {result_file}")

def combine_appid_info(
        response_length_tokens: int,
        llm_time_sec: float,
        tpot_sec_per_token: float
):
    return f"response_length_tokens {response_length_tokens} / llm_time_sec {int(llm_time_sec * 1000)} ms / tpot_sec_per_token  {int(tpot_sec_per_token * 1000)} ms"

# Precompile with f-string (automatically optimized in Python 3.12+)
def generate_work_tpot_detail(filtered_appid_df):
    result_list = []
    # Predefine function to reduce attribute lookups
    get_type = lambda x: 'p' if x else 'd'

    for row in filtered_appid_df.itertuples():
        result_list.append(
            f"t{row.Index}-{get_type(row.with_prefill)}-b{row.batch_num} {int(row.step_total_time)}"
        )
    return result_list


def generate_worker_dict_lookup_map(worker_dict):
    # Create a mapping from (title, start time) to index
    # Assume the combination of title and time can uniquely identify the record
    lookup_map = {}
    for i in range(0, len(worker_dict)):
        record = worker_dict.iloc[i]
        key = (record['title'], record['step_start_time'])
        # print("key:", key)
        lookup_map[key] = i
    return lookup_map

def generate_appid_wait_detail(filtered_appid_df, worker_dict):
    lookup_map = generate_worker_dict_lookup_map(worker_dict)
    results = [None]
    for i in range(1, len(filtered_appid_df)):
        current_record = filtered_appid_df.iloc[i]
        previous_record = filtered_appid_df.iloc[i-1]

        # 1) Time interval
        time_interval = int((current_record['step_start_time'] - previous_record['step_finished_time']) * 1000) # s -> ms

        # 2) Lookup index
        current_key = (current_record['title'], current_record['step_start_time'])
        previous_key = (previous_record['title'], previous_record['step_start_time'])
        # print("current_key:", current_key, " previous_key:", previous_key)

        if current_key in lookup_map and previous_key in lookup_map:
            current_idx = lookup_map[current_key]
            previous_idx = lookup_map[previous_key]
            # print("previous_idx:", previous_idx, " current_idx:", current_idx)

            # Determine start and end indexes
            start_idx = min(current_idx, previous_idx)
            end_idx = max(current_idx, previous_idx)
            # print("start_idx:", start_idx, " end_idx:", end_idx)

            # Calculate the number of intermediate records and the sum of batch_num
            index_gap = end_idx - start_idx - 1
            batch_sum = 0

            # Calculate the sum of batch_num for intermediate records
            for j in range(start_idx + 1, end_idx):
                if j < len(worker_dict):
                    batch_sum += worker_dict.iloc[j].get('batch_num', 0)

            # Mark if the order is reversed
            if current_idx < previous_idx:
                index_gap = -index_gap  # Negative number indicates reversed order

            result_item = f"time-{time_interval} step-{index_gap} batch-{batch_sum}"
            results.append(result_item)
        else:
            results.append(None)
    return results

def generate_appid_wait_detail_optimized(filtered_appid_df, worker_dict):
    """Further optimized version, suitable for large data volumes"""
    lookup_map = generate_worker_dict_lookup_map(worker_dict)
    n = len(filtered_appid_df)
    results = [None] * n

    # Use itertuples() instead of iloc for faster speed
    records = list(filtered_appid_df.itertuples(index=False, name='Record'))

    # Precompute batch_num prefix sum - using vectorized operation
    # Assume worker_dict has a 'batch_num' column
    if 'batch_num' in worker_dict.columns:
        batch_nums = worker_dict['batch_num'].fillna(0).values
    else:
        batch_nums = [0] * len(worker_dict)

    # Calculate prefix sum with numpy, use loop if numpy is not available
    try:
        import numpy as np
        batch_prefix_sum = np.zeros(len(worker_dict) + 1)
        batch_prefix_sum[1:] = np.cumsum(batch_nums)
        batch_prefix_sum = batch_prefix_sum.tolist()
    except ImportError:
        # Use pure Python if numpy is not available
        batch_prefix_sum = [0] * (len(worker_dict) + 1)
        for i in range(1, len(worker_dict) + 1):
            batch_prefix_sum[i] = batch_prefix_sum[i-1] + batch_nums[i-1]

    lookup_map_get = lookup_map.get

    for i in range(1, n):
        current = records[i]
        previous = records[i-1]

        time_interval = int((current.step_start_time - previous.step_finished_time) * 1000)

        # Use frozenset or custom key for faster lookup (if applicable)
        current_key = (current.title, current.step_start_time)
        previous_key = (previous.title, previous.step_start_time)

        current_idx = lookup_map_get(current_key)
        previous_idx = lookup_map_get(previous_key)

        if current_idx is not None and previous_idx is not None:
            start_idx = min(current_idx, previous_idx)
            end_idx = max(current_idx, previous_idx)

            index_gap = end_idx - start_idx - 1
            batch_sum = batch_prefix_sum[end_idx] - batch_prefix_sum[start_idx + 1]

            if current_idx < previous_idx:
                index_gap = -index_gap

            # Pre-allocate string buffer
            results[i] = f"time-{time_interval} step-{index_gap} batch-{batch_sum}"

    return results

def tpot_details_post_proc(final_tpot_details_df):
    # Calculate statistics
    mean_row = final_tpot_details_df.mean(numeric_only=True).to_frame().T
    min_row = final_tpot_details_df.min(numeric_only=True).to_frame().T
    max_row = final_tpot_details_df.max(numeric_only=True).to_frame().T

    # Add identifiers
    mean_row['appID'] = 'mean'
    min_row['appID'] = 'min'
    max_row['appID'] = 'max'

    # Reorder columns to put appID in the first column
    cols = ['appID'] + [col for col in final_tpot_details_df.columns if col != 'appID']
    mean_row = mean_row[cols]
    min_row = min_row[cols]
    max_row = max_row[cols]

    # Keep 2 decimal places for numeric values in mean_row, min_row, max_row
    # Get numeric columns
    numeric_cols = final_tpot_details_df.select_dtypes(include=[np.number]).columns.tolist()

    # Keep 2 decimal places for numeric columns in mean_row
    for col in numeric_cols:
        if col in mean_row.columns:
            mean_row[col] = mean_row[col].round(2)

    # Keep 2 decimal places for numeric columns in min_row
    for col in numeric_cols:
        if col in min_row.columns:
            min_row[col] = min_row[col].round(2)

    # Keep 2 decimal places for numeric columns in max_row
    for col in numeric_cols:
        if col in max_row.columns:
            max_row[col] = max_row[col].round(2)

    # Merge into the original DataFrame
    df_with_stats = pd.concat([final_tpot_details_df, mean_row, min_row, max_row], ignore_index=True)

    # Now calculate the percentage ratio based on the mean row
    # Get the index of the mean row (the third last row)
    mean_idx = len(df_with_stats) - 3

    # Columns that need to calculate percentage
    ratio_columns = [
        'prepare_input_time',
        'aclgraph_dispatcher_time',
        'forward_time',
        'kvconnectoroutput_time',
        'post_process_time',
        'pop_captured_sync_time',
        'post_process_compute_logits_time',
        'post_process_sampler_time',
        'post_process_other_time'
    ]
    # Denominator column
    denominator_column = 'step_total_time'

    # Create ratio row (copy the mean row first, then modify)
    ratio_row = df_with_stats.loc[mean_idx].copy().to_dict()

    # Fix: get value directly from dictionary to avoid Series issues
    if denominator_column in ratio_row:
        denominator_value = ratio_row[denominator_column]

        if denominator_value != 0:
            for col in ratio_columns:
                if col in ratio_row:
                    # Calculate percentage (%)
                    ratio_row[col] = (ratio_row[col] / denominator_value) * 100

        # Set other numeric columns (except appID and ratio_columns) to 0
        numeric_cols = final_tpot_details_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in ratio_columns and col in ratio_row:
                ratio_row[col] = 0

    # Modify appID identifier
    ratio_row['appID'] = 'ratio(%)'

    # Ensure all numeric columns in ratio_row keep 2 decimal places
    for col in numeric_cols:
        if col in ratio_row:
            # Keep 2 decimal places if it is a float
            if isinstance(ratio_row[col], (int, float)):
                ratio_row[col] = round(ratio_row[col], 2)

    # Add the ratio row to the DataFrame
    df_with_stats = pd.concat([df_with_stats, pd.DataFrame([ratio_row])], ignore_index=True)

    return df_with_stats

# Generate detailed TPOT information for each appid
def generate_appid_topt_detail(appid, response_length_tokens, llm_time_sec, tpot_sec_per_token, appid_df):
    stat_appid_df = {}
    # If profiling is enabled, filter out data where step_total_time > 1000
    if _is_profiling:
        # filtered_appid_df_decode_no_prorfiling = appid_df[appid_df['is_profiling'] != False]
        stat_appid_df = appid_df[appid_df['step_total_time'] <= 500]
    else:
        stat_appid_df = appid_df
    filtered_appid_df_prefill = stat_appid_df[stat_appid_df['with_prefill'] == True]
    ttft = 0
    if len(filtered_appid_df_prefill) > 0:
        prefill_total_time = int(filtered_appid_df_prefill['step_total_time'].sum())
        ttft = prefill_total_time / len(filtered_appid_df_prefill)
    filtered_appid_df_decode = stat_appid_df[stat_appid_df['with_prefill'] == False]
    decode_total_time = int(filtered_appid_df_decode['step_total_time'].sum())
    tpot = decode_total_time / len(filtered_appid_df_decode)

    # Select numeric columns (exclude string columns)
    numeric_cols = stat_appid_df.select_dtypes(include=['float64', 'int64']).columns
    stats_df = stat_appid_df[numeric_cols].describe().loc[['mean', 'min', 'max']]
    tpot_with_prefill = stats_df.loc['mean', 'step_total_time']
    tpot_ms_per_token = tpot_sec_per_token * 1000
    # print(stats_df)
    # Count the number of rows that satisfy both conditions
    decode_with_prefill_true_count = len(appid_df[(appid_df['attn_state'] == 'AscendAttentionState.DecodeOnly') & (appid_df['with_prefill'] == 'TRUE')])

    tpot_detail = pd.DataFrame({
        'appID': appid,
        'llm_time_sec': llm_time_sec,
        'response_length_tokens': response_length_tokens,
        'tpot_sec_per_token': tpot_sec_per_token,
        'tpot_ms_per_token': tpot_ms_per_token,
        'vllm_ttft_ms': ttft,
        'vllm_tpot_ms': tpot,
        'vllm_tpot_with_prefill_ms': tpot_with_prefill,
        'rollout_vllm_tpot_gap_ms': tpot_ms_per_token - tpot_with_prefill,
        'prepare_input_time': stats_df.loc['mean', 'prepare_input_time'],
        'aclgraph_dispatcher_time': stats_df.loc['mean', 'aclgraph_dispatcher_time'],
        'forward_time': stats_df.loc['mean', 'forward_time'],
        'kvconnectoroutput_time': stats_df.loc['mean', 'kvconnectoroutput_time'],
        'post_process_time': stats_df.loc['mean', 'post_process_time'],
        'pop_captured_sync_time': stats_df.loc['mean', 'pop_captured_sync_time'],
        'step_total_time': stats_df.loc['mean', 'step_total_time'],
        'step_inter_time': stats_df.loc['mean', 'step_inter_time'],
        'post_process_compute_logits_time': stats_df.loc['mean', 'post_process_compute_logits_time'],
        'post_process_sampler_time': stats_df.loc['mean', 'post_process_sampler_time'],
        'post_process_other_time': stats_df.loc['mean', 'post_process_other_time'],
        'decode_with_prefill_true_count': decode_with_prefill_true_count
    }, index=[0])
    return tpot_detail

def analyze_vllm_statistic_sync(
        vllm_stats_file_dict: dict,
        appid: str,
        response_length_tokens: int,
        llm_time_sec: float,
        tpot_sec_per_token: float,
        data_path: str
):
    results = {
        appid: combine_appid_info(response_length_tokens, llm_time_sec, tpot_sec_per_token)
    }
    all_tpot_details = []
    for worker in vllm_stats_file_dict:
        worker_dict = vllm_stats_file_dict[worker]
        filtered_appid_df = worker_dict[worker_dict['title'].str.contains(appid)]
        if len(filtered_appid_df) == 0:
            continue
        tpot_detail = generate_appid_topt_detail(appid, response_length_tokens, llm_time_sec, tpot_sec_per_token, filtered_appid_df)
        all_tpot_details.append(tpot_detail)
        step_total_time_sum = int(filtered_appid_df['step_total_time'].sum())

        results[worker + " sum"] = (f"total {step_total_time_sum} ms / tpot_per_token {tpot_detail.loc[0, 'vllm_tpot_with_prefill_ms']} ms")
        results[worker + " detail"] = generate_work_tpot_detail(filtered_appid_df)
        results[worker + " wait"] = generate_appid_wait_detail_optimized(filtered_appid_df, worker_dict)

    if len(results) == 1:
        print(f"warning：找不到appid '{appid}'，请确认appid是否传递正确")
    result_file = f"{data_path}/{appid}_statistic.xlsx"
    write_appid_statistic_to_file(results, result_file)

    # Merge all all_tpot_details, take the max value for multiple workers
    tpot_detail_max = []
    if all_tpot_details:
        combined_df = pd.concat(all_tpot_details, ignore_index=True)
        # Group by appID, take the maximum value for numeric columns
        numeric_columns = [col for col in combined_df.columns if col != 'appID']
        # Use groupby and max
        tpot_detail_max = combined_df.groupby('appID')[numeric_columns].max().reset_index()
    return tpot_detail_max

async def analyze_vllm_statistic_async(vllm_stats_file_dict, appid, response_length_tokens, llm_time_sec, tpot_sec_per_token, data_path):
    """
    Async wrapper, keep the original function unchanged, execute using thread pool
    """
    loop = asyncio.get_event_loop()

    # Execute the entire sync function in the thread pool
    return await loop.run_in_executor(
        None,
        analyze_vllm_statistic_sync,
        vllm_stats_file_dict,
        appid,
        response_length_tokens,
        llm_time_sec,
        tpot_sec_per_token,
        data_path
    )

def write_work_sum_statistic_to_file(worker_sum_results, result_file):
    # Convert dictionary to DataFrame
    data_list = []
    for worker, stats in worker_sum_results.items():
        row = {
            'worker': worker,
            'rollout_e2etime(s)': stats['rollout_e2etime(s)'],
            'worker_total_time(s)': stats['worker_total_time(s)'],
            'worker_time_ratio': stats['worker_time_ratio'],
            'worker_generate_tokens': stats['worker_generate_tokens'],
            'worker_total_batch': stats['worker_total_batch'],
            'worker_total_q_tokens': stats['worker_total_q_tokens'],
            'woker_req_num': stats['woker_req_num']
        }
        data_list.append(row)

    # Create DataFrame
    df = pd.DataFrame(data_list)

    # Specify column order
    columns_order = ['worker', 'rollout_e2etime(s)', 'worker_total_time(s)', 'worker_time_ratio',
                     'worker_generate_tokens', 'worker_total_batch', 'worker_total_q_tokens', 'woker_req_num']
    df = df[columns_order]

    # Export to Excel file
    df.to_excel(result_file, index=False)
    print(f"数据已成功输出到 {result_file}")

# Get the number of requests processed by the worker, remove duplicates
def get_req_num(woker_dict):
    all_chatcmpl_ids = []
    for title in woker_dict['title']:
        # Dummy run has no chatcmpl, filter out
        if pd.isna(title) or "chatcmpl" not in title:
            continue
        # Split by vertical line, then filter out parts starting with chatcmpl
        parts = title.split('|')
        for part in parts:
            # Find the position starting with chatcmpl
            if 'chatcmpl' in part:
                # Extract the complete chatcmpl ID (from chatcmpl to the next separator)
                # Assume the ID format is: chatcmpl-...--0
                start_idx = part.find('chatcmpl')
                # Find the end position of the ID (encounter space, slash or end)
                for end_idx in range(start_idx, len(part)):
                    if part[end_idx] in [' ', '/', '|']:
                        break
                else:
                    end_idx = len(part)

                chatcmpl_id = part[start_idx:end_idx]
                all_chatcmpl_ids.append(chatcmpl_id)
    # Remove duplicates
    unique_chatcmpl_ids = set(all_chatcmpl_ids)
    return len(unique_chatcmpl_ids)


# Summary information by worker dimension
def analyze_work_summary_statistic(
        vllm_stats_file_dict: dict,
        rollout_e2etime_sec: float,
        data_path: str
):
    rollout_e2etime_ms = rollout_e2etime_sec * 1000
    worker_sum_results = {}
    for worker in vllm_stats_file_dict:
        worker_dict = vllm_stats_file_dict[worker]
        step_total_time_sum = worker_dict['step_total_time'].sum()
        worker_sum_results[worker] = {"rollout_e2etime(s)" : round(rollout_e2etime_ms / 1000.0, 3),
                                      "worker_total_time(s)" : round(step_total_time_sum / 1000.0, 3),
                                      "worker_time_ratio" : (step_total_time_sum / rollout_e2etime_ms),
                                      "worker_generate_tokens" : len(worker_dict),
                                      "worker_total_batch" : worker_dict['batch_num'].sum(),
                                      "worker_total_q_tokens" : worker_dict['num_actual_tokens'].sum(),
                                      "woker_req_num" : get_req_num(worker_dict)}
    # print("worker_sum_results:", worker_sum_results)
    last_dir = os.path.basename(data_path)
    result_file = f"{data_path}/{last_dir}_woker_summary_statistic.xlsx"
    write_work_sum_statistic_to_file(worker_sum_results, result_file)

def analyze_rollout_vllm_log(vllm_stats_file_dict, log_file, data_path):
    rollout_tpots = load_topts_from_rollout_log_to_df(log_file)
    appid_tpot_details = []
    for index, row in rollout_tpots.iterrows():
        tpot_detail_max = analyze_vllm_statistic_sync(vllm_stats_file_dict=vllm_stats_file_dict,
                            appid=f"{row['appID']}--{row['step_idx']}",
                            response_length_tokens=row['response_length_tokens'],
                            llm_time_sec=row['llm_time_sec'],
                            tpot_sec_per_token=row['tpot_sec_per_token'],
                            data_path=data_path)
        appid_tpot_details.append(tpot_detail_max)
    # Output new rollout_tpots
    last_dir = os.path.basename(data_path)
    result_file = f"{data_path}/{last_dir}_tpot_detail.xlsx"
    pd.DataFrame(appid_tpot_details).to_excel(result_file, index=False)
    print(f"数据已成功输出到 {result_file}")

def analyze_rollout_vllm_log_parallel(vllm_stats_file_dict, log_file, data_path):
    """
    Main function for parallel processing
    """
    rollout_tpots = load_topts_from_rollout_log_to_df(log_file)
    appid_tpot_details = []

    async def process_row(row):
        """
        Async function to process a single row of data
        """
        tpot_detail_max = await analyze_vllm_statistic_async(
            vllm_stats_file_dict=vllm_stats_file_dict,
            appid=f"{row['appID']}--{row['step_idx']}",
            response_length_tokens=row['response_length_tokens'],
            llm_time_sec=row['llm_time_sec'],
            tpot_sec_per_token=row['tpot_sec_per_token'],
            data_path=data_path
        )
        return tpot_detail_max

    async def process_all_rows():
        """
        Async function to process all rows in parallel
        """
        tasks = []
        for index, row in rollout_tpots.iterrows():
            task = asyncio.create_task(process_row(row))
            tasks.append(task)

        # Execute all tasks in parallel with asyncio.gather
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"处理行时出错: {result}")
            else:
                valid_results.append(result)

        return valid_results

    # Run async main loop
    try:
        appid_tpot_details = asyncio.run(process_all_rows())
    except RuntimeError as e:
        if "Event loop is closed" in str(e) or "already running" in str(e):
            # If an event loop is already running (e.g. in Jupyter notebook)
            loop = asyncio.get_event_loop()
            appid_tpot_details = loop.run_until_complete(process_all_rows())
        else:
            raise

    # Process results
    last_dir = os.path.basename(data_path)
    result_file = f"{data_path}/{last_dir}_tpot_detail.xlsx"

    # Ensure results can be merged
    valid_details = []
    for detail in appid_tpot_details:
        if detail is not None:
            valid_details.append(detail)

    if valid_details:
        final_df = pd.concat(valid_details, ignore_index=True)
        # Add some statistical information
        final_df = tpot_details_post_proc(final_df)
        final_df.to_excel(result_file, index=False)
        print(f"数据已成功输出到 {result_file}")
    else:
        print("没有有效数据可以输出")

def main():
    args = argumentParse()
    vllm_stats_file_dict = load_vllm_stats_file_to_dict(args.data_path)

    if args.rollout_log_analysis_file:
        analyze_rollout_vllm_log_parallel(vllm_stats_file_dict=vllm_stats_file_dict,
        # analyze_rollout_vllm_log(vllm_stats_file_dict=vllm_stats_file_dict,
                                 log_file=args.rollout_log_analysis_file,
                                 data_path=args.data_path)
        rollout_e2etime_sec = parse_e2e_rollout_time_from_log_file(log_file=args.rollout_logs_file, data_path=args.data_path)
        analyze_work_summary_statistic(vllm_stats_file_dict=vllm_stats_file_dict,
                                   rollout_e2etime_sec=rollout_e2etime_sec,
                                   data_path=args.data_path)
        analyze_profiling_step_info(vllm_stats_file_dict=vllm_stats_file_dict,
                                    data_path=args.data_path)
    else:
        analyze_vllm_statistic_sync(vllm_stats_file_dict=vllm_stats_file_dict,
                            appid=args.appid,
                            response_length_tokens=args.response_length_tokens,
                            llm_time_sec=args.llm_time_sec,
                            tpot_sec_per_token=args.tpot_sec_per_token,
                            data_path=args.data_path)
if __name__ == "__main__":
    main()