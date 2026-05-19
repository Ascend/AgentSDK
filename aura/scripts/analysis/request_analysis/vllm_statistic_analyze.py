#!/usr/bin/env python3
import os
import re

# import shutil
# from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from typing import Dict
import argparse
import pandas as pd
import asyncio
import numpy as np

"""
参数：
- appid: 应用程序ID
- response_length_tokens: 响应token数量
- llm_time_sec: LLM处理时间（秒）
- tpot_sec_per_token: 每个token的处理时间（秒/token）
"""


def argumentParse():
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="LLM性能分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python %(prog)s --appid myapp --tokens 1000 --time 5.2 --tpot 0.005
  python %(prog)s -a test_app -t 500 -l 2.5 -p 0.003
        """,
    )

    # 添加参数
    parser.add_argument("--appid", type=str, required=False, help="应用程序ID")

    parser.add_argument(
        "--response_length_tokens", type=int, required=False, help="响应token数量", dest="response_length_tokens"
    )

    parser.add_argument("--llm_time_sec", type=float, required=False, help="LLM处理时间（秒）", dest="llm_time_sec")

    parser.add_argument(
        "--tpot_sec_per_token",
        type=float,
        required=False,
        help="每个token的处理时间（秒/token）",
        dest="tpot_sec_per_token",
    )

    parser.add_argument("--data_path", type=str, required=True, help="log存放的路径", dest="data_path")

    parser.add_argument(
        "--rollout_log_analysis_file",
        type=str,
        required=False,
        help="rollout log_analysis表格存放的路径",
        dest="rollout_log_analysis_file",
    )

    parser.add_argument(
        "--rollout_logs_file", type=str, required=False, help="rollout logs.txt存放的路径", dest="rollout_logs_file"
    )

    parser.add_argument(
        "--is_profiling", type=bool, required=False, help="是否开启了profiling采集", dest="is_profiling", default=False
    )

    global _is_profiling
    _is_profiling = parser.parse_args().is_profiling

    # 解析参数
    return parser.parse_args()


# 从rollout日志中解析rollout每个iteration的耗时
def parse_rollout_time(text_content):
    """
    使用简单方法解析日志
    """
    matches = []

    # 按行处理文本
    lines = text_content.split('\n')
    for line in lines:
        # 检查是否包含handle_full_batch_trajectories和特定模式
        if 'handle_full_batch_trajectories' in line and '===rollout iteration:' in line and 'timing/rollout :' in line:
            # 提取iteration
            iteration_match = re.search(r'===rollout iteration:\s*(\d+)', line)
            # 提取timing
            timing_match = re.search(r'timing/rollout\s*:\s*([\d\.]+)', line)

            if iteration_match and timing_match:
                iteration = int(iteration_match.group(1))
                timing = float(timing_match.group(1))
                matches.append({'iteration': iteration, 'rollout_time(s)': timing})

    return matches


def parse_e2e_rollout_time_from_log_file(log_file: str, data_path: str):
    """
    从日志文件解析rollout数据
    """
    # 检查文件是否存在
    if not os.path.exists(log_file):
        print(f"错误: 文件 {log_file} 不存在")
        return None

    # 读取文件内容
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        with open(log_file, 'r', encoding='gbk') as f:
            text_content = f.read()

    matches = parse_rollout_time(text_content)

    if not matches:
        print("未找到匹配的rollout数据")
        return None

    print(f"找到 {len(matches)} 条匹配的rollout数据")

    # 创建DataFrame
    df = pd.DataFrame(matches)
    df = df.sort_values('iteration').reset_index(drop=True)

    # 计算汇总
    total_time = df['rollout_time(s)'].sum()
    avg_time = df['rollout_time(s)'].mean()
    max_time = df['rollout_time(s)'].max()
    min_time = df['rollout_time(s)'].min()

    # 添加汇总行
    summary_row = pd.DataFrame({'iteration': ['SUMMARY'], 'rollout_time(s)': [total_time]})

    df_with_summary = pd.concat([df, summary_row], ignore_index=True)

    # 输出到Excel
    output_file = f"{data_path}/{os.path.basename(data_path)}_e2e_rollout_time_analysis.xlsx"
    df_with_summary.to_excel(output_file, index=False)

    print(f"\n数据已导出到: {output_file}")
    print("\n解析结果:")
    print(df_with_summary.to_string())

    return total_time


def filter_worker_pid(filename):
    # 匹配从开头到时间戳之前的所有内容
    pattern = r'^(.+?)-\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.[a-zA-Z0-9]+$'
    result = re.search(pattern, filename)

    if result:
        extracted = result.group(1)
        return extracted  # 输出: 0.0.0.0 IntegratedWorker pid=43131
    return filename


def load_vllm_stats_file_to_dict(data_path):
    """
    遍历指定文件夹中的所有CSV文件，读取为DataFrame并存储在字典中

    Parameters:
    data_path (str): 包含CSV文件的文件夹路径

    Returns:
    dict: 键为文件名，值为对应的DataFrame
    """
    file_dict = {}

    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"错误: 路径 '{data_path}' 不存在")
        return file_dict

    # 支持的文件扩展名
    supported_extensions = ['.csv', '.xlsx', '.xls']

    # 遍历文件夹中的所有文件
    for filename in os.listdir(data_path):
        if "IntegratedWorker" not in filename:
            continue
        # 获取文件扩展名
        file_ext = os.path.splitext(filename)[1].lower()

        # 检查是否为支持的文件类型
        if file_ext in supported_extensions:
            # 构建完整的文件路径
            file_path = os.path.join(data_path, filename)

            try:
                # 根据文件类型选择读取方法
                if file_ext == '.csv':
                    df = pd.read_csv(file_path)
                    file_type = "CSV"
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                    file_type = "Excel"
                else:
                    continue  # 理论上不会执行到这里

                # 将DataFrame存储在字典中，键为文件名
                worker_pid = filter_worker_pid(filename)
                if worker_pid in file_dict:
                    # 总是尝试纵向合并
                    combined_df = pd.concat([file_dict[worker_pid], df], ignore_index=True, sort=False)
                    file_dict[worker_pid] = combined_df
                    print(f"已追加: {filename} 到已存在的键 '{worker_pid}' (合并后包含 {len(combined_df)} 行)")
                else:
                    file_dict[worker_pid] = df
                    print(f"已加载: {filename} ({file_type}, 包含 {len(df)} 行, {len(df.columns)} 列)")

            except Exception as e:
                print(f"错误: 无法读取文件 {filename} - {e}")

    # 打印汇总信息
    if file_dict:
        print(f"\n总共加载了 {len(file_dict)} 个文件")
    else:
        print("未找到任何CSV或Excel文件")

    return file_dict


def write_appid_statistic_to_file(results, filename):
    # 创建DataFrame
    df_data = []

    for key, value in results.items():
        if isinstance(value, list):
            # 如果是列表，第一列放key，后面的列依次放列表的每个元素
            row_data = [key] + value
        else:
            # 如果不是列表，第一列放key，第二列放value
            row_data = [key, value]
        df_data.append(row_data)

    # 找到最长的行，用于设置列数
    max_len = max(len(row) for row in df_data)

    # 确保所有行的长度一致（用None填充）
    for row in df_data:
        if len(row) < max_len:
            row.extend([None] * (max_len - len(row)))

    # 创建DataFrame
    df = pd.DataFrame(df_data)

    # 写入Excel文件
    df.to_excel(filename, index=False, header=False)

    print(f"数据已写入 {filename}")


def load_topts_from_rollout_log_to_df(file_path):
    # 读取Excel文件中的指定页签
    sheet_name = 'TPOT详细数据'
    # 读取数据
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'，请确认文件路径是否正确")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")


def load_rollout_e2etime_from_rollout_log_to_df(file_path):
    # 读取Excel文件中的指定页签
    sheet_name = '迭代时间统计'
    # 读取数据
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
    从DataFrame中解析is_profiling为TRUE的数据
    返回(ip, pid, step_id)到(with_prefill, is_dummy_run)的映射字典
    可以处理字符串和布尔值的混合类型
    """
    # 预编译正则表达式
    IP_PATTERN = re.compile(r'^([\d\.]+)')
    PID_PATTERN = re.compile(r'pid=(\d+)')
    STEP_ID_PATTERN = re.compile(r'/(\d+)$')

    # 统一is_profiling列的类型：转换为小写字符串进行比较
    # 这样比逐行调用to_bool函数快得多
    if worker_df['is_profiling'].dtype == 'object':
        is_profiling_mask = worker_df['is_profiling'].astype(str).str.lower().eq('true')
    else:
        is_profiling_mask = worker_df['is_profiling'].astype(bool)

    # 过滤出需要处理的行
    profiling_df = worker_df[is_profiling_mask].copy()

    if profiling_df.empty:
        return {}

    # 向量化提取title中的信息
    titles = profiling_df['title'].astype(str)

    # 提取IP - 使用向量化操作
    ip_matches = titles.str.extract(IP_PATTERN, expand=False)
    ips = ip_matches.fillna('unknown')

    # 提取PID - 使用向量化操作
    pid_matches = titles.str.extract(PID_PATTERN, expand=False)
    pids = pid_matches.fillna('unknown')

    # 提取step_id - 使用向量化操作
    step_id_matches = titles.str.extract(STEP_ID_PATTERN, expand=False)
    step_ids = step_id_matches.fillna('unknown')

    # 转换with_prefill和is_dummy_run为布尔值
    # 使用向量化操作
    def vectorized_to_bool(series: pd.Series) -> pd.Series:
        if series.dtype == 'object':
            return series.astype(str).str.lower().eq('true')
        return series.astype(bool)

    with_prefill = vectorized_to_bool(profiling_df['with_prefill'])
    is_dummy_run = vectorized_to_bool(profiling_df['is_dummy_run'])

    # 构建结果字典 - 使用zip和循环（比iterrows快）
    result = {}
    for ip, pid, step_id, wp, idr in zip(ips, pids, step_ids, with_prefill, is_dummy_run):
        key = f"{ip}_{pid}_{step_id}"
        result[key] = f"{wp}_{idr}"
        # key = (ip, pid, step_id)
        # result[key] = (wp, idr)

    return result


def analyze_profiling_step_info(vllm_stats_file_dict: dict, data_path: str):
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


def combine_appid_info(response_length_tokens: int, llm_time_sec: float, tpot_sec_per_token: float):
    return f"response_length_tokens {response_length_tokens} / llm_time_sec {int(llm_time_sec * 1000)} ms / tpot_sec_per_token  {int(tpot_sec_per_token * 1000)} ms"


# 使用 f-string 预编译（Python 3.12+ 自动优化）
def generate_work_tpot_detail(filtered_appid_df):
    result_list = []
    # 预定义函数减少属性查找
    get_type = lambda x: 'p' if x else 'd'

    for row in filtered_appid_df.itertuples():
        result_list.append(f"t{row.Index}-{get_type(row.with_prefill)}-b{row.batch_num} {int(row.step_total_time)}")
    return result_list


def generate_worker_dict_lookup_map(worker_dict):
    # 创建从(标题, 开始时间)到索引的映射
    # 假设标题和时间组合可以唯一标识记录
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
        previous_record = filtered_appid_df.iloc[i - 1]

        # 1) 时间间隔
        time_interval = int(
            (current_record['step_start_time'] - previous_record['step_finished_time']) * 1000
        )  # s -> ms

        # 2) 查找索引
        current_key = (current_record['title'], current_record['step_start_time'])
        previous_key = (previous_record['title'], previous_record['step_start_time'])
        # print("current_key:", current_key, " previous_key:", previous_key)

        if current_key in lookup_map and previous_key in lookup_map:
            current_idx = lookup_map[current_key]
            previous_idx = lookup_map[previous_key]
            # print("previous_idx:", previous_idx, " current_idx:", current_idx)

            # 确定开始和结束索引
            start_idx = min(current_idx, previous_idx)
            end_idx = max(current_idx, previous_idx)
            # print("start_idx:", start_idx, " end_idx:", end_idx)

            # 计算中间记录数和batch_num之和
            index_gap = end_idx - start_idx - 1
            batch_sum = 0

            # 统计中间记录的batch_num之和
            for j in range(start_idx + 1, end_idx):
                if j < len(worker_dict):
                    batch_sum += worker_dict[j].get('batch_num', 0)

            # 如果是相反的顺序，标记一下
            if current_idx < previous_idx:
                index_gap = -index_gap  # 负数表示顺序相反

            result_item = f"time-{time_interval} step-{index_gap} batch-{batch_sum}"
            results.append(result_item)
        else:
            results.append(None)
    return results


def generate_appid_wait_detail_optimized(filtered_appid_df, worker_dict):
    """进一步优化的版本，适用于大数据量"""
    lookup_map = generate_worker_dict_lookup_map(worker_dict)
    n = len(filtered_appid_df)
    results = [None] * n

    # 使用itertuples()而不是iloc，速度更快
    records = list(filtered_appid_df.itertuples(index=False, name='Record'))

    # 预计算batch_num前缀和 - 使用向量化操作
    # 假设worker_dict有一个'batch_num'列
    if 'batch_num' in worker_dict.columns:
        batch_nums = worker_dict['batch_num'].fillna(0).values
    else:
        batch_nums = [0] * len(worker_dict)

    # 使用numpy计算前缀和，如果没有numpy则使用循环
    try:
        import numpy as np

        batch_prefix_sum = np.zeros(len(worker_dict) + 1)
        batch_prefix_sum[1:] = np.cumsum(batch_nums)
        batch_prefix_sum = batch_prefix_sum.tolist()
    except ImportError:
        # 如果没有numpy，使用纯Python
        batch_prefix_sum = [0] * (len(worker_dict) + 1)
        for i in range(1, len(worker_dict) + 1):
            batch_prefix_sum[i] = batch_prefix_sum[i - 1] + batch_nums[i - 1]

    lookup_map_get = lookup_map.get

    for i in range(1, n):
        current = records[i]
        previous = records[i - 1]

        time_interval = int((current.step_start_time - previous.step_finished_time) * 1000)

        # 使用frozenset或自定义键加速查找（如果适用）
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

            # 预分配字符串缓冲区
            results[i] = f"time-{time_interval} step-{index_gap} batch-{batch_sum}"

    return results


def tpot_details_post_proc(final_tpot_details_df):
    # 计算统计量
    mean_row = final_tpot_details_df.mean(numeric_only=True).to_frame().T
    min_row = final_tpot_details_df.min(numeric_only=True).to_frame().T
    max_row = final_tpot_details_df.max(numeric_only=True).to_frame().T

    # 添加标识
    mean_row['appID'] = 'mean'
    min_row['appID'] = 'min'
    max_row['appID'] = 'max'

    # 重新排列列顺序，使appID在第一列
    cols = ['appID'] + [col for col in final_tpot_details_df.columns if col != 'appID']
    mean_row = mean_row[cols]
    min_row = min_row[cols]
    max_row = max_row[cols]

    # 对mean_row、min_row、max_row的数值保留2位小数
    # 获取数值列
    numeric_cols = final_tpot_details_df.select_dtypes(include=[np.number]).columns.tolist()

    # 对mean_row的数值列保留2位小数
    for col in numeric_cols:
        if col in mean_row.columns:
            mean_row[col] = mean_row[col].round(2)

    # 对min_row的数值列保留2位小数
    for col in numeric_cols:
        if col in min_row.columns:
            min_row[col] = min_row[col].round(2)

    # 对max_row的数值列保留2位小数
    for col in numeric_cols:
        if col in max_row.columns:
            max_row[col] = max_row[col].round(2)

    # 合并到原DataFrame
    df_with_stats = pd.concat([final_tpot_details_df, mean_row, min_row, max_row], ignore_index=True)

    # 现在根据mean行计算百分比ratio
    # 获取mean行的索引（最后第三行）
    mean_idx = len(df_with_stats) - 3

    # 需要计算百分比的列
    ratio_columns = [
        'prepare_input_time',
        'aclgraph_dispatcher_time',
        'forward_time',
        'kvconnectoroutput_time',
        'post_process_time',
        'pop_captured_sync_time',
        'post_process_compute_logits_time',
        'post_process_sampler_time',
        'post_process_other_time',
    ]
    # 分母列
    denominator_column = 'step_total_time'

    # 创建ratio行（先复制mean行的值，然后修改）
    ratio_row = df_with_stats.loc[mean_idx].copy().to_dict()

    # 修复：直接从字典中获取值，避免Series问题
    if denominator_column in ratio_row:
        denominator_value = ratio_row[denominator_column]

        if denominator_value != 0:
            for col in ratio_columns:
                if col in ratio_row:
                    # 计算百分比（%）
                    ratio_row[col] = (ratio_row[col] / denominator_value) * 100

        # 将其他数值列（除了appID和ratio_columns）设为0
        numeric_cols = final_tpot_details_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in ratio_columns and col in ratio_row:
                ratio_row[col] = 0

    # 修改appID标识
    ratio_row['appID'] = 'ratio(%)'

    # 确保ratio_row中所有数值列都保留2位小数
    for col in numeric_cols:
        if col in ratio_row:
            # 如果是浮点数，保留2位小数
            if isinstance(ratio_row[col], (int, float)):
                ratio_row[col] = round(ratio_row[col], 2)

    # 将ratio行添加到DataFrame
    df_with_stats = pd.concat([df_with_stats, pd.DataFrame([ratio_row])], ignore_index=True)

    return df_with_stats


# 生成每个appid的tpot的详细信息
def generate_appid_topt_detail(appid, response_length_tokens, llm_time_sec, tpot_sec_per_token, appid_df):
    stat_appid_df = {}
    # 如果携带了profiling，过滤掉step_total_time > 1000的数据
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

    # 选择数值列（排除字符串类型的列）
    numeric_cols = stat_appid_df.select_dtypes(include=['float64', 'int64']).columns
    stats_df = stat_appid_df[numeric_cols].describe().loc[['mean', 'min', 'max']]
    tpot_with_prefill = stats_df.loc['mean', 'step_total_time']
    tpot_ms_per_token = tpot_sec_per_token * 1000
    # print(stats_df)
    # 统计同时满足两个条件的行数
    decode_with_prefill_true_count = len(
        appid_df[(appid_df['attn_state'] == 'AscendAttentionState.DecodeOnly') & (appid_df['with_prefill'] == 'TRUE')]
    )

    tpot_detail = pd.DataFrame(
        {
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
            'decode_with_prefill_true_count': decode_with_prefill_true_count,
        },
        index=[0],
    )
    return tpot_detail


def analyze_vllm_statistic_sync(
    vllm_stats_file_dict: dict,
    appid: str,
    response_length_tokens: int,
    llm_time_sec: float,
    tpot_sec_per_token: float,
    data_path: str,
):
    results = {appid: combine_appid_info(response_length_tokens, llm_time_sec, tpot_sec_per_token)}
    all_tpot_details = []
    for worker in vllm_stats_file_dict:
        worker_dict = vllm_stats_file_dict[worker]
        filtered_appid_df = worker_dict[worker_dict['title'].str.contains(appid)]
        if len(filtered_appid_df) == 0:
            continue
        tpot_detail = generate_appid_topt_detail(
            appid, response_length_tokens, llm_time_sec, tpot_sec_per_token, filtered_appid_df
        )
        all_tpot_details.append(tpot_detail)
        #
        step_total_time_sum = int(filtered_appid_df['step_total_time'].sum())

        results[worker + " sum"] = (
            f"total {step_total_time_sum} ms / tpot_per_token {tpot_detail.loc[0, 'vllm_tpot_with_prefill_ms']} ms"
        )
        results[worker + " detail"] = generate_work_tpot_detail(filtered_appid_df)
        results[worker + " wait"] = generate_appid_wait_detail_optimized(filtered_appid_df, worker_dict)

    if len(results) == 1:
        print(f"warning：找不到appid '{appid}'，请确认appid是否传递正确")
    result_file = f"{data_path}/{appid}_statistic.xlsx"
    write_appid_statistic_to_file(results, result_file)

    # 合并所有all_tpot_details,多个work的值，取max
    tpot_detail_max = []
    if all_tpot_details:
        combined_df = pd.concat(all_tpot_details, ignore_index=True)
        # 按appID分组，对数值列取最大值
        numeric_columns = [col for col in combined_df.columns if col != 'appID']
        # 使用groupby和max
        tpot_detail_max = combined_df.groupby('appID')[numeric_columns].max().reset_index()
    return tpot_detail_max


async def analyze_vllm_statistic_async(
    vllm_stats_file_dict, appid, response_length_tokens, llm_time_sec, tpot_sec_per_token, data_path
):
    """
    异步包装器，保持原函数不变，使用线程池执行
    """
    loop = asyncio.get_event_loop()

    # 在线程池中执行整个同步函数
    return await loop.run_in_executor(
        None,
        analyze_vllm_statistic_sync,
        vllm_stats_file_dict,
        appid,
        response_length_tokens,
        llm_time_sec,
        tpot_sec_per_token,
        data_path,
    )


def write_work_sum_statistic_to_file(worker_sum_results, result_file):
    # 将字典转换为DataFrame
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
            'woker_req_num': stats['woker_req_num'],
        }
        data_list.append(row)

    # 创建DataFrame
    df = pd.DataFrame(data_list)

    # 指定列顺序
    columns_order = [
        'worker',
        'rollout_e2etime(s)',
        'worker_total_time(s)',
        'worker_time_ratio',
        'worker_generate_tokens',
        'worker_total_batch',
        'worker_total_q_tokens',
        'woker_req_num',
    ]
    df = df[columns_order]

    # 输出到Excel文件
    df.to_excel(result_file, index=False)
    print(f"数据已成功输出到 {result_file}")


# 获取worker处理了多少个req，去重
def get_req_num(woker_dict):
    all_chatcmpl_ids = []
    for title in woker_dict['title']:
        # dummy run是没有chatcmpl，过滤掉
        if pd.isna(title) or "chatcmpl" not in title:
            continue
        # 按竖线分割，然后过滤出以chatcmpl开头的部分
        parts = title.split('|')
        for part in parts:
            # 找到chatcmpl开头的位置
            if 'chatcmpl' in part:
                # 提取完整的chatcmpl ID（从chatcmpl开始到下一个分隔符）
                # 这里假设ID格式为：chatcmpl-...--0
                start_idx = part.find('chatcmpl')
                # 找到ID结束的位置（遇到空格、斜杠或结束）
                for end_idx in range(start_idx, len(part)):
                    if part[end_idx] in [' ', '/', '|']:
                        break
                else:
                    end_idx = len(part)

                chatcmpl_id = part[start_idx:end_idx]
                all_chatcmpl_ids.append(chatcmpl_id)
    # 去重
    unique_chatcmpl_ids = set(all_chatcmpl_ids)
    return len(unique_chatcmpl_ids)


# 以work维度的汇总信息
def analyze_work_summary_statistic(vllm_stats_file_dict: dict, rollout_e2etime_sec: float, data_path: str):
    rollout_e2etime_ms = rollout_e2etime_sec * 1000
    worker_sum_results = {}
    for worker in vllm_stats_file_dict:
        worker_dict = vllm_stats_file_dict[worker]
        step_total_time_sum = worker_dict['step_total_time'].sum()
        worker_sum_results[worker] = {
            "rollout_e2etime(s)": round(rollout_e2etime_ms / 1000.0, 3),
            "worker_total_time(s)": round(step_total_time_sum / 1000.0, 3),
            "worker_time_ratio": (step_total_time_sum / rollout_e2etime_ms),
            "worker_generate_tokens": len(worker_dict),
            "worker_total_batch": worker_dict['batch_num'].sum(),
            "worker_total_q_tokens": worker_dict['num_actual_tokens'].sum(),
            "woker_req_num": get_req_num(worker_dict),
        }
    # print("worker_sum_results:", worker_sum_results)
    last_dir = os.path.basename(data_path)
    result_file = f"{data_path}/{last_dir}_woker_summary_statistic.xlsx"
    write_work_sum_statistic_to_file(worker_sum_results, result_file)


def analyze_rollout_vllm_log(vllm_stats_file_dict, log_file, data_path):
    rollout_tpots = load_topts_from_rollout_log_to_df(log_file)
    appid_tpot_details = []
    for index, row in rollout_tpots.iterrows():
        tpot_detail_max = analyze_vllm_statistic_sync(
            vllm_stats_file_dict=vllm_stats_file_dict,
            appid=f"{row['appID']}--{row['step_idx']}",
            response_length_tokens=row['response_length_tokens'],
            llm_time_sec=row['llm_time_sec'],
            tpot_sec_per_token=row['tpot_sec_per_token'],
            data_path=data_path,
        )
        appid_tpot_details.append(tpot_detail_max)
    # 输出新的rollout_tpots
    last_dir = os.path.basename(data_path)
    result_file = f"{data_path}/{last_dir}_tpot_detail.xlsx"
    pd.DataFrame(appid_tpot_details).to_excel(result_file, index=False)
    print(f"数据已成功输出到 {result_file}")


def analyze_rollout_vllm_log_parallel(vllm_stats_file_dict, log_file, data_path):
    """
    并行处理的主函数
    """
    rollout_tpots = load_topts_from_rollout_log_to_df(log_file)
    appid_tpot_details = []

    async def process_row(row):
        """
        处理单行数据的异步函数
        """
        tpot_detail_max = await analyze_vllm_statistic_async(
            vllm_stats_file_dict=vllm_stats_file_dict,
            appid=f"{row['appID']}--{row['step_idx']}",
            response_length_tokens=row['response_length_tokens'],
            llm_time_sec=row['llm_time_sec'],
            tpot_sec_per_token=row['tpot_sec_per_token'],
            data_path=data_path,
        )
        return tpot_detail_max

    async def process_all_rows():
        """
        并行处理所有行的异步函数
        """
        tasks = []
        for index, row in rollout_tpots.iterrows():
            task = asyncio.create_task(process_row(row))
            tasks.append(task)

        # 使用asyncio.gather并行执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"处理行时出错: {result}")
            else:
                valid_results.append(result)

        return valid_results

    # 运行异步主循环
    try:
        appid_tpot_details = asyncio.run(process_all_rows())
    except RuntimeError as e:
        if "Event loop is closed" in str(e) or "already running" in str(e):
            # 如果已经有事件循环在运行（如在Jupyter notebook中）
            loop = asyncio.get_event_loop()
            appid_tpot_details = loop.run_until_complete(process_all_rows())
        else:
            raise

    # 处理结果
    last_dir = os.path.basename(data_path)
    result_file = f"{data_path}/{last_dir}_tpot_detail.xlsx"

    # 确保结果可以合并
    valid_details = []
    for detail in appid_tpot_details:
        if detail is not None:
            valid_details.append(detail)

    if valid_details:
        final_df = pd.concat(valid_details, ignore_index=True)
        # 增加一些统计信息
        final_df = tpot_details_post_proc(final_df)
        final_df.to_excel(result_file, index=False)
        print(f"数据已成功输出到 {result_file}")
    else:
        print("没有有效数据可以输出")


def main():
    args = argumentParse()
    vllm_stats_file_dict = load_vllm_stats_file_to_dict(args.data_path)

    if args.rollout_log_analysis_file:
        analyze_rollout_vllm_log_parallel(
            vllm_stats_file_dict=vllm_stats_file_dict,
            # analyze_rollout_vllm_log(vllm_stats_file_dict=vllm_stats_file_dict,
            log_file=args.rollout_log_analysis_file,
            data_path=args.data_path,
        )
        rollout_e2etime_sec = parse_e2e_rollout_time_from_log_file(
            log_file=args.rollout_logs_file, data_path=args.data_path
        )
        analyze_work_summary_statistic(
            vllm_stats_file_dict=vllm_stats_file_dict, rollout_e2etime_sec=rollout_e2etime_sec, data_path=args.data_path
        )
        analyze_profiling_step_info(vllm_stats_file_dict=vllm_stats_file_dict, data_path=args.data_path)
    else:
        analyze_vllm_statistic_sync(
            vllm_stats_file_dict=vllm_stats_file_dict,
            appid=args.appid,
            response_length_tokens=args.response_length_tokens,
            llm_time_sec=args.llm_time_sec,
            tpot_sec_per_token=args.tpot_sec_per_token,
            data_path=args.data_path,
        )


if __name__ == "__main__":
    main()
