import pandas as pd
import numpy as np
import json
import glob
import re
from datetime import datetime
import os
from collections import defaultdict
from tqdm import tqdm
import warnings
from typing import Dict, Optional

# 忽略警告
warnings.filterwarnings('ignore')


class EnhancedHierarchicalTraceViewGenerator:
    def __init__(self, csv_file: str = "request_statistics_enhanced.csv", xlsx_dir: str = "."):
        """
        初始化增强的分层Trace View生成器（包含Step细粒度事件）

        Args:
            csv_file: 增强的请求统计CSV文件路径
            xlsx_dir: 包含 *_statistic.xlsx 文件的目录
        """
        self.csv_file = csv_file
        self.xlsx_dir = xlsx_dir
        self.df = None
        self.step_data = None  # 存储从xlsx解析的step数据
        self.worker_data = None  # 存储IntegratedWorker数据
        self.trace_events = []
        self.metadata_events = []

        # 缓存数据
        self.step_events_by_request = {}

    def load_enhanced_data(self) -> pd.DataFrame:
        """
        加载增强的请求统计数据

        Returns:
            DataFrame
        """
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"文件不存在: {self.csv_file}")

        print("=" * 70)
        print("数据字段关系说明")
        print("=" * 70)
        print("表格文件 'request_statistics_enhanced.csv' 包含以下核心关系:")
        print("1. address (处理节点): 每个DP实例的IP地址+端口")
        print("2. app_id (应用ID): 每个轨迹/应用的唯一标识")
        print("3. original_request_key (请求键): 每个推理请求的唯一标识")
        print("\n事件阶段划分:")
        print("  每个请求分为多个阶段，在Chrome Tracing中分开显示:")
        print("    - schedule_dur: 调度阶段 (cat='schedule')")
        print("    - prefill_steps: 预填充细粒度步骤 (cat='prefill_step_N')")
        print("    - decode_steps: 解码细粒度步骤 (cat='decode_step_N')")
        print("    - total_execution: 总执行阶段 (cat='total_execution') - 单独轨道")
        print("    - router: 框架总执行阶段 (cat='router') - 单独轨道")
        print("=" * 70)
        print(f"\n加载增强请求统计文件: {self.csv_file}")

        try:
            # 1. 加载CSV数据
            self._load_csv_data()

            # 2. 加载XLSX文件数据
            self._load_xlsx_statistics()

            # 3. 加载IntegratedWorker文件
            self._load_integrated_worker_files()

            # 4. 关联数据
            self._enhance_step_data()

            # 5. 数据预处理
            self._preprocess_data()

            # 6. 显示数据层次统计
            self._display_hierarchy_stats()

            return self.df

        except Exception as e:
            raise Exception(f"加载文件失败: {str(e)}")

    def _load_csv_data(self):
        """加载CSV数据并进行初步处理"""
        self.df = pd.read_csv(self.csv_file)
        # ## todo
        # self.df = self.df.iloc[:500]

        # 处理时间字段 - 使用向量化操作提高性能
        time_columns_to_process = ['start_time', 'end_time']
        time_format = "%Y-%m-%d %H:%M:%S.%f"

        for col in time_columns_to_process:
            ts_col = f"{col}_ts"
            if col in self.df.columns:
                # 尝试使用向量化操作
                try:
                    # 首先转换非空字符串
                    mask = self.df[col].notna() & (self.df[col].astype(str).str.len() > 0)
                    if mask.any():
                        self.df.loc[mask, ts_col] = (
                            pd.to_datetime(self.df.loc[mask, col], format=time_format, errors='coerce').astype('int64')
                            / 10**9
                        )  # 转换为秒级时间戳
                    else:
                        self.df[ts_col] = np.nan
                    print(f"✓ 已处理{col}时间字段")
                except Exception as e:
                    print(f"警告: 处理{col}时间字段时出错: {str(e)}")
                    # 备选方案
                    try:
                        self.df[ts_col] = pd.to_numeric(self.df[col], errors='coerce')
                        print(f"✓ 使用备选方案处理{col}字段")
                    except:
                        print(f"❌ 无法处理{col}字段")

        print(f"✓ 加载成功，共 {len(self.df)} 条记录")

        # 检查必要字段
        required_columns = [
            'address',
            'app_id',
            'original_request_key',
            'add_tick',
            'schedule_tick',
            'prefill_done_tick',
            'finish_tick',
        ]

        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"警告: 缺少必要字段: {missing_columns}")
            print(f"可用字段: {list(self.df.columns)}")

    def _load_xlsx_statistics(self):
        """加载并解析所有 *_statistic.xlsx 文件"""
        print("\n" + "=" * 70)
        print("加载XLSX细粒度统计文件")
        print("=" * 70)

        # 查找所有 *_statistic.xlsx 文件
        pattern = os.path.join(self.xlsx_dir, "*_statistic.xlsx")
        xlsx_files = glob.glob(pattern)

        if not xlsx_files:
            print("警告: 未找到 *_statistic.xlsx 文件")
            return

        print(f"找到 {len(xlsx_files)} 个XLSX统计文件")

        all_step_data = []

        for xlsx_file in tqdm(xlsx_files, desc="处理XLSX文件"):
            try:
                filename = os.path.basename(xlsx_file)

                # 使用openpyxl引擎，只读取必要的数据
                df_sheet = pd.read_excel(xlsx_file, engine='openpyxl', header=None)

                if df_sheet.shape[0] < 3 or df_sheet.shape[1] < 1:
                    continue

                # 解析第一行第一列作为request_key
                request_key_raw = df_sheet.iloc[0, 0]
                if pd.isna(request_key_raw):
                    continue

                # 清理request_key
                request_key_clean = str(request_key_raw).strip()
                if request_key_clean.startswith('chatcmpl-'):
                    request_key_clean = request_key_clean[9:]

                # 解析第二行第一列获取address_pid (格式: IP:pid)
                address_pid = None
                second_row_cell = df_sheet.iloc[1, 0]
                if isinstance(second_row_cell, str) and 'pid=' in second_row_cell:
                    # 提取IP地址 (格式: xxx.xxx.xxx.xxx)
                    ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', second_row_cell)
                    # 提取PID (格式: pid=数字)
                    pid_match = re.search(r'pid=(\d+)', second_row_cell)
                    if ip_match and pid_match:
                        ip_address = ip_match.group(1)
                        pid = pid_match.group(1)
                        address_pid = f"{ip_address}:{pid}"
                    elif pid_match:
                        # 如果没有IP地址，只使用PID
                        address_pid = pid_match.group(1)

                # 解析第三行数据（步骤信息）
                row_data = df_sheet.iloc[2]
                for col_idx in range(len(row_data)):
                    cell_content = row_data.iloc[col_idx]
                    if pd.isna(cell_content):
                        continue

                    step_info = self._parse_step_info(str(cell_content))
                    if step_info:
                        all_step_data.append(
                            {
                                'request_key': request_key_clean,
                                'step_type': step_info['type'],
                                'step_number': step_info['step_num'],
                                'step_duration_ms': step_info['duration_ms'],
                                'step_sequence': step_info['step_seq'],
                                'address_pid': address_pid,
                                'source_file': filename,
                            }
                        )

            except Exception as e:
                print(f"  警告: 处理文件 {os.path.basename(xlsx_file)} 时出错: {str(e)}")
                continue

        if all_step_data:
            self.step_data = pd.DataFrame(all_step_data)
            print(f"✓ 解析完成，共 {len(self.step_data)} 个步骤记录")

            # 显示统计信息
            if 'step_type' in self.step_data.columns:
                type_counts = self.step_data['step_type'].value_counts()
                print("  步骤类型分布:")
                for step_type, count in type_counts.items():
                    print(f"    {step_type}: {count} 个步骤")

                # 统计每个请求的步骤数
                request_step_counts = self.step_data.groupby('request_key').size()
                print(f"  每个请求平均步骤数: {request_step_counts.mean():.1f}")
                print(f"  最多步骤的请求: {request_step_counts.max()} 个步骤")
                print(f"  最少步骤的请求: {request_step_counts.min()} 个步骤")
        else:
            print("警告: 未从XLSX文件中解析到任何步骤数据")

        print("=" * 70)

    def _load_integrated_worker_files(self):
        """加载并解析所有 *IntegratedWorker*.csv 文件"""
        print("\n" + "=" * 70)
        print("加载IntegratedWorker文件")
        print("=" * 70)

        # 查找所有 *IntegratedWorker*.csv 文件
        pattern = os.path.join(self.xlsx_dir, "*IntegratedWorker*.csv")
        worker_files = glob.glob(pattern)

        if not worker_files:
            print("警告: 未找到 *IntegratedWorker*.csv 文件")
            return

        print(f"找到 {len(worker_files)} 个IntegratedWorker文件")

        all_worker_dataframes = []

        for worker_file in tqdm(worker_files, desc="处理Worker文件"):
            try:
                filename = os.path.basename(worker_file)

                # 从文件名中提取address_pid (格式: IP:pid)
                address_pid = None
                # 提取IP地址 (文件名开头的IP地址，格式: xxx.xxx.xxx.xxx)
                ip_match = re.search(r'^(\d+\.\d+\.\d+\.\d+)', filename)
                # 提取PID (格式: pid=数字)
                pid_match = re.search(r'pid=(\d+)', filename)
                if ip_match and pid_match:
                    ip_address = ip_match.group(1)
                    pid = pid_match.group(1)
                    address_pid = f"{ip_address}:{pid}"
                elif pid_match:
                    # 如果没有IP地址，只使用PID
                    address_pid = pid_match.group(1)

                # 读取CSV文件
                df_sheet = pd.read_csv(worker_file, low_memory=False)

                if df_sheet.empty:
                    continue

                # 处理 address_pid 字段
                if 'address_pid' not in df_sheet.columns and address_pid is not None:
                    df_sheet['address_pid'] = address_pid
                elif 'address_pid' not in df_sheet.columns:
                    df_sheet['address_pid'] = None

                # 添加源文件名字段
                if 'source_file' not in df_sheet.columns:
                    df_sheet['source_file'] = filename

                all_worker_dataframes.append(df_sheet)

            except Exception as e:
                print(f"  警告: 处理文件 {os.path.basename(worker_file)} 时出错: {str(e)}")
                continue

        if all_worker_dataframes:
            # 合并所有DataFrame，使用concat的优化参数
            self.worker_data = pd.concat(all_worker_dataframes, ignore_index=True, copy=False)
            print(f"✓ 解析完成，共 {len(self.worker_data)} 条worker记录")
            print(f"  合并后的字段: {list(self.worker_data.columns)}")

            # 保存合并文件为xlsx格式
            output_file = os.path.join(self.xlsx_dir, "all_IntegratedWorker.xlsx")
            self.worker_data.to_excel(output_file, index=False)
            print(f"✓ 已保存合并文件: {output_file}")
        else:
            print("警告: 未从IntegratedWorker文件中解析到任何数据")
            self.worker_data = None

        print("=" * 70)

    def _enhance_step_data(self):
        """增强step_data，关联IntegratedWorker数据"""
        if self.step_data is None or self.step_data.empty:
            print("警告: 没有步骤数据可增强")
            return

        if self.worker_data is None or self.worker_data.empty:
            print("警告: 没有worker数据可用于增强")
            return

        print("\n" + "=" * 70)
        print("增强步骤数据（关联IntegratedWorker）")
        print("=" * 70)
        print(f"左表(step_data)记录数: {len(self.step_data)}")
        print(f"右表(worker_data)记录数: {len(self.worker_data)}")

        if 'title' not in self.worker_data.columns:
            print("警告: worker_data中没有'title'字段，无法进行关联")
            return

        if 'request_key' not in self.step_data.columns:
            print("警告: step_data中没有'request_key'字段，无法进行关联")
            return

        if 'address_pid' not in self.step_data.columns:
            print("警告: step_data中没有'address_pid'字段，无法进行关联")
            return

        if 'address_pid' not in self.worker_data.columns:
            print("警告: worker_data中没有'address_pid'字段，无法进行关联")
            return

        # 创建request_key到worker记录的映射
        print("创建request_key映射...")
        worker_dict = defaultdict(list)

        # 预处理worker_data，创建索引
        for _, row in tqdm(self.worker_data.iterrows(), total=len(self.worker_data), desc="构建映射"):
            title = str(row['title']) if not pd.isna(row['title']) else ''
            # worker_address_pid = row['address_pid'] if not pd.isna(row['address_pid']) else None
            if title:
                # 提取可能的request_key
                for request_key in self.step_data['request_key'].unique():
                    if str(request_key) in title:
                        # 只保存需要的列
                        worker_row = {}
                        for col in self.worker_data.columns:
                            if col != 'title':
                                worker_row[col] = row[col]
                        worker_dict[request_key].append(worker_row)

        # 创建增强后的步骤数据
        print("关联数据...")
        enhanced_rows = []

        for _, step_row in tqdm(self.step_data.iterrows(), total=len(self.step_data), desc="关联步骤"):
            request_key = step_row['request_key']
            step_address_pid = step_row['address_pid'] if not pd.isna(step_row['address_pid']) else None
            enhanced_step = step_row.to_dict()

            if pd.isna(request_key) or request_key == '':
                # 如果没有request_key，worker字段设为空list
                for col in self.worker_data.columns:
                    if col != 'title':
                        enhanced_step[col] = []
            else:
                # 查找匹配的worker记录（需要同时满足request_key和address_pid条件）
                candidate_workers = worker_dict.get(request_key, [])

                # 进一步过滤：检查address_pid是否相等
                matching_workers = []
                for worker in candidate_workers:
                    worker_address_pid = worker.get('address_pid')
                    # 如果step的address_pid为空，则不匹配
                    if step_address_pid is None:
                        continue
                    # 如果worker的address_pid为空，则不匹配
                    if worker_address_pid is None or pd.isna(worker_address_pid):
                        continue
                    # 比较address_pid是否相等
                    if str(step_address_pid) == str(worker_address_pid):
                        matching_workers.append(worker)

                if matching_workers:
                    # 将worker记录的所有列数据按list汇总
                    for col in self.worker_data.columns:
                        if col != 'title':
                            col_values = [worker[col] for worker in matching_workers]
                            enhanced_step[col] = col_values
                else:
                    # 如果没有匹配，worker字段设为空list
                    for col in self.worker_data.columns:
                        if col != 'title':
                            enhanced_step[col] = []

            enhanced_rows.append(enhanced_step)

        # 更新step_data
        self.step_data = pd.DataFrame(enhanced_rows)
        print(f"✓ 数据增强完成，共 {len(self.step_data)} 条增强步骤记录")

        # 统计关联结果
        match_count = sum(
            1
            for row in enhanced_rows
            if any(len(row.get(col, [])) > 0 for col in self.worker_data.columns if col != 'title')
        )

        print("\n关联统计:")
        print(f"  成功关联的步骤: {match_count}/{len(self.step_data)} ({match_count / len(self.step_data) * 100:.1f}%)")

        print("=" * 70)

    def _parse_step_info(self, cell_content: str) -> Optional[Dict]:
        """解析步骤信息"""
        if not isinstance(cell_content, str):
            return None

        cell_content = cell_content.strip()

        # 使用正则表达式解析
        pattern = r't(\d+)-([pd])-b\d+\s+(\d+)'
        match = re.search(pattern, cell_content, re.IGNORECASE)

        if match:
            step_num = int(match.group(1))
            step_type = match.group(2).upper()
            duration_ms = int(match.group(3))
            step_seq = cell_content.split()[0] if ' ' in cell_content else cell_content

            return {'type': step_type, 'step_num': step_num, 'duration_ms': duration_ms, 'step_seq': step_seq}

        # 备选解析方案
        parts = cell_content.split()
        if len(parts) >= 2:
            step_seq = parts[0]
            try:
                duration_ms = int(parts[1])

                if '-p-' in step_seq.lower():
                    step_type = 'P'
                elif '-d-' in step_seq.lower():
                    step_type = 'D'
                else:
                    return None

                step_num_match = re.search(r't(\d+)-', step_seq.lower())
                step_num = int(step_num_match.group(1)) if step_num_match else 0

                return {'type': step_type, 'step_num': step_num, 'duration_ms': duration_ms, 'step_seq': step_seq}
            except:
                return None

        return None

    def _preprocess_data(self):
        """数据预处理"""
        print("\n预处理数据...")

        # 确保时间字段为浮点数 - 使用向量化操作
        time_columns = ['add_tick', 'schedule_tick', 'prefill_done_tick', 'finish_tick']
        for col in time_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 过滤无效数据
        initial_count = len(self.df)

        # 过滤时间戳为NaN的行
        vllm_time_columns = [col for col in time_columns if col in self.df.columns]
        if vllm_time_columns:
            time_mask = self.df[vllm_time_columns].notna().all(axis=1)
            self.df = self.df[time_mask].copy()

        # 过滤时间顺序错误的行
        if all(col in self.df.columns for col in time_columns):
            valid_mask = (
                (self.df['schedule_tick'] >= self.df['add_tick'])
                & (self.df['prefill_done_tick'] >= self.df['schedule_tick'])
                & (self.df['finish_tick'] >= self.df['prefill_done_tick'])
            )
            self.df = self.df[valid_mask].copy()

        filtered_count = initial_count - len(self.df)
        if filtered_count > 0:
            print(f"✓ 过滤掉 {filtered_count} 条无效数据，剩余 {len(self.df)} 条")

        # 创建显示名称 - 使用向量化操作
        if 'original_request_key' in self.df.columns:
            self.df['request_display'] = self.df['original_request_key'].apply(
                lambda x: f"req_{str(x)[-8:]}" if isinstance(x, str) and len(x) > 10 else str(x)
            )
        else:
            self.df['request_display'] = self.df.index.astype(str)

        # 创建简短ID
        if 'app_id' in self.df.columns:
            self.df['short_app_id'] = self.df['app_id'].apply(
                lambda x: f"app_{str(x)[-6:]}" if isinstance(x, str) and len(x) > 10 else str(x)
            )

        # 准备request_key用于关联
        if 'original_request_key' in self.df.columns:
            self.df['request_key_clean'] = self.df['original_request_key'].apply(
                lambda x: str(x)[9:] if isinstance(x, str) and x.startswith('chatcmpl-') else str(x)
            )

        # 关联XLSX步骤数据到主表
        if self.step_data is not None and not self.step_data.empty and 'request_key_clean' in self.df.columns:
            print("\n关联XLSX步骤数据到主表...")

            # 预先计算聚合字典
            agg_dict = {'step_type': list, 'step_number': list, 'step_duration_ms': list, 'step_sequence': list}

            # 添加其他字段
            for col in self.step_data.columns:
                if col not in ['request_key', 'step_type', 'step_number', 'step_duration_ms', 'step_sequence']:
                    agg_dict[col] = lambda x: list(x) if len(x) > 0 else []

            # 合并步骤数据
            step_summary = self.step_data.groupby('request_key').agg(agg_dict).reset_index()
            step_summary = step_summary.rename(
                columns={
                    'request_key': 'request_key_clean',
                    'step_type': 'step_types',
                    'step_number': 'step_numbers',
                    'step_duration_ms': 'step_durations_ms',
                    'step_sequence': 'step_sequences',
                }
            )

            # 使用merge优化内存使用
            self.df = pd.merge(self.df, step_summary, on='request_key_clean', how='left')

            # 统计关联结果
            matched_count = self.df['step_types'].notna().sum() if 'step_types' in self.df.columns else 0
            print(f"✓ 关联完成: {matched_count}/{len(self.df)} 个请求有步骤数据")

            # 为每个请求预计算步骤时间线
            self._precompute_step_timelines()

    def _precompute_step_timelines(self):
        """为每个请求预计算步骤时间线"""
        print("预计算步骤时间线...")

        total_steps = 0

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="计算步骤时间线"):
            request_key = row.get('request_key_clean')
            step_types = row.get('step_types', [])

            if not isinstance(step_types, list) or len(step_types) == 0:
                continue

            step_events = []

            # 计算prefill步骤
            if pd.notna(row.get('schedule_tick')) and pd.notna(row.get('prefill_done_tick')):
                prefill_steps = self._compute_prefill_steps(row)
                step_events.extend(prefill_steps)

            # 计算decode步骤
            if pd.notna(row.get('prefill_done_tick')) and pd.notna(row.get('finish_tick')):
                decode_steps = self._compute_decode_steps(row)
                step_events.extend(decode_steps)

            if step_events:
                self.step_events_by_request[request_key] = step_events
                total_steps += len(step_events)

        print(f"✓ 预计算完成: {total_steps} 个步骤事件")

    def _compute_prefill_steps(self, row):
        """计算prefill步骤时间线"""
        steps = []
        step_types = row.get('step_types', [])

        if not isinstance(step_types, list):
            return steps

        # 找出所有prefill步骤
        prefill_indices = [i for i, step_type in enumerate(step_types) if step_type == 'P']
        if not prefill_indices:
            return steps

        # prefill_start_time = row['schedule_tick']
        # prefill_end_time = row['prefill_done_tick']

        # 获取时间戳列表
        step_start_time_ts_list = row.get('step_start_time', [])
        step_finished_time_ts_list = row.get('step_finished_time', [])

        if not isinstance(step_start_time_ts_list, list):
            step_start_time_ts_list = []
        if not isinstance(step_finished_time_ts_list, list):
            step_finished_time_ts_list = []

        step_durations_ms = row.get('step_durations_ms', [])
        step_sequences = row.get('step_sequences', [])
        step_numbers = row.get('step_numbers', [])

        for idx in prefill_indices:
            # 准备步骤数据
            step_num = step_numbers[idx] if idx < len(step_numbers) else idx + 1
            step_seq = step_sequences[idx] if idx < len(step_sequences) else f"P-step{idx + 1}"

            # 获取持续时间
            if idx < len(step_durations_ms):
                step_duration_seconds = step_durations_ms[idx] / 1000.0
            else:
                continue

            # 获取时间戳
            start_time_ts = None
            end_time_ts = None

            if idx < len(step_start_time_ts_list):
                start_val = step_start_time_ts_list[0]
                if isinstance(start_val, list) and not pd.isna(start_val).all():
                    start_time_ts = float(start_val[idx])

            if idx < len(step_finished_time_ts_list):
                end_val = step_finished_time_ts_list[0]
                if isinstance(end_val, list) and not pd.isna(end_val).all():
                    end_time_ts = float(end_val[idx])

            # 如果时间戳无效，使用相对时间计算
            # if start_time_ts is None or end_time_ts is None:
            #     total_prefill_duration = prefill_end_time - prefill_start_time
            #     step_count = len(prefill_indices)
            #     if step_count > 0:
            #         step_duration_relative = total_prefill_duration / step_count
            #         step_index_in_prefill = prefill_indices.index(idx)
            #         start_time_ts = prefill_start_time + step_index_in_prefill * step_duration_relative
            #         end_time_ts = start_time_ts + step_duration_seconds

            if start_time_ts is not None and end_time_ts is not None:
                step_event = {
                    'type': 'P',
                    'step_num': step_num,
                    'step_seq': step_seq,
                    'start_time': start_time_ts,
                    'duration': step_duration_seconds,
                    'end_time': end_time_ts,
                    'absolute_start': start_time_ts,
                    'absolute_end': end_time_ts,
                }
                steps.append(step_event)

        return steps

    def _compute_decode_steps(self, row):
        """计算decode步骤时间线"""
        steps = []
        step_types = row.get('step_types', [])

        if not isinstance(step_types, list):
            return steps

        # 找出所有decode步骤
        decode_indices = [i for i, step_type in enumerate(step_types) if step_type == 'D']
        if not decode_indices:
            return steps

        decode_start_time = row['prefill_done_tick']
        decode_end_time = row['finish_tick']

        # 获取时间戳列表
        step_start_time_ts_list = row.get('step_start_time', [])
        step_finished_time_ts_list = row.get('step_finished_time', [])

        if not isinstance(step_start_time_ts_list, list):
            step_start_time_ts_list = []
        if not isinstance(step_finished_time_ts_list, list):
            step_finished_time_ts_list = []

        step_durations_ms = row.get('step_durations_ms', [])
        step_sequences = row.get('step_sequences', [])
        step_numbers = row.get('step_numbers', [])

        for idx in decode_indices:
            # 准备步骤数据
            step_num = step_numbers[idx] if idx < len(step_numbers) else idx + 1
            step_seq = step_sequences[idx] if idx < len(step_sequences) else f"D-step{idx + 1}"

            # 获取持续时间
            if idx < len(step_durations_ms):
                step_duration_seconds = step_durations_ms[idx] / 1000.0
            else:
                continue

            # 获取时间戳
            start_time_ts = None
            end_time_ts = None

            if idx < len(step_start_time_ts_list):
                start_val = step_start_time_ts_list[0]
                if isinstance(start_val, list) and not pd.isna(start_val).all():
                    start_time_ts = float(start_val[idx])

            if idx < len(step_finished_time_ts_list):
                end_val = step_finished_time_ts_list[0]
                if isinstance(end_val, list) and not pd.isna(end_val).all():
                    end_time_ts = float(end_val[idx])

            # # 如果时间戳无效，使用相对时间计算
            # if start_time_ts is None or end_time_ts is None:
            #     total_decode_duration = decode_end_time - decode_start_time
            #     step_count = len(decode_indices)
            #     if step_count > 0:
            #         step_duration_relative = total_decode_duration / step_count
            #         step_index_in_decode = decode_indices.index(idx)
            #         start_time_ts = decode_start_time + step_index_in_decode * step_duration_relative
            #         end_time_ts = start_time_ts + step_duration_seconds

            if start_time_ts is not None and end_time_ts is not None:
                step_event = {
                    'type': 'D',
                    'step_num': step_num,
                    'step_seq': step_seq,
                    'start_time': start_time_ts,
                    'duration': step_duration_seconds,
                    'end_time': end_time_ts,
                    'absolute_start': start_time_ts,
                    'absolute_end': end_time_ts,
                }
                steps.append(step_event)

        return steps

    def _display_hierarchy_stats(self):
        """显示数据层次统计信息"""
        if self.df is None or self.df.empty:
            return

        print("\n" + "=" * 70)
        print("数据层次统计")
        print("=" * 70)

        # 统计各级数量
        stats = {}
        if 'address' in self.df.columns:
            stats['unique_addresses'] = self.df['address'].nunique()
        if 'app_id' in self.df.columns:
            stats['unique_app_ids'] = self.df['app_id'].nunique()
        if 'original_request_key' in self.df.columns:
            stats['unique_requests'] = self.df['original_request_key'].nunique()

        print(f"总记录数: {len(self.df)}")
        for key, value in stats.items():
            name = key.replace('unique_', '').replace('_', ' ')
            print(f"唯一 {name}: {value}")

        # 统计步骤数据
        if self.step_data is not None:
            print("\n步骤数据统计:")
            print(f"  总步骤数: {len(self.step_data)}")
            if 'step_type' in self.step_data.columns:
                type_counts = self.step_data['step_type'].value_counts()
                for step_type, count in type_counts.items():
                    print(f"  {step_type}步骤数: {count}")

        # 统计worker数据
        if self.worker_data is not None:
            print("\nWorker数据统计:")
            print(f"  Worker记录数: {len(self.worker_data)}")
            if 'address_pid' in self.worker_data.columns:
                unique_pids = self.worker_data['address_pid'].nunique()
                print(f"  唯一PID数: {unique_pids}")

        print("=" * 70)

    def _calculate_microseconds(self, tick_value: float) -> int:
        """将秒级时间戳转换为微秒"""
        if pd.isna(tick_value):
            return 0
        return int(tick_value * 1_000_000)

    def _calculate_duration_microseconds(self, start: float, end: float) -> int:
        """计算持续时间（微秒）"""
        if pd.isna(start) or pd.isna(end):
            return 0
        duration = end - start
        return max(0, int(duration * 1_000_000))

    def _create_router_event(self, row: pd.Series, pid: str, tid: str) -> Optional[Dict]:
        """创建框架Router事件"""
        if 'start_time' not in row or pd.isna(row['start_time']):
            return None

        ts = self._calculate_microseconds(row['start_time'])

        if 'end_time_ts' in row and pd.notna(row['end_time']):
            dur = self._calculate_duration_microseconds(row['start_time'], row['end_time'])
        else:
            dur = 0

        router_tid = f"{tid}_router"

        return {
            "name": f"框架Router: {row['request_display']}",
            "cat": "router",
            "ph": "X",
            "ts": ts,
            "dur": dur,
            "pid": pid,
            "tid": router_tid,
            "args": {
                "request_key": str(row.get('original_request_key', 'unknown')),
                "request_display": str(row.get('request_display', 'unknown')),
                "app_id": str(row.get('app_id', 'unknown')),
                "address": str(row.get('address', 'unknown')),
                "stage": "router",
                "start_time": float(row['start_time']),
                "end_time": float(row.get('end_time_ts', 0))
                if 'end_time_ts' in row and pd.notna(row['end_time_ts'])
                else 0,
            },
        }

    def _create_schedule_event(self, row: pd.Series, pid: str, tid: str) -> Optional[Dict]:
        """创建调度事件"""
        if 'add_tick' not in row or 'schedule_tick' not in row:
            return None

        ts = self._calculate_microseconds(row['add_tick'])
        dur = self._calculate_duration_microseconds(row['add_tick'], row['schedule_tick'])

        schedule_tid = f"{tid}_schedule"

        return {
            "name": f"调度: {row['request_display']}",
            "cat": "schedule",
            "ph": "X",
            "ts": ts,
            "dur": dur,
            "pid": pid,
            "tid": schedule_tid,
            "args": {
                "request_key": str(row.get('original_request_key', 'unknown')),
                "request_display": str(row.get('request_display', 'unknown')),
                "app_id": str(row.get('app_id', 'unknown')),
                "address": str(row.get('address', 'unknown')),
                "stage": "schedule_dur",
                "add_tick": float(row['add_tick']),
                "schedule_tick": float(row['schedule_tick']),
                "wait_time_ms": round((row['schedule_tick'] - row['add_tick']) * 1000, 2),
            },
        }

    def _create_step_event(self, row: pd.Series, step_info: Dict, pid: str, tid: str, idx: int) -> Optional[Dict]:
        """创建步骤事件"""
        if not step_info:
            return None

        start_time = step_info.get('start_time')
        end_time = step_info.get('end_time')

        if start_time is None or end_time is None or pd.isna(start_time) or pd.isna(end_time):
            return None

        ts = self._calculate_microseconds(start_time)
        dur = self._calculate_duration_microseconds(start_time, end_time)

        step_type = step_info.get('type', '')
        step_num = step_info.get('step_num', 0)
        step_seq = step_info.get('step_seq', f"{step_type}-step{step_num}")

        if step_type == 'P':
            step_tid = f"{tid}_prefill"
            cat_base = "prefill"
            step_name = "Prefill"
        else:
            step_tid = f"{tid}_decode"
            cat_base = "decode"
            step_name = "Decode"

        return {
            "name": f"{step_name}-{step_seq}: {row['request_display']}",
            "cat": cat_base,
            "ph": "X",
            "ts": ts,
            "dur": dur,
            "pid": pid,
            "tid": step_tid,
            "args": {
                "request_key": str(row.get('original_request_key', 'unknown')),
                "request_display": str(row.get('request_display', 'unknown')),
                "app_id": str(row.get('app_id', 'unknown')),
                "address": str(row.get('address', 'unknown')),
                "stage": f"{cat_base}_step",
                "step_type": step_type,
                "step_number": step_num,
                "step_sequence": step_seq,
                "step_duration_ms": round(step_info.get('duration', 0) * 1000, 2),
                "absolute_start_time": float(step_info.get('absolute_start', start_time)),
                "absolute_end_time": float(step_info.get('absolute_end', end_time)),
                "relative_start": float(start_time - row.get('add_tick', 0)),
                "attn_state": row.get('attn_state')[0][idx],
                "seq_lens": row.get('seq_lens')[0][idx],
                "with_prefill": row.get('with_prefill')[0][idx],
                "batch_num": row.get('batch_num')[0][idx],
                "prepare_input_time": row.get('prepare_input_time')[0][idx],
                "aclgraph_dispatcher_time": row.get('aclgraph_dispatcher_time')[0][idx],
                "forward_time": row.get('forward_time')[0][idx],
                "post_process_time": row.get('post_process_time')[0][idx],
                "step_inter_time": row.get('step_inter_time')[0][idx],
                "num_actual_tokens": row.get('num_actual_tokens')[0][idx],
            },
        }

    def _create_total_execution_event(self, row: pd.Series, pid: str, tid: str) -> Optional[Dict]:
        """创建总执行事件"""
        if 'add_tick' not in row or 'finish_tick' not in row:
            return None

        ts = self._calculate_microseconds(row['add_tick'])
        dur = self._calculate_duration_microseconds(row['add_tick'], row['finish_tick'])

        total_tid = f"{tid}_total"

        return {
            "name": f"总执行: {row['request_display']}",
            "cat": "total_execution",
            "ph": "X",
            "ts": ts,
            "dur": dur,
            "pid": pid,
            "tid": total_tid,
            "args": {
                "request_key": str(row.get('original_request_key', 'unknown')),
                "request_display": str(row.get('request_display', 'unknown')),
                "app_id": str(row.get('app_id', 'unknown')),
                "address": str(row.get('address', 'unknown')),
                "stage": "total_execution",
                "add_tick": float(row['add_tick']),
                "finish_tick": float(row['finish_tick']),
                "total_time_ms": round((row['finish_tick'] - row['add_tick']) * 1000, 2),
                "schedule_time_ms": round((row['schedule_tick'] - row['add_tick']) * 1000, 2),
                "prefill_time_ms": round((row['prefill_done_tick'] - row['schedule_tick']) * 1000, 2),
                "decode_time_ms": round((row['finish_tick'] - row['prefill_done_tick']) * 1000, 2),
            },
        }

    def _create_hierarchy_metadata_events(self):
        """创建分层结构的元数据事件"""
        print("\n创建分层元数据事件...")

        if self.df is None or self.df.empty:
            return

        # 创建进程元数据
        if 'address' in self.df.columns:
            unique_addresses = self.df['address'].unique()
            for address in unique_addresses:
                address_data = self.df[self.df['address'] == address]
                app_count = address_data['app_id'].nunique()
                request_count = len(address_data)

                self.metadata_events.append(
                    {
                        "name": "process_name",
                        "ph": "M",
                        "pid": str(address),
                        "args": {
                            "name": f"DP实例: {address}",
                            "description": f"处理节点，服务{app_count}个应用，{request_count}个请求",
                        },
                    }
                )

        # 创建线程元数据
        if 'app_id' in self.df.columns:
            unique_apps = self.df['app_id'].unique()
            thread_types = [
                ("router", "Router阶段"),
                ("schedule", "调度阶段"),
                ("total", "总执行阶段"),
                ("prefill", "Prefill阶段"),
                ("decode", "Decode阶段"),
            ]

            for app_id in unique_apps:
                short_app_id = f"app_{str(app_id)[-6:]}" if len(str(app_id)) > 10 else str(app_id)
                app_addresses = self.df[self.df['app_id'] == app_id]['address'].unique()

                for address in app_addresses:
                    for thread_key, thread_name in thread_types:
                        self.metadata_events.append(
                            {
                                "name": "thread_name",
                                "ph": "M",
                                "pid": str(address),
                                "tid": f"{app_id}_{thread_key}",
                                "args": {
                                    "name": f"{short_app_id} - {thread_name}",
                                    "app_id": str(app_id),
                                    "address": str(address),
                                    "stage": thread_key,
                                },
                            }
                        )

        # 创建trace范围的元数据
        self.metadata_events.append(
            {
                "name": "trace_metadata",
                "ph": "M",
                "args": {
                    "trace_name": "VLLM细粒度步骤时间线（带时间戳）",
                    "trace_description": "包含Prefill和Decode细粒度步骤的完整分析，带精确时间戳",
                    "hierarchy": "address(pid) → app_id×stage(tid) → original_request_key(event)",
                    "stages": "router, schedule, prefill_steps, decode_steps, total_execution",
                    "generated_at": datetime.now().isoformat(),
                    "total_records": int(len(self.df)),
                    "total_addresses": int(self.df['address'].nunique() if 'address' in self.df.columns else 0),
                    "total_apps": int(self.df['app_id'].nunique() if 'app_id' in self.df.columns else 0),
                    "has_step_data": self.step_data is not None and len(self.step_data) > 0,
                    "has_worker_data": self.worker_data is not None and len(self.worker_data) > 0,
                    "time_range_start": float(self.df['add_tick'].min() if 'add_tick' in self.df.columns else 0),
                    "time_range_end": float(self.df['finish_tick'].max() if 'finish_tick' in self.df.columns else 0),
                },
            }
        )

    def generate_separated_phase_trace_events(self):
        """生成阶段分离的trace事件"""
        if self.df is None or self.df.empty:
            print("警告: 没有数据可生成事件")
            return

        print("\n生成阶段分离的Trace事件（包含细粒度步骤）...")
        print("每个请求的多个阶段在不同轨道上显示...")

        total_events = 0
        step_events_count = 0

        # 按address分组生成事件
        if 'address' in self.df.columns:
            for address, address_group in self.df.groupby('address'):
                pid = str(address)

                # 在这个address下，按app_id分组
                if 'app_id' in address_group.columns:
                    for app_id, app_group in address_group.groupby('app_id'):
                        tid = str(app_id)

                        # 为这个app_id下的每个请求生成事件
                        for _, row in app_group.iterrows():
                            try:
                                # 1. Router事件
                                router_event = self._create_router_event(row, pid, tid)
                                if router_event:
                                    self.trace_events.append(router_event)
                                    total_events += 1

                                # 2. 调度事件
                                schedule_event = self._create_schedule_event(row, pid, tid)
                                if schedule_event:
                                    self.trace_events.append(schedule_event)
                                    total_events += 1

                                # 3. 步骤事件
                                request_key = row.get('request_key_clean')
                                if request_key in self.step_events_by_request:
                                    step_list = self.step_events_by_request[request_key]
                                    for idx, step_info in enumerate(step_list):
                                        step_event = self._create_step_event(row, step_info, pid, tid, idx)
                                        if step_event:
                                            self.trace_events.append(step_event)
                                            total_events += 1
                                            step_events_count += 1

                                # 4. 总执行事件
                                total_event = self._create_total_execution_event(row, pid, tid)
                                if total_event:
                                    self.trace_events.append(total_event)
                                    total_events += 1

                            except Exception as e:
                                request_display = row.get('request_display', 'unknown')
                                print(f"    警告: 为请求 {request_display} 生成事件时出错: {str(e)}")
                                continue

        print(f"\n✓ 生成 {total_events} 个分离阶段Trace事件")
        print(f"  其中步骤事件: {step_events_count} 个")

        # 创建分层元数据
        self._create_hierarchy_metadata_events()

    def save_separated_phase_trace_json(self, output_file: str = "separated_phase_trace_view.json"):
        """保存阶段分离的Trace JSON文件"""
        if not self.trace_events and not self.metadata_events:
            print("警告: 没有事件数据可保存")
            return

        print(f"\n保存阶段分离Trace JSON到: {output_file}")

        try:
            # 合并所有事件
            all_events = self.trace_events + self.metadata_events

            # 按时间戳排序
            all_events.sort(key=lambda x: x.get('ts', 0))

            # 构建完整的trace对象
            trace_data = {
                "traceEvents": all_events,
                "displayTimeUnit": "ms",
                "otherData": {
                    "version": "5.0",
                    "generator": "VLLM增强细粒度步骤Trace生成器",
                    "description": "包含Prefill和Decode细粒度步骤的完整性能分析，带精确时间戳",
                    "separation_strategy": {
                        "router_phase": "cat='router', tid='{app_id}_router'",
                        "schedule_phase": "cat='schedule', tid='{app_id}_schedule'",
                        "prefill_steps": "cat='prefill_step_N', tid='{app_id}_prefill' (同一行显示)",
                        "decode_steps": "cat='decode_step_N', tid='{app_id}_decode' (同一行显示)",
                        "total_phase": "cat='total_execution', tid='{app_id}_total'",
                    },
                    "data_sources": {
                        "main_data": "request_statistics_enhanced.csv",
                        "step_data": "*_statistic.xlsx",
                        "worker_data": "*IntegratedWorker*.xlsx",
                    },
                },
            }

            # 保存为JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False)

            print(f"✓ 已保存 {len(all_events)} 个事件到 {output_file}")

            # 显示阶段分离统计信息
            self._display_separated_phase_statistics(all_events)

        except Exception as e:
            raise Exception(f"保存JSON文件失败: {str(e)}")

    def _display_separated_phase_statistics(self, all_events: list):
        """显示阶段分离统计信息"""
        print("\n" + "=" * 70)
        print("阶段分离Trace文件统计")
        print("=" * 70)

        # 按category统计
        category_stats = defaultdict(int)
        tid_stats = defaultdict(set)

        for event in all_events:
            cat = event.get('cat', '')
            tid = event.get('tid', '')
            pid = event.get('pid', '')

            if cat:
                category_stats[cat] += 1

            if tid and pid:
                tid_stats[pid].add(tid)

        print(f"总事件数: {len(all_events)}")

        # 统计基础事件
        for cat, count in category_stats.items():
            print(f"  {cat}: {count} 事件")

        # 统计线程数
        total_threads = sum(len(tids) for tids in tid_stats.values())
        print(f"\n总线程数: {total_threads}")

        # 统计元数据事件
        metadata_count = sum(1 for e in all_events if e.get('ph') == 'M')
        print(f"元数据事件数: {metadata_count}")

        print("=" * 70)

    def generate_separated_phase_trace_view(self, output_file: str = "separated_phase_trace_view.json"):
        """生成完整的阶段分离Trace View"""
        print("=" * 70)
        print("VLLM增强细粒度步骤Trace View生成器")
        print("=" * 70)

        try:
            # 1. 加载数据
            self.load_enhanced_data()

            if self.df is None or self.df.empty:
                print("警告: 没有有效数据")
                return

            # 2. 生成阶段分离的事件
            self.generate_separated_phase_trace_events()

            # 3. 保存JSON
            self.save_separated_phase_trace_json(output_file)

            print("\n" + "=" * 70)
            print("增强细粒度步骤Trace生成完成!")
            print("=" * 70)

        except Exception as e:
            print(f"生成阶段分离Trace View时出错: {str(e)}")
            import traceback

            traceback.print_exc()


def main():
    """主函数"""
    # 配置参数
    csv_file = r"\xxx\xxx\request_statistics_enhanced.csv"
    xlsx_dir = r"\xxx\xxx\\64token"
    output_file = r"\xxx\xxx\\trace_view_enhanced_with_timestamps_64token.json"

    # 创建生成器并执行
    generator = EnhancedHierarchicalTraceViewGenerator(csv_file, xlsx_dir)
    generator.generate_separated_phase_trace_view(output_file)


if __name__ == "__main__":
    main()
