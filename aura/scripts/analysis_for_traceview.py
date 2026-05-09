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

import pandas as pd
import numpy as np
import json
import glob
import re
from datetime import datetime, timedelta
import os
from collections import defaultdict
import time
from tqdm import tqdm, trange
import warnings
from typing import Dict, List, Any, Optional

warnings.filterwarnings("ignore")


class EnhancedHierarchicalTraceViewGenerator:
    def __init__(self, csv_file: str = "request_statistics_enhanced.csv",
                 xlsx_dir: str = "."):
        """
        Initialize enhanced hierarchical Trace View generator (with fine-grained Step events)
        """
        self.csv_file = csv_file
        self.xlsx_dir = xlsx_dir
        self.df = None
        self.step_data = None
        self.worker_data = None
        self.trace_events = []
        self.metadata_events = []

        self.step_events_by_request = {}

    def load_enhanced_data(self) -> pd.DataFrame:
        """
        Load enhanced request statistics data
        """
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"File {self.csv_file} does not exist")

        print("=" * 70)
        print("Data Field Relationship Description")
        print("=" * 70)
        print("Table file ‘request_statistics_enhanced.csv’ contains the following core relationships:")
        print("1. address (processing node): IP address + port of each DP instance")
        print("2. app_id (application ID): unique identifier of each trajectory/application")
        print("3. original_request_key (request key): unique identifier of each inference request")
        print("\nEvent phase breakdown:")
        print("  Each request is divided into multiple phases, displayed separately in Chrome Tracing:")
        print("    - schedule_dur: scheduling phase (cat=’schedule’")
        print("    - prefill_steps: fine-grained prefill steps (cat=’prefill_step_N’")
        print("    - decode_steps: fine-grained decode steps (cat=’decode_step_N’")
        print("    - total_execution: total execution phase (cat=’total_execution’) - separate track")
        print("    - router: framework total execution phase (cat=’router’) - separate track")
        print("=" * 70)
        print(f"\nLoading enhanced request statistics file: {self.csv_file}")
    
        try:
            # 1. Load CSV data
            self._load_csv_data()

            # 2. Load XLSX file data
            self._load_xlsx_statistics()

            # 3. Load IntegratedWorker files
            self._load_integrated_worker_files()

            # 4. Correlate data
            self._enhance_step_data()

            # 5. Data preprocessing
            self._preprocess_data()

            # 6. Display data hierarchy statistics
            self._display_hierarchy_stats()

            return self.df
        
        except Exception as e:
            raise Exception(f"Failed to load file: {str(e)}")
        
    def _load_csv_data(self):
        """
        Load CSV data and perform initial processing
        """
        self.df = pd.read_csv(self.csv_file)

        time_columns_to_process = ['start_time', 'end_time']
        time_format = "%Y-%m-%d %H:%M:%S.%f"

        for col in time_columns_to_process:
            ts_col = f"{col}_ts"
            if col in self.df.columns:
                try:
                    mask = self.df[col].notna() & (self.df[col].astype(str).str.len() > 0)
                    if mask.any():
                        self.df.loc[mask, ts_col] = pd.to_datetime(self.df.loc[mask, col], format=time_format, errors='coerce'
                        ).astype('int64') / 10**9
                    else:
                        self.df[ts_col] = np.nan
                    print(f"Processed {col} time field")
                except Exception as e:
                    print(f"Warning: Error processing {col} time field: {str(e)}")
                    try:
                        self.df[ts_col] = pd.to_numeric(self.df[col], errors='coerce')
                        print(f"Used fallback method to process {col} field")
                    except:
                        print(f"Failed to process {col} field")
                
        print(f"Loading successful, total {len(self.df)} records")

        required_columns = [
            'address', 'app_id', 'original_request_key',
            'add_tick', 'schedule_tick', 'prefill_done_tick', 'finish_tick'
        ]

        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"Warning: Missing required fields: {missing_columns}")
            print(f"Available fields: {list(self.df.columns)}")
    
    def _load_xlsx_statistics(self):
        """
        Load and parse all *_statistic.xlsx files
        """
        print("\n" + "=" * 70)
        print("Loading XLSX fine-grained statistics files")
        print("=" * 70)

        pattern = os.path.join(self.xlsx_dir, "*_statistic.xlsx")
        xlsx_files = glob.glob(pattern)

        if not xlsx_files:
            print("Warning: No *_statistic.xlsx files found")
            return
        
        print(f"Found {len(xlsx_files)} matching XLSX files")

        all_step_data = []

        for xlsx_file in tqdm(xlsx_files, desc="Processing XLSX files"):
            try:
                filename = os.path.basename(xlsx_file)

                df_sheet = pd.read_excel(xlsx_file, engine='openpyxl', header=None)

                if df_sheet.shape[0] < 3 or df_sheet.shape[1] < 1:
                    continue
            
                request_key_raw = df_sheet.iloc[0, 0]
                if pd.isna(request_key_raw):
                    continue

                request_key_clean = str(request_key_raw).strip()
                if request_key_clean.startswith('chatcmpl-'):
                    request_key_clean = request_key_clean[9:]
                
                address_pid = None
                second_row_cell = df_sheet.iloc[1, 0]
                if isinstance(second_row_cell, str) and 'pid=' in second_row_cell:
                    ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', second_row_cell)
                    pid_match = re.search(r'pid=(\d+)', second_row_cell)
                    if ip_match and pid_match:
                        ip_address = ip_match.group(1)
                        pid = pid_match.group(1)
                        address_pid = f"{ip_address}:{pid}"
                    elif pid_match:
                        address_pid = pid_match.group(1)
                    
                row_data = df_sheet.iloc[2]
                for col_idx in range(len(row_data)):
                    cell_content = row_data.iloc[col_idx]
                    if pd.isna(cell_content):
                        continue

                    step_info = self._parse_step_info(str(cell_content))
                    if step_info:
                        all_step_data.append({
                            'request_key': request_key_clean,
                            'step_type': step_info['type'],
                            'step_number': step_info['step_num'],
                            'step_duration_ms': step_info['duration_ms'],
                            'step_sequence': step_info['step_seq'],
                            'address_pid': address_pid,
                            'source_file': filename
                        })
            except Exception as e:
                print(f"Warning: Error processing file {os.path.basename(xlsx_file)}: {str(e)}")
                continue
        
        if all_step_data:
            self.step_data = pd.DataFrame(all_step_data)
            print(f"Parsing complete, total {len(self.step_data)} step records")

            if 'step_type' in self.step_data.columns:
                type_counts = self.step_data['step_type'].value_counts()
                print("Step type distribution:")
                for step_type, count in type_counts.items():
                    print(f"  {step_type}: {count} steps")

                request_step_counts = self.step_data.groupby('request_key').size()
                print(f"Average steps per request: {request_step_counts.mean():.1f}")
                print(f"Maximum steps: {request_step_counts.max()}")
                print(f"Minimum steps: {request_step_counts.min()}")
        else:
            print("Warning: No step data parsed from XLSX files")

        print("=" * 70)

    def _load_integrated_worker_files(self):
        """
        Load and parse all *IntegratedWorker*.csv files
        """
        pattern = os.path.join(self.xlsx_dir, "*IntegratedWorker*.csv")
        worker_files = glob.glob(pattern)

        if not worker_files:
            print("Warning: No matching IntegratedWorker files found")
            return 
        
        print(f"Found {len(worker_files)} matching IntegratedWorker files")

        all_worker_dataframes = []

        for worker_file in tqdm(worker_files, desc="Processing IntegratedWorker files"):
            try:
                filename = os.path.basename(worker_file)

                address_pid = None
                ip_match = re.search(r'^(\d+\.\d+\.\d+\.\d+)', filename)
                pid_match = re.search(r'pid=(\d+)', filename)
                if ip_match and pid_match:
                    ip_address = ip_match.group(1)
                    pid = pid_match.group(1)
                    address_pid = f"{ip_address}:{pid}"
                elif pid_match:
                    address_pid = pid_match.group(1)
                
                df_sheet = pd.read_csv(worker_file, low_memory=False)

                if df_sheet.empty:
                    continue
                
                if 'address_pid' not in df_sheet.columns and address_pid is not None:
                    df_sheet['address_pid'] = address_pid
                elif 'address_pid' not in df_sheet.columns:
                    df_sheet['address_pid'] = None
                
                if 'source_file' not in df_sheet.columns:
                    df_sheet['source_file'] = filename
                
                all_worker_dataframes.append(df_sheet)
            
            except Exception as e:
                print(f"Warning: Error processing file {os.path.basename(worker_file)}: {str(e)}")
                continue
            
        if all_worker_dataframes:
            self.worker_data = pd.concat(all_worker_dataframes, ignore_index=True, copy=False)
            print(f"Parsing complete, total {len(self.worker_data)} worker records")
            print(f"Merged fields: {list(self.worker_data.columns)}")

            output_file = os.path.join(self.xlsx_dir, 'all_IntegratedWorker.xlsx')
            self.worker_data.to_excel(output_file, index=False)
            print(f"Saved merged IntegratedWorker data to: {output_file}")

        else:
            print("Warning: No data parsed from IntegratedWorker files")
            self.worker_data = None
        
        print("=" * 70)
    
    def _enhance_step_data(self):
        """
        Enhance step_data: correlate with IntegratedWorker data
        """
        if self.step_data is None or self.step_data.empty:
            print("Warning: No step data to enhance")
            return

        if self.worker_data is None or self.worker_data.empty:
            print("Warning: No Worker data available for enhancement")
            return
        
        print("\n" + "=" * 70)
        print("Enhancing step data (correlating with IntegratedWorker)")
        print("=" * 70)
        print(f"Left table (step_data) record count: {len(self.step_data)}")
        print(f"Right table (worker_data) record count: {len(self.worker_data)}")

        if 'title' not in self.worker_data.columns:
            print("Warning: worker_data is missing 'title' field, cannot correlate")
            return
        
        if 'request_key' not in self.step_data.columns:
            print("Warning: step_data is missing 'request_key' field, cannot correlate")
            return
        
        if 'address_pid' not in self.step_data.columns:
            print("Warning: step_data is missing 'address_pid' field, cannot correlate")
            return

        if 'address_pid' not in self.worker_data.columns:
            print("Warning: worker_data is missing 'address_pid' field, cannot correlate")
            return
        
        print("Creating request_key mapping...")
        worker_dict = defaultdict(list)

        for _, row in tqdm(self.worker_data.iterrows(), total=len(self.worker_data), desc="Creating mapping"):
            title = str(row['title']) if not pd.isna(row['title']) else ''
            if title:
                for request_key in self.step_data['request_key'].unique():
                    if str(request_key) in title:
                        worker_row = {}
                        for col in self.worker_data.columns:
                            if col != 'title':
                                worker_row[col] = row[col]
                        worker_dict[request_key].append(worker_row)

        print("Correlating data...")
        enhanced_rows = []

        for _, step_row in tqdm(self.step_data.iterrows(), total=len(self.step_data), desc="Correlating steps"):
            request_key = step_row['request_key']
            step_address_pid = step_row['address_pid'] if not pd.isna(step_row['address_pid']) else None
            enhanced_step = step_row.to_dict()

            if pd.isna(request_key) or request_key == '':
                for col in self.worker_data.columns:
                    if col != 'title':
                        enhanced_step[col] = []
            else:
                candidate_workers = worker_dict.get(request_key, [])

                matching_workers = []
                for worker in candidate_workers:
                    worker_address_pid = worker.get('address_pid')
                    if step_address_pid is None:
                        continue
                    if worker_address_pid is None or pd.isna(worker_address_pid):
                        continue
                    if str(step_address_pid) == str(worker_address_pid):
                        matching_workers.append(worker)
                
                if matching_workers:
                    for col in self.worker_data.columns:
                        if col != 'title':
                            col_values = [worker[col] for worker in matching_workers]
                            enhanced_step[col] = col_values
                else:
                    for col in self.worker_data.columns:
                        if col != 'title':
                            enhanced_step[col] = []
            
            enhanced_rows.append(enhanced_step)

        self.step_data = pd.DataFrame(enhanced_rows)
        print(f"Data enhancement complete, total {len(self.step_data)} enhanced step records")

        match_count = sum(1 for row in enhanced_rows if any(len(row.get(col, [])) > 0 for col in self.worker_data.columns if col != 'title'))

        print(f"\nCorrelation statistics:")
        print(f"Successfully correlated steps: {match_count}/{len(self.step_data)} ({match_count/len(self.step_data)*100:.1f}%)")
        print("=" * 70)


    def _parse_step_info(self, cell_content: str) -> Optional[Dict]:
        """
        Parse step information
        """
        if not isinstance(cell_content, str):
            return None
        
        cell_content = cell_content.strip()

        pattern = r't(\d+)-([pd])-b\d+\s+(\d+)'
        match = re.search(pattern, cell_content, re.IGNORECASE)

        if match:
            step_num = int(match.group(1))
            step_type = match.group(2).upper()
            duration_ms = int(match.group(3))
            step_seq = cell_content.split()[0] if ' ' in cell_content else cell_content

            return {
                'type': step_type,
                'step_num': step_num,
                'duration_ms': duration_ms,
                'step_seq': step_seq
            }
        
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

                return {
                    'type': step_type,
                    'step_num': step_num,
                    'duration_ms': duration_ms,
                    'step_seq': step_seq
                }
            except:
                return None
        
        return None

    def _preprocess_data(self):
        """
        Data preprocessing
        """
        print("\nPreprocessing data...")

        time_columns = ['add_tick', 'schedule_tick', 'prefill_done_tick', 'finish_tick']
        for col in time_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        initial_count = len(self.df)

        vllm_time_columns = [col for col in time_columns if col in self.df.columns]
        if vllm_time_columns:
            time_mask = self.df[vllm_time_columns].notna().all(axis=1)
            self.df = self.df[time_mask].copy()

        if all(col in self.df.columns for col in time_columns):
            valid_mask = (
                (self.df['schedule_tick'] >= self.df['add_tick']) &
                (self.df['prefill_done_tick'] >= self.df['schedule_tick']) &
                (self.df['finish_tick'] >= self.df['prefill_done_tick'])
            )
            self.df = self.df[valid_mask].copy()
        
        filtered_count = initial_count - len(self.df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} invalid records, {len(self.df)} remaining")

        if 'original_request_key' in self.df.columns:
            self.df['request_display'] = self.df['original_request_key'].apply(
                lambda x: f"req_{str(x)[-8:]}" if isinstance(x, str) and len(x) > 10 else str(x)
            )
        else:
            self.df['request_display'] = self.df.index.astype(str)
        
        if 'app_id' in self.df.columns:
            self.df['short_app_id'] = self.df['app_id'].apply(
                lambda x: f"app_{str(x)[-6:]}" if isinstance(x, str) and len(x) > 10 else str(x)
            )

        if 'original_request_key' in self.df.columns:
            self.df['request_key_clean'] = self.df['original_request_key'].apply(
                lambda x: str(x)[9:] if isinstance(x, str) and x.startswith('chatcmpl-') else str(x)
            )

        if self.step_data is not None and not self.step_data.empty and 'request_key_clean' in self.df.columns:
            print("\nCorrelating XLSX step data to main table...")

            agg_dict = {
                'step_type': list,
                'step_number': list,
                'step_duration_ms': list,
                'step_sequence': list
            }

            for col in self.step_data.columns:
                if col not in ['request_key', 'step_type', 'step_number', 'step_duration_ms', 'step_sequence']:
                    agg_dict[col] = lambda x: list(x) if len(x) > 0 else []

            step_summary = self.step_data.groupby('request_key').agg(agg_dict).reset_index()
            step_summary = step_summary.rename(columns={
                'request_key': 'request_key_clean',
                'step_type': 'step_types',
                'step_number': 'step_numbers',
                'step_duration_ms': 'step_durations_ms',
                'step_sequence': 'step_sequences'
            })

            self.df = pd.merge(self.df, step_summary, on='request_key_clean', how='left')

            matched_count = self.df['step_types'].notna().sum() if 'step_types' in self.df.columns else 0
            print(f"Correlation complete: {matched_count}/{len(self.df)} request step data")

            self._precompute_step_timelines()

    def _precompute_step_timelines(self):
        """
        Pre-compute step timelines for each request
        """
        print("Pre-computing step timelines...")

        total_steps = 0

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Computing step timelines"):
            request_key = row.get('request_key_clean')
            step_types = row.get('step_types', [])

            if not isinstance(step_types, list) or len(step_types) == 0:
                continue
        
            step_events = []

            if pd.notna(row.get('schedule_tick')) and pd.notna(row.get('prefill_done_tick')):
                prefill_steps = self._compute_prefill_steps(row)
                step_events.extend(prefill_steps)
            
            if pd.notna(row.get('prefill_done_tick')) and pd.notna(row.get('finish_tick')):
                decode_steps = self._compute_decode_steps(row)
                step_events.extend(decode_steps)
            
            if step_events:
                self.step_events_by_request[request_key] = step_events
                total_steps += len(step_events)

        print(f"Pre-computation complete, total {total_steps} step events")

    def _compute_prefill_steps(self, row):
        """
        Compute Prefill step timelines
        """
        steps = []
        step_types = row.get('step_types', [])

        if not isinstance(step_types, list):
            return steps
        
        prefill_indices = [i for i, step_type in enumerate(step_types) if step_type == 'P']
        if not prefill_indices:
            return steps
        
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
            step_num = step_numbers[idx] if idx < len(step_numbers) else idx + 1
            step_seq = step_sequences[idx] if idx < len(step_sequences) else f"P-step{idx + 1}"

            if idx < len(step_durations_ms):
                step_duration_seconds = step_durations_ms[idx] / 1000.0
            else:
                continue

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
        """
        Compute Decode step timelines
        """
        steps = []
        step_types = row.get('step_types', [])

        if not isinstance(step_types, list):
            return steps
        
        decode_indices = [i for i, step_type in enumerate(step_types) if step_type == 'D']
        if not decode_indices:
            return steps

        decode_start_time = row['prefill_done_tick']
        decode_end_time = row['finish_tick']

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
            step_num = step_numbers[idx] if idx < len(step_numbers) else idx + 1
            step_seq = step_sequences[idx] if idx < len(step_sequences) else f"D-step{idx + 1}"

            if idx < len(step_durations_ms):
                step_duration_seconds = step_durations_ms[idx] / 1000.0
            else:
                continue

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
        """
        Display data hierarchy statistics
        """
        if self.df is None or self.df.empty:
            return

        print("\n" + "=" * 70)
        print("Data Hierarchy Statistics")
        print("=" * 70)

        stats = {}
        if 'address' in self.df.columns:
            stats['unique_addresses'] = self.df['address'].nunique()
        if 'app_id' in self.df.columns:
            stats['unique_app_ids'] = self.df['app_id'].nunique()
        if 'original_request_key' in self.df.columns:
            stats['unique_requests'] = self.df['original_request_key'].nunique()

        print(f"Total records: {len(self.df)}")
        for key, value in stats.items():
            name = key.replace('unique_', '').replace('_', ' ')
            print(f"Unique {name}: {value}")

        if self.step_data is not None:
            print(f"\nStep data statistics:")
            print(f"Total steps: {len(self.step_data)}")
            if 'step_type' in self.step_data.columns:
                type_counts = self.step_data['step_type'].value_counts()
                for step_type, count in type_counts.items():
                    print(f" {step_type} steps: {count}")

        if self.worker_data is not None:
            print(f"\nWorker data statistics:")
            print(f"Worker records: {len(self.worker_data)}")
            if 'address_pid' in self.worker_data.columns:
                unique_pids = self.worker_data['address_pid'].nunique()
                print(f"Unique PIDs: {unique_pids}")
        
        print("=" * 70)

    def _calculate_microseconds(self, tick_value: float) -> int:
        """
        Convert timestamp to microseconds
        """
        if pd.isna(tick_value):
            return 0
        return int(tick_value * 1_000_000)

    def _calculate_duration_microseconds(self, start: float, end: float) -> int:
        """
        Calculate time interval (microseconds)
        """
        if pd.isna(start) or pd.isna(end):
            return 0
        duration = end - start
        return max(0, int(duration * 1_000_000))

    def _create_router_event(self, row: pd.Series, pid: str, tid: str) -> Optional[Dict]:
        """
        Create framework Router event
        """
        if 'start_time' not in row or pd.isna(row['start_time']):
            return None
        
        ts = self._calculate_microseconds(row['start_time'])

        if 'end_time_ts' in row and pd.notna(row['end_time']):
            dur = self._calculate_duration_microseconds(row['start_time'], row['end_time'])
        else:
            dur = 0

        router_tid = f"{tid}_router"

        return {
            "name": f"Router: {row['request_display']}",
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
                "end_time": float(row.get("end_time_ts", 0)) if 'end_time_ts' in row and pd.notna(
                    row['end_time_ts']) else 0
            }
        }

    def _create_schedule_event(self, row: pd.Series, pid: str, tid: str) -> Optional[Dict]:
        """
        Create scheduling event
        """
        if 'add_tick' not in row or 'schedule_tick' not in row:
            return None
        
        ts = self._calculate_microseconds(row['add_tick'])
        dur = self._calculate_duration_microseconds(row['add_tick'], row['schedule_tick'])

        schedule_tid = f"{tid}_schedule"

        return {
            "name": f"Schedule: {row['request_display']}",
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
                "wait_time_ms": round((row['schedule_tick'] - row['add_tick']) * 1_000, 2)
            }
        }

    def _create_step_event(self, row: pd.Series, step_info: Dict, pid: str, tid: str, idx: int) -> Optional[Dict]:
        """
        Create step event
        """
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
                "absolute_start_time": float(step_info.get("absolute_start", start_time)),
                "absolute_end_time": float(step_info.get("absolute_end", end_time)),
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
            }
        }

    def _create_total_execution_event(self, row: pd.Series, pid: str, tid: str) -> Optional[Dict]:
        """
        Create total execution event
        """
        if 'add_tick' not in row or 'finish_tick' not in row:
            return None

        ts = self._calculate_microseconds(row['add_tick'])
        dur = self._calculate_duration_microseconds(row['add_tick'], row['finish_tick'])

        total_tid = f"{tid}_total"

        return {
            "name": f"Total Execution: {row['request_display']}",
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
                "total_time_ms": round((row['finish_tick'] - row['add_tick']) * 1_000, 2),
                "schedule_time_ms": round((row['schedule_tick'] - row['add_tick']) * 1_000, 2),
                "prefill_time_ms": round((row['prefill_done_tick'] - row['schedule_tick']) * 1_000, 2),
                "decode_time_ms": round((row['finish_tick'] - row['prefill_done_tick']) * 1_000, 2),
            }
        }

    def _create_hierarchy_metadata_events(self):
        """
        Create hierarchical metadata events
        """
        print("\nCreating hierarchical metadata events...")

        if self.df is None or self.df.empty:
            return

        if 'address' in self.df.columns:
            unique_addresses = self.df['address'].unique()
            for address in unique_addresses:
                address_data = self.df[self.df['address'] == address]
                app_count = address_data['app_id'].nunique()
                request_count = len(address_data)

                self.metadata_events.append({
                    "name": "process_name",
                    "ph": "M",
                    "pid": str(address),
                    "args": {
                        "name": f"DP Instance: {address}",
                        "description": f"Processing node, serving {app_count} applications, {request_count} requests"
                    }
                })
            
        if 'app_id' in self.df.columns:
            unique_apps = self.df['app_id'].unique()
            thread_types = [
                ("router", "Router Phase"),
                ("schedule", "Schedule Phase"),
                ("total", "Total Execution Phase"),
                ("prefill", "Prefill Phase"),
                ("decode", "Decode Phase"),
            ]

            for app_id in unique_apps:
                short_app_id = f"app_{str(app_id)[-6:]}" if len(str(app_id)) > 10 else str(app_id)
                app_addresses = self.df[self.df['app_id'] == app_id]['address'].unique()

                for address in app_addresses:
                    for thread_key, thread_name in thread_types:
                        self.metadata_events.append({
                            "name": "thread_name",
                            "ph": "M",
                            "pid": str(address),
                            "tid": f"{app_id}_{thread_key}",
                            "args": {
                                "name": f"{short_app_id} - {thread_name}",
                                "app_id": str(app_id),
                                "address": str(address),
                                "stage": thread_key
                            }
                        })

        self.metadata_events.append({
            "name": "trace_metadata",
            "ph": "M",
            "args": {
                "trace_name": "VLLM Fine-grained Step Timeline (with timestamps)",
                "trace_description": "Complete analysis with Prefill and Decode fine-grained steps, with precise timestamps",
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
            }
        })    

    def generate_separated_phase_trace_events(self):
        """
        Generate phase-separated trace events
        """
        if self.df is None or self.df.empty:
            print("Warning: No data to generate events")
            return

        print("\nGenerating phase-separated Trace events (with fine-grained steps)...")
        print("Multiple phases of each request displayed on different tracks...")

        total_events = 0
        step_events_count = 0

        if 'address' in self.df.columns:
            for address, address_group in self.df.groupby('address'):
                pid = str(address)

                if 'app_id' in address_group.columns:
                    for app_id, app_group in address_group.groupby('app_id'):
                        tid = str(app_id)

                        for _, row in app_group.iterrows():
                            try:
                                router_event = self._create_router_event(row, pid, tid)
                                if router_event:
                                    self.trace_events.append(router_event)
                                    total_events += 1

                                schedule_event = self._create_schedule_event(row, pid, tid)
                                if schedule_event:
                                    self.trace_events.append(schedule_event)
                                    total_events += 1
                                
                                request_key = row.get('request_key_clean')
                                if request_key in self.step_events_by_request:
                                    step_list = self.step_events_by_request[request_key]
                                    for idx, step_info in enumerate(step_list):
                                        step_event = self._create_step_event(row, step_info, pid, tid, idx)
                                        if step_event:
                                            self.trace_events.append(step_event)
                                            total_events += 1
                                            step_events_count += 1
                                
                                total_event = self._create_total_execution_event(row, pid, tid)
                                if total_event:
                                    self.trace_events.append(total_event)
                                    total_events += 1
                                
                            except Exception as e:
                                request_display = row.get('request_display', 'unknown')
                                print(f"Warning: Error generating events for request {request_display}: {str(e)}")
                                continue

        print(f"\nGenerated {total_events} phase-separated Trace events")
        print(f" Including step events: {step_events_count}")

        self._create_hierarchy_metadata_events()

    def save_separated_phase_trace_json(self, output_file: str = "separated_phase_trace_view.json"):
        """
        Save phase-separated Trace events to JSON file
        """
        if not self.trace_events and not self.metadata_events:
            print("Warning: No event data to save")
            return

        print(f"\nSaving phase-separated Trace JSON to: {output_file}")

        try:
            all_events = self.trace_events + self.metadata_events

            all_events.sort(key=lambda x: x.get('ts', 0))

            trace_data = {
                "traceEvents": all_events,
                "displayTimeUnit": "ms",
                "otherData": {
                    "version": "5.0",
                    "generator": "VLLM Enhanced Fine-grained Step Trace Generator",
                    "description": "Complete performance analysis with Prefill and Decode fine-grained steps, with precise timestamps",
                    "separation_strategy": {
                        "router_phase": "cat='router', tid='{app_id}_router'",
                        "schedule_phase": "cat='schedule', tid='{app_id}_schedule'",
                        "prefill_steps": "cat='prefill_step_N', tid='{app_id}_prefill' (displayed on same row)",
                        "decode_steps": "cat='decode_step_N', tid='{app_id}_decode' (displayed on same row)",
                        "total_phase": "cat='total_execution', tid='{app_id}_total'",
                    },
                    "data_sources": {
                        "main_data": "request_statistics_enhanced.csv",
                        "step_data": "*_statistic.xlsx",
                        "worker_data": "*IntegratedWorker*.xlsx"
                    }
                }
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(all_events)} events to {output_file}")

            self._display_separated_phase_statistics(all_events)
        
        except Exception as e:
            raise Exception(f"Failed to save JSON file: {str(e)}")

    def _display_separated_phase_statistics(self, all_events: list):
        """
        Display phase-separated statistics
        """
        print("\n" + "=" * 70)
        print("Phase-separated Statistics")
        print("=" * 70)

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
        
        print(f"Total events: {len(all_events)}")
        
        for cat, count in category_stats.items():
            print(f"{cat}: {count} events")

        total_threads = sum(len(tids) for tids in tid_stats.values())
        print(f"\nTotal threads: {total_threads}")

        metadata_count = sum(1 for e in all_events if e.get('ph') == 'M')
        print(f"Metadata events: {metadata_count}")

        print("=" * 70)

    def generate_separated_phase_trace_view(self, output_file: str = "separated_phase_trace_view.json"):
        """
        Generate phase-separated Trace View
        """
        print("=" * 70)
        print("VLLM Enhanced Fine-grained Step Trace View Generator")
        print("=" * 70)

        try:
            self.load_enhanced_data()

            if self.df is None or self.df.empty:
                print("Warning: No valid data")
                return

            self.generate_separated_phase_trace_events()

            self.save_separated_phase_trace_json(output_file)

            print("\n" + "=" * 70)
            print("Enhanced fine-grained step Trace generation complete")
            print("=" * 70)
        
        except Exception as e:
            print(f"Error generating phase-separated Trace View: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    csv_file = r"\xxx\xxx\request_statistics_enhanced.csv"
    xlsx_dir = r"\xxx\xxx\64token"
    output_file = r"\xxx\xxx\trace_view_enhanced_with_timestamps_64token.json"

    generator = EnhancedHierarchicalTraceViewGenerator(csv_file, xlsx_dir)
    generator.generate_separated_phase_trace_view(output_file)


if __name__ == "__main__":
    main()
