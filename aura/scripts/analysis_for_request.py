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

import os
import json
import pandas as pd
import numpy as np
from glob import glob
import re
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class VLLMPerformanceAnalyzer:
    def __init__(self, schedule_dir: str = ".", stats_file: str = "request_statistics.csv"):
        """
        Initialize VLLMPerformanceAnalyzer class
        :param schedule_dir: Directory containing request_statistics.csv files
        :param stats_file: request_statistics.csv file name
        """
        self.schedule_dir = schedule_dir
        self.stats_file = stats_file
        self.vllm_data = None
        self.request_stats = None
        self.merged_data = None

    def load_vllm_schedule_files(self) -> pd.DataFrame:
        """
        Load and merge all vllm_schedule_* files
        """
        print("Loading VLLM scheduling performance files...")

        pattern = os.path.join(self.schedule_dir, "vllm_schedule_*.json")
        file_list = glob(pattern)

        if not file_list:
            raise FileNotFoundError(f"No vllm_schedule_* files found in directory {self.schedule_dir}")

        print(f"Found {len(file_list)} VLLM scheduling files")

        all_requests = []
        for file_path in file_list:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                timestamp = data.get('timestamp', 0)

                if 'request' in data:
                    requests_dict = data['request']
                    
                    for request_key, request_data in requests_dict.items():
                        record = {
                            'original_request_key': request_key,
                            'timestamp': timestamp,
                            'file_source': os.path.basename(file_path)
                        }

                        if request_data:
                            record.update(request_data)
                        all_requests.append(record)
                print(f"Processed: {os.path.basename(file_path)} - {len(requests_dict) if 'request' in data else 0} requests")

            except Exception as e:
                print(f"Warning: Error processing file {file_path}: {str(e)}")
                continue
        
        if not all_requests:
            raise ValueError("No request data parsed from any file")
        
        self.vllm_data = pd.DataFrame(all_requests)
        print(f"Done merging, total {len(self.vllm_data)} VLLM request records")
        
        self._clean_vllm_data()

        return self.vllm_data

    def _clean_vllm_data(self):
        """
        Clean and preprocess VLLM data
        """
        if self.vllm_data is None or self.vllm_data.empty:
            return
        
        print("Cleaning VLLM data...")

        # 1. Generate app_id for correlation (remove chatcmpl- prefix)
        def extract_app_id(request_key: str) -> str:
            """
            Extract application ID from request key
            """
            if not isinstance(request_key, str):
                return ""
            
            if request_key.startswith('chatcmpl-'):
                parts = request_key.split('-', 2)
                if len(parts) >= 3:
                    return f"{parts[1]}-{parts[2]}"
                else:
                    return request_key.replace('chatcmpl-', '')
            else:
                return request_key
        
        self.vllm_data['extracted_app_id'] = self.vllm_data['original_request_key'].apply(extract_app_id)

        # 2. Handle null values
        numeric_columns = ['add_tick', 'schedule_tick', 'prefill_done_tick', 'finish_tick', 'prompt_len', 'output_len']

        for col in numeric_columns:
            if col in self.vllm_data.columns:
                self.vllm_data[col] = pd.to_numeric(self.vllm_data[col], errors='coerce')

        # 3. Calculate time-related metrics (unit: seconds)
        self.vllm_data['add_to_schedule_ms'] = (self.vllm_data['schedule_tick'] - self.vllm_data['add_tick']) * 1000
        self.vllm_data['schedule_to_prefill_ms'] = (self.vllm_data['prefill_done_tick'] - self.vllm_data['schedule_tick']) * 1000
        self.vllm_data['prefill_to_finish_ms'] = (self.vllm_data['finish_tick'] - self.vllm_data['prefill_done_tick']) * 1000
        self.vllm_data['total_execution_ms'] = (self.vllm_data['finish_tick'] - self.vllm_data['add_tick']) * 1000

        # 4. Calculate TPOT (Time Per Output Token)
        # Formula: (finish_tick - prefill_done_tick) / (output_len - 1)
        def calculate_tpot(row):
            if pd.notna(row['finish_tick']) and pd.notna(row['prefill_done_tick']) and pd.notna(row['output_len']):
                if row['output_len'] > 1:
                    return (row['finish_tick'] - row['prefill_done_tick']) / (row['output_len'] - 1)
                elif row['output_len'] == 1:
                    return row['finish_tick'] - row['prefill_done_tick']
                return np.nan
        
        self.vllm_data['tpot_seconds'] = self.vllm_data.apply(calculate_tpot, axis=1)
        self.vllm_data['tpot_ms'] = self.vllm_data['tpot_seconds'] * 1000

        # 5. Calculate throughput-related metrics
        self.vllm_data['total_tokens'] = self.vllm_data['prompt_len'] + self.vllm_data['output_len']
        self.vllm_data['tokens_per_second'] = self.vllm_data['output_len'] / (self.vllm_data['finish_tick'] - self.vllm_data['add_tick'])

        print(f"Data cleaning complete, added {len(self.vllm_data.columns) - len(['original_request_key', 'timestamp', 'file_source'])} computed fields")

    def load_request_statistics(self) -> pd.DataFrame:
        """
        Load request statistics data
        """
        if not os.path.exists(self.stats_file):
            raise FileNotFoundError(f"Request statistics file does not exist: {self.stats_file}")

        print(f"Loading request statistics file: {self.stats_file}")

        try:
            self.request_stats = pd.read_csv(self.stats_file)
            print(f"Loading complete, total {len(self.request_stats)} request statistics records")
            
            if 'app_id' not in self.request_stats.columns:
                raise ValueError("Missing 'app_id' field in request statistics file")
            
            self.request_stats['app_id_cleaned'] = self.request_stats['app_id'].astype(str).str.strip()

            return self.request_stats
            
        except Exception as e:
            raise Exception(f"Error loading request statistics file: {str(e)}")

    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge VLLM data and request statistics data
        """
        if self.vllm_data is None:
            self.load_vllm_schedule_files()
        
        if self.request_stats is None:
            self.load_request_statistics()
        
        if self.vllm_data.empty or self.request_stats.empty:
            print("Warning: Data is empty, cannot merge")
            return pd.DataFrame()

        print("Merging datasets...")

        # Show data samples for debugging
        print("\nVLLM data sample (first 5 extracted_app_ids):")
        print(self.vllm_data[['original_request_key', 'extracted_app_id']].head())

        print("\nRequest statistics data sample (first 5 app_ids):")
        print(self.request_stats[['app_id', 'app_id_cleaned']].head())

        merged_data = None

        # Strategy 1: Merge directly using extracted_app_id and app_id_cleaned
        try:
            merged_data = pd.merge(
                self.request_stats,
                self.vllm_data,
                left_on='request_id',
                right_on='extracted_app_id',
                how='left',
                suffixes=('_stats', '_vllm')
            )

            matched_count = merged_data['extracted_app_id'].notna().sum()
            total_count = len(merged_data)
            match_rate = matched_count / total_count * 100

            print(f"Merge result: {matched_count}/{total_count} records matched successfully ({match_rate:.1f}%)")

            if matched_count == 0:
                print("Warning: No records matched successfully, trying alternative matching strategy...")
                # Strategy 2: Try a more lenient match
                merged_data = self._try_alternative_merge()
        
        except Exception as e:
            print(f"Error merging data: {str(e)}, trying fallback merge strategy...")
            merged_data = self._try_alternative_merge()
        
        self.merged_data = merged_data

        if self.merged_data is not None and not self.merged_data.empty:
            print(f"Merge complete, final data {len(self.merged_data)} records")
            
            self._calculate_comprehensive_metrics()
        
        return self.merged_data
    
    def _try_alternative_merge(self) -> pd.DataFrame:
        """
        Try fallback merge strategy
        """
        print("Trying fallback merge strategy...")

        # Strategy 2: Extract app_id from request_id for matching
        if 'request_id' in self.request_stats.columns:
            
            def extract_app_id_from_request(request_id: str) -> str:
                if isinstance(request_id, str):
                    parts = request_id.split('--')
                    if len(parts) > 0:
                        return parts[0]
                return request_id
            
            self.request_stats['extracted_from_request_id'] = self.request_stats['request_id'].apply(
                extract_app_id_from_request
            )
            
            merged = pd.merge(
                self.request_stats,
                self.vllm_data,
                left_on='extracted_from_request_id',
                right_on='extracted_app_id',
                how='left',
                suffixes=('_stats', '_vllm')
            )

            matched_count = merged['extracted_app_id'].notna().sum()
            print(f"Fallback merge strategy result: {matched_count}/{len(merged)} records")

            return merged
        
        return pd.DataFrame()
    
    def _calculate_comprehensive_metrics(self):
        """
        Calculate comprehensive performance metrics
        """
        if self.merged_data is None or self.merged_data.empty:
            return
        
        print("Calculating comprehensive performance metrics...")

        # 1. Time consistency check
        if 'duration_ms' in self.merged_data.columns and 'total_execution_ms' in self.merged_data.columns:
            self.merged_data['time_diff_ms'] = self.merged_data['duration_ms'] - self.merged_data['total_execution_ms']
            self.merged_data['time_diff_percent'] = (self.merged_data['time_diff_ms'] / self.merged_data['duration_ms'] * 100).replace([np.inf, -np.inf], np.nan)

        # 2. Performance classification
        def classify_tpot(tpot_ms):
            if pd.isna(tpot_ms):
                return 'unknown'
            elif tpot_ms < 10:
                return 'excellent'
            elif tpot_ms < 50:
                return 'good'
            elif tpot_ms < 100:
                return 'fair'
            else:
                return 'poor'
        
        if 'tpot_ms' in self.merged_data.columns:
            self.merged_data['tpot_category'] = self.merged_data['tpot_ms'].apply(classify_tpot)
        
        # 3. Efficiency metrics
        if 'prompt_len' in self.merged_data.columns and 'total_execution_ms' in self.merged_data.columns:
            self.merged_data['prompt_processing_speed'] = (
                self.merged_data['prompt_len'] / (
                    self.merged_data['prefill_done_tick'] - self.merged_data['add_tick']
                )
            ).replace([np.inf, -np.inf], np.nan)
        
        if 'output_len' in self.merged_data.columns and 'tpot_ms' in self.merged_data.columns:
            self.merged_data['output_speed_tps'] = 1000 / self.merged_data['tpot_ms']
        
        print('Comprehensive performance metrics calculation complete')

    def export_enhanced_statistics(self, output_file: str = "request_statistics_enhanced.csv"):
        """
        Export enhanced request statistics
        """

        if self.merged_data is None or self.merged_data.empty:
            print("Warning: No merged data, cannot export")
            return

        print(f"Exporting enhanced request statistics to {output_file}...")

        export_columns = [
            # Original statistics columns
            'request_id', 'app_id', 'address', 'start_time', 'end_time',
            'duration_seconds', 'duration_ms',

            # VLLM performance columns
            'add_tick', 'schedule_tick', 'prefill_done_tick', 'finish_tick',
            'prompt_len', 'output_len', 'total_tokens',
            
            # Calculated metrics
            'tpot_seconds', 'tpot_ms', 'tpot_category',
            'add_to_schedule_ms', 'schedule_to_prefill_ms', 'prefill_to_finish_ms', 
            'total_execution_ms', 'tokens_per_second',
            
            # Other
            'original_request_key', 'file_source',
        ]

        available_columns = [col for col in export_columns if col in self.merged_data.columns]

        self.merged_data[available_columns].to_csv(output_file, index=False, encoding='utf-8')
        print(f"Export complete, {len(available_columns)} columns, total {len(self.merged_data)} records")

        self.export_analysis_report(output_file.replace('.csv', '_report.txt'))

    def export_analysis_report(self, report_file: str = "performance_analysis_report.txt"):
        """
        Export performance analysis report
        """
        if self.merged_data is None or self.merged_data.empty:
            return
        
        print(f"Exporting performance analysis report to: {report_file}...")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("VLLM Performance Analysis Report\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data source directory: {self.schedule_dir}\n")
            f.write(f"Statistics file: {self.stats_file}\n\n")

            f.write("1. Data Overview\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total requests: {len(self.merged_data):,}\n")
        
            matched_count = self.merged_data['extracted_app_id'].notna().sum()
            match_rate = matched_count / len(self.merged_data) * 100
            f.write(f"Requests matched with VLLM data: {matched_count:,} ({match_rate:.1f}%)\n\n")

            if 'tpot_ms' in self.merged_data.columns:
                valid_tpot = self.merged_data['tpot_ms'].dropna()
                if len(valid_tpot) > 0:
                    f.write("2. TPOT (Time Per Output Token) Analysis\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Average TPOT: {valid_tpot.mean():.2f} ms/token\n")
                    f.write(f"TPOT Median: {valid_tpot.median():.2f} ms/token\n")
                    f.write(f"TPOT Min: {valid_tpot.min():.2f} ms/token\n")
                    f.write(f"TPOT Max: {valid_tpot.max():.2f} ms/token\n")
                    f.write(f"TPOT Std Dev: {valid_tpot.std():.2f} ms/token\n\n")

                    if 'tpot_category' in self.merged_data.columns:
                        f.write("TPOT Category Statistics:\n")
                        category_counts = self.merged_data['tpot_category'].value_counts()
                        for category, count in category_counts.items():
                            percentage = count / len(self.merged_data) * 100
                            f.write(f"{category}: {count:,} ({percentage:.1f}%)\n")
                    f.write("\n")
            
            time_columns = ['add_to_schedule_ms', 'schedule_to_prefill_ms', 'prefill_to_finish_ms']
            available_time_cols = [col for col in time_columns if col in self.merged_data.columns]

            if available_time_cols:
                f.write("3. Phase Duration Analysis (milliseconds)\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Phase':<25} {'Mean':<10} {'Median':<10} {'P95':<10}\n")
                f.write("-" * 60 + "\n")

                for col in available_time_cols:
                    valid_data = self.merged_data[col].dropna()
                    if len(valid_data) > 0:
                        mean_val = valid_data.mean()
                        median_val = valid_data.median()
                        p95_val = valid_data.quantile(0.95)
                        f.write(f"{col:<25} {mean_val:<10.2f} {median_val:<10.2f} {p95_val:<10.2f}\n")
                f.write("\n")

            if 'tokens_per_second' in self.merged_data.columns:
                valid_tps = self.merged_data['tokens_per_second'].dropna()
                if len(valid_tps) > 0:
                    f.write("4. Throughput Analysis\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Average throughput: {valid_tps.mean():.2f} tokens/s\n")
                    f.write(f"Median throughput: {valid_tps.median():.2f} tokens/s\n")
                    f.write(f"Peak throughput: {valid_tps.max():.2f} tokens/s\n\n")

            if 'address' in self.merged_data.columns and 'tpot_ms' in self.merged_data.columns:
                f.write("5. Performance Ranking by Processing Node\n")
                f.write("-" * 40 + "\n")

                valid_data = self.merged_data[self.merged_data['tpot_ms'].notna()]

                if not valid_data.empty:
                    address_stats = valid_data.groupby('address').agg({
                        'tpot_ms': ['mean', 'count'],
                        'total_execution_ms': 'mean'
                    }).round(2)

                    address_stats.columns = ['avg_tpot_ms', 'request_count', 'avg_execution_ms']
                    address_stats = address_stats.sort_values('avg_tpot_ms')

                    f.write(f"{'Address':<25} {'Avg TPOT':<12} {'Requests':<10} {'Avg Total':<12}\n")
                    f.write("-" * 60 + "\n")

                    for address,  row in address_stats.head(10).iterrows():
                        f.write(f"{address[:24]:<25} {row['avg_tpot_ms']:<12.2f} {row['request_count']:<10} {row['avg_execution_ms']:<12.2f}\n")

        print(f"Analysis report export complete: {report_file}")

        def analyze(self, output_file: str = "request_statistics_enhanced.csv"):
            """
            Execute the full analysis workflow
            """

            print("=" * 70)
            print("VLLM Performance Data Analysis Tool")
            print("=" * 70)

            try:
                # 1. Load VLLM data
                self.load_vllm_schedule_files()

                # 2. Load request statistics data
                self.load_request_statistics()

                # 3. Merge data
                self.merge_datasets()

                if self.merged_data is not None and not self.merged_data.empty:
                    # 4. Export enhanced data
                    self.export_enhanced_statistics(output_file)

                    # 5. Display key statistics
                    self._display_summary()
                else:
                    print("Warning: No data after merging, analysis terminated")
            
            except Exception as e:
                print(f"Error during analysis: {str(e)}")
                import traceback
                traceback.print_exc()
        
        def _display_summary(self):
            """
            Display analysis summary
            """
            if self.merged_data is None or self.merged_data.empty:
                return 
            
            print("\n" + "=" * 70)
            print("Analysis Summary")
            print("=" * 70)

            matched_count = self.merged_data['extracted_app_id'].notna().sum()
            match_rate = matched_count / len(self.merged_data) * 100

            print(f"Data matching: {matched_count}/{len(self.merged_data)} ({match_rate:.1f}%)")

            if 'tpot_ms' in self.merged_data.columns:
                valid_tpot = self.merged_data['tpot_ms'].dropna()
                if len(valid_tpot) > 0:
                    print(f"TPOT statistics: mean {valid_tpot.mean():.2f} ms/token, "
                          f"median {valid_tpot.median():.2f} ms/token")
            
            print(f"Enhanced data saved to: request_statistics_enhanced.csv")
            print(f"Detailed report saved to: performance_analysis_report.txt")


def main():
    """
    Main function
    """
    schedule_dir = r"\xxx\xxx\64token"
    stats_file = r"\xxx\xxx\request_statistics.csv"
    output_file = r"\xxx\xxx\request_statistics_enhanced.csv"

    analyzer = VLLMPerformanceAnalyzer(schedule_dir, stats_file)
    analyzer.analyze(output_file)


if __name__ == "__main__":
    main()
