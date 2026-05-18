import os
import json
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class VLLMPerformanceAnalyzer:
    def __init__(self, schedule_dir: str = ".", stats_file: str = "request_statistics.csv"):
        """
        初始化VLLM性能分析器

        Args:
            schedule_dir: 包含vllm_schedule_*.json文件的目录
            stats_file: request_statistics.csv文件路径
        """
        self.schedule_dir = schedule_dir
        self.stats_file = stats_file
        self.vllm_data = None  # 合并后的VLLM数据
        self.request_stats = None  # 请求统计数据
        self.merged_data = None  # 合并后的数据

    def load_vllm_schedule_files(self) -> pd.DataFrame:
        """
        加载并合并所有vllm_schedule_*文件

        Returns:
            合并后的VLLM性能DataFrame
        """
        print("开始加载VLLM调度性能文件...")

        # 查找所有vllm_schedule_*文件
        pattern = os.path.join(self.schedule_dir, "vllm_schedule_*.json")
        file_list = glob(pattern)

        if not file_list:
            raise FileNotFoundError(f"在目录 {self.schedule_dir} 中未找到vllm_schedule_*文件")

        print(f"找到 {len(file_list)} 个VLLM调度文件")

        all_requests = []

        for file_path in file_list:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取timestamp和requests
                timestamp = data.get('timestamp', 0)

                # 处理requests字典
                if 'request' in data:
                    requests_dict = data['request']

                    for request_key, request_data in requests_dict.items():
                        # 创建记录
                        record = {
                            'original_request_key': request_key,
                            'timestamp': timestamp,
                            'file_source': os.path.basename(file_path),
                        }

                        # 添加request_data中的所有字段
                        if request_data:
                            record.update(request_data)

                        all_requests.append(record)

                print(
                    f"  已处理: {os.path.basename(file_path)} - {len(requests_dict) if 'request' in data else 0} 个请求"
                )

            except Exception as e:
                print(f"  警告: 处理文件 {file_path} 时出错: {str(e)}")
                continue

        if not all_requests:
            raise ValueError("未从任何文件中解析到请求数据")

        # 创建DataFrame
        self.vllm_data = pd.DataFrame(all_requests)
        print(f"✓ 合并完成，共 {len(self.vllm_data)} 个VLLM请求记录")

        # 数据清洗和转换
        self._clean_vllm_data()

        return self.vllm_data

    def _clean_vllm_data(self):
        """清洗和预处理VLLM数据"""
        if self.vllm_data is None or self.vllm_data.empty:
            return

        print("清洗VLLM数据...")

        # 1. 生成关联用的app_id（去掉chatcmpl-前缀）
        def extract_app_id(request_key):
            """从request_key中提取app_id"""
            if not isinstance(request_key, str):
                return ""

            # 去掉chatcmpl-前缀
            if request_key.startswith('chatcmpl-'):
                # 去掉前缀后，第一个-之后的部分就是app_id
                parts = request_key.split('-', 2)  # 最多分割2次
                if len(parts) >= 3:
                    return f"{parts[1]}-{parts[2]}"
                else:
                    return request_key.replace('chatcmpl-', '')
            else:
                return request_key

        self.vllm_data['extracted_app_id'] = self.vllm_data['original_request_key'].apply(extract_app_id)

        # 2. 处理空值
        numeric_columns = ['add_tick', 'schedule_tick', 'prefill_done_tick', 'finish_tick', 'prompt_len', 'output_len']

        for col in numeric_columns:
            if col in self.vllm_data.columns:
                # 转换类型并处理空值
                self.vllm_data[col] = pd.to_numeric(self.vllm_data[col], errors='coerce')

        # 3. 计算时间相关指标（单位为秒）
        self.vllm_data['add_to_schedule_ms'] = (self.vllm_data['schedule_tick'] - self.vllm_data['add_tick']) * 1000
        self.vllm_data['schedule_to_prefill_ms'] = (
            self.vllm_data['prefill_done_tick'] - self.vllm_data['schedule_tick']
        ) * 1000
        self.vllm_data['prefill_to_finish_ms'] = (
            self.vllm_data['finish_tick'] - self.vllm_data['prefill_done_tick']
        ) * 1000
        self.vllm_data['total_execution_ms'] = (self.vllm_data['finish_tick'] - self.vllm_data['add_tick']) * 1000

        # 4. 计算TPOT（Time Per Output Token）
        # 公式: (finish_tick - prefill_done_tick) / (output_len - 1)
        # 避免除零错误
        def calculate_tpot(row):
            if pd.notna(row['finish_tick']) and pd.notna(row['prefill_done_tick']) and pd.notna(row['output_len']):
                if row['output_len'] > 1:
                    return (row['finish_tick'] - row['prefill_done_tick']) / (row['output_len'] - 1)
                elif row['output_len'] == 1:
                    # 如果output_len为1，则使用整个prefill到finish的时间
                    return row['finish_tick'] - row['prefill_done_tick']
            return np.nan

        self.vllm_data['tpot_seconds'] = self.vllm_data.apply(calculate_tpot, axis=1)
        self.vllm_data['tpot_ms'] = self.vllm_data['tpot_seconds'] * 1000

        # 5. 计算吞吐量相关指标
        self.vllm_data['total_tokens'] = self.vllm_data['prompt_len'] + self.vllm_data['output_len']
        self.vllm_data['tokens_per_second'] = self.vllm_data['output_len'] / (
            self.vllm_data['finish_tick'] - self.vllm_data['add_tick']
        )

        print(
            f"✓ 数据清洗完成，新增 {len(self.vllm_data.columns) - len(['original_request_key', 'timestamp', 'file_source'])} 个计算字段"
        )

    def load_request_statistics(self) -> pd.DataFrame:
        """
        加载请求统计数据

        Returns:
            请求统计DataFrame
        """
        if not os.path.exists(self.stats_file):
            raise FileNotFoundError(f"请求统计文件不存在: {self.stats_file}")

        print(f"加载请求统计文件: {self.stats_file}")

        try:
            self.request_stats = pd.read_csv(self.stats_file)
            print(f"✓ 加载完成，共 {len(self.request_stats)} 条记录")

            # 确保app_id字段存在
            if 'app_id' not in self.request_stats.columns:
                raise ValueError("请求统计文件中缺少'app_id'字段")

            # 清理app_id字段
            self.request_stats['app_id_cleaned'] = self.request_stats['app_id'].astype(str).str.strip()

            return self.request_stats

        except Exception as e:
            raise Exception(f"加载请求统计文件失败: {str(e)}")

    def merge_datasets(self) -> pd.DataFrame:
        """
        合并VLLM数据和请求统计数据

        Returns:
            合并后的DataFrame
        """
        if self.vllm_data is None:
            self.load_vllm_schedule_files()

        if self.request_stats is None:
            self.load_request_statistics()

        if self.vllm_data.empty or self.request_stats.empty:
            print("警告: 数据为空，无法合并")
            return pd.DataFrame()

        print("开始合并数据集...")

        # 准备合并键
        # VLLM数据使用extracted_app_id
        # 请求统计数据使用app_id_cleaned

        # 显示数据示例，帮助调试
        print("\nVLLM数据示例 (前5个extracted_app_id):")
        print(self.vllm_data[['original_request_key', 'extracted_app_id']].head())

        print("\n请求统计数据示例 (前5个app_id):")
        print(self.request_stats[['app_id', 'app_id_cleaned']].head())

        # 尝试多种合并策略
        merged_data = None

        # 策略1: 直接使用extracted_app_id和app_id_cleaned合并
        try:
            merged_data = pd.merge(
                self.request_stats,
                self.vllm_data,
                # left_on='app_id_cleaned',
                left_on='request_id',
                right_on='extracted_app_id',
                how='left',
                suffixes=('_stats', '_vllm'),
            )

            matched_count = merged_data['extracted_app_id'].notna().sum()
            total_count = len(merged_data)
            match_rate = matched_count / total_count * 100

            print(f"合并结果: {matched_count}/{total_count} 条记录匹配成功 ({match_rate:.1f}%)")

            if matched_count == 0:
                print("警告: 没有记录匹配成功，尝试其他匹配策略...")
                # 策略2: 尝试更宽松的匹配
                merged_data = self._try_alternative_merge()

        except Exception as e:
            print(f"合并时出错: {str(e)}，尝试备用合并策略...")
            merged_data = self._try_alternative_merge()

        self.merged_data = merged_data

        if self.merged_data is not None and not self.merged_data.empty:
            print(f"✓ 合并完成，最终数据 {len(self.merged_data)} 条记录")

            # 计算综合性能指标
            self._calculate_comprehensive_metrics()

        return self.merged_data

    def _try_alternative_merge(self) -> pd.DataFrame:
        """
        尝试备用合并策略
        """
        print("尝试备用合并策略...")

        # 策略2: 从request_id中提取app_id进行匹配
        if 'request_id' in self.request_stats.columns:
            # 从request_id中提取类似app_id的部分
            def extract_app_id_from_request(request_id):
                if isinstance(request_id, str):
                    # 假设格式为 "X-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX--Y"
                    parts = request_id.split('--')
                    if len(parts) > 0:
                        return parts[0]
                return request_id

            self.request_stats['extracted_from_request_id'] = self.request_stats['request_id'].apply(
                extract_app_id_from_request
            )

            # 尝试合并
            merged = pd.merge(
                self.request_stats,
                self.vllm_data,
                left_on='extracted_from_request_id',
                right_on='extracted_app_id',
                how='left',
                suffixes=('_stats', '_vllm'),
            )

            matched_count = merged['extracted_app_id'].notna().sum()
            print(f"备用策略匹配结果: {matched_count}/{len(merged)} 条记录")

            return merged

        return pd.DataFrame()

    def _calculate_comprehensive_metrics(self):
        """计算综合性能指标"""
        if self.merged_data is None or self.merged_data.empty:
            return

        print("计算综合性能指标...")

        # 1. 时间一致性检查
        # 检查VLLM的total_execution_ms和请求统计的duration_ms是否一致
        if 'duration_ms' in self.merged_data.columns and 'total_execution_ms' in self.merged_data.columns:
            self.merged_data['time_diff_ms'] = self.merged_data['duration_ms'] - self.merged_data['total_execution_ms']
            self.merged_data['time_diff_percent'] = (
                self.merged_data['time_diff_ms'] / self.merged_data['duration_ms'] * 100
            ).replace([np.inf, -np.inf], np.nan)

        # 2. 性能分类
        # 根据TPOT对请求进行分类
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

        # 3. 效率指标
        if 'prompt_len' in self.merged_data.columns and 'total_execution_ms' in self.merged_data.columns:
            self.merged_data['prompt_processing_speed'] = (
                self.merged_data['prompt_len'] / (self.merged_data['prefill_done_tick'] - self.merged_data['add_tick'])
            ).replace([np.inf, -np.inf], np.nan)

        if 'output_len' in self.merged_data.columns and 'tpot_ms' in self.merged_data.columns:
            self.merged_data['output_speed_tps'] = 1000 / self.merged_data['tpot_ms']

        print("✓ 综合指标计算完成")

    def export_enhanced_statistics(self, output_file: str = "request_statistics_enhanced.csv"):
        """
        导出增强的请求统计数据

        Args:
            output_file: 输出文件名
        """
        if self.merged_data is None or self.merged_data.empty:
            print("警告: 没有合并数据可导出")
            return

        print(f"导出增强统计数据到: {output_file}")

        # 选择要导出的列（可以根据需要调整）
        export_columns = [
            # 原始统计列
            'request_id',
            'app_id',
            'address',
            'start_time',
            'end_time',
            'duration_seconds',
            'duration_ms',
            # VLLM性能列
            'add_tick',
            'schedule_tick',
            'prefill_done_tick',
            'finish_tick',
            'prompt_len',
            'output_len',
            'total_tokens',
            # 计算指标
            'tpot_seconds',
            'tpot_ms',
            'tpot_category',
            'add_to_schedule_ms',
            'schedule_to_prefill_ms',
            'prefill_to_finish_ms',
            'total_execution_ms',
            'tokens_per_second',
            # 其他
            'original_request_key',
            'file_source',
        ]

        # 只保留实际存在的列
        available_columns = [col for col in export_columns if col in self.merged_data.columns]

        # 导出数据
        self.merged_data[available_columns].to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ 已导出 {len(available_columns)} 列数据，共 {len(self.merged_data)} 条记录")

        # 同时导出分析报告
        self.export_analysis_report(output_file.replace('.csv', '_report.txt'))

    def export_analysis_report(self, report_file: str = "performance_analysis_report.txt"):
        """导出性能分析报告"""
        if self.merged_data is None or self.merged_data.empty:
            return

        print(f"生成性能分析报告: {report_file}")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("VLLM性能分析报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据源目录: {self.schedule_dir}\n")
            f.write(f"统计文件: {self.stats_file}\n\n")

            # 数据概览
            f.write("1. 数据概览\n")
            f.write("-" * 40 + "\n")
            f.write(f"总请求数: {len(self.merged_data):,}\n")

            matched_count = self.merged_data['extracted_app_id'].notna().sum()
            match_rate = matched_count / len(self.merged_data) * 100
            f.write(f"匹配VLLM数据的请求: {matched_count:,} ({match_rate:.1f}%)\n\n")

            # TPOT分析
            if 'tpot_ms' in self.merged_data.columns:
                valid_tpot = self.merged_data['tpot_ms'].dropna()
                if len(valid_tpot) > 0:
                    f.write("2. TPOT (Time Per Output Token) 分析\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"平均TPOT: {valid_tpot.mean():.2f} ms/token\n")
                    f.write(f"TPOT中位数: {valid_tpot.median():.2f} ms/token\n")
                    f.write(f"TPOT最小值: {valid_tpot.min():.2f} ms/token\n")
                    f.write(f"TPOT最大值: {valid_tpot.max():.2f} ms/token\n")
                    f.write(f"TPOT标准差: {valid_tpot.std():.2f} ms/token\n\n")

                    # TPOT分类统计
                    if 'tpot_category' in self.merged_data.columns:
                        f.write("TPOT分类统计:\n")
                        category_counts = self.merged_data['tpot_category'].value_counts()
                        for category, count in category_counts.items():
                            percentage = count / len(self.merged_data) * 100
                            f.write(f"  {category}: {count:,} ({percentage:.1f}%)\n")
                    f.write("\n")

            # 各阶段耗时分析
            time_columns = ['add_to_schedule_ms', 'schedule_to_prefill_ms', 'prefill_to_finish_ms']
            available_time_cols = [col for col in time_columns if col in self.merged_data.columns]

            if available_time_cols:
                f.write("3. 各阶段耗时分析 (毫秒)\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'阶段':<25} {'平均':<10} {'中位数':<10} {'P95':<10}\n")
                f.write("-" * 60 + "\n")

                for col in available_time_cols:
                    valid_data = self.merged_data[col].dropna()
                    if len(valid_data) > 0:
                        mean_val = valid_data.mean()
                        median_val = valid_data.median()
                        p95_val = valid_data.quantile(0.95)
                        f.write(f"{col:<25} {mean_val:<10.2f} {median_val:<10.2f} {p95_val:<10.2f}\n")
                f.write("\n")

            # 吞吐量分析
            if 'tokens_per_second' in self.merged_data.columns:
                valid_tps = self.merged_data['tokens_per_second'].dropna()
                if len(valid_tps) > 0:
                    f.write("4. 吞吐量分析\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"平均吞吐量: {valid_tps.mean():.2f} tokens/秒\n")
                    f.write(f"中位数吞吐量: {valid_tps.median():.2f} tokens/秒\n")
                    f.write(f"峰值吞吐量: {valid_tps.max():.2f} tokens/秒\n\n")

            # 按地址的性能分析
            if 'address' in self.merged_data.columns and 'tpot_ms' in self.merged_data.columns:
                f.write("5. 按处理节点的性能排名\n")
                f.write("-" * 40 + "\n")

                # 筛选有VLLM数据的记录
                valid_data = self.merged_data[self.merged_data['tpot_ms'].notna()]

                if not valid_data.empty:
                    address_stats = (
                        valid_data.groupby('address')
                        .agg({'tpot_ms': ['mean', 'count'], 'total_execution_ms': 'mean'})
                        .round(2)
                    )

                    # 重命名列
                    address_stats.columns = ['avg_tpot_ms', 'request_count', 'avg_execution_ms']
                    address_stats = address_stats.sort_values('avg_tpot_ms')

                    f.write(f"{'地址':<25} {'平均TPOT':<12} {'请求数':<10} {'平均总耗时':<12}\n")
                    f.write("-" * 60 + "\n")

                    for address, row in address_stats.head(10).iterrows():
                        f.write(
                            f"{address[:24]:<25} {row['avg_tpot_ms']:<12.2f} "
                            f"{row['request_count']:<10} {row['avg_execution_ms']:<12.2f}\n"
                        )

        print(f"✓ 分析报告已保存到: {report_file}")

    def analyze(self, output_file: str = "request_statistics_enhanced.csv"):
        """
        执行完整的分析流程

        Args:
            output_file: 输出文件名
        """
        print("=" * 70)
        print("VLLM性能数据分析工具")
        print("=" * 70)

        try:
            # 1. 加载VLLM数据
            self.load_vllm_schedule_files()

            # 2. 加载请求统计数据
            self.load_request_statistics()

            # 3. 合并数据
            self.merge_datasets()

            if self.merged_data is not None and not self.merged_data.empty:
                # 4. 导出增强数据
                self.export_enhanced_statistics(output_file)

                # 5. 显示关键统计信息
                self._display_summary()
            else:
                print("警告: 合并后没有数据，分析终止")

        except Exception as e:
            print(f"分析过程中出错: {str(e)}")
            import traceback

            traceback.print_exc()

    def _display_summary(self):
        """显示分析摘要"""
        if self.merged_data is None or self.merged_data.empty:
            return

        print("\n" + "=" * 70)
        print("分析摘要")
        print("=" * 70)

        matched_count = self.merged_data['extracted_app_id'].notna().sum()
        match_rate = matched_count / len(self.merged_data) * 100

        print(f"✓ 数据匹配: {matched_count}/{len(self.merged_data)} ({match_rate:.1f}%)")

        if 'tpot_ms' in self.merged_data.columns:
            valid_tpot = self.merged_data['tpot_ms'].dropna()
            if len(valid_tpot) > 0:
                print(f"✓ TPOT统计: 平均 {valid_tpot.mean():.2f} ms/token, 中位数 {valid_tpot.median():.2f} ms/token")

        print("✓ 增强数据已保存到: request_statistics_enhanced.csv")
        print("✓ 详细报告已保存到: performance_analysis_report.txt")


def main():
    """主函数"""
    # 配置参数
    schedule_dir = r"\xxx\xxx\64token"  # vllm_schedule_*.json文件所在目录
    stats_file = r"\xxx\xxx\request_statistics.csv"  # 请求统计文件
    output_file = r"\xxx\xxx\request_statistics_enhanced.csv"  # 输出文件

    # 创建分析器并执行分析
    analyzer = VLLMPerformanceAnalyzer(schedule_dir, stats_file)
    analyzer.analyze(output_file)


if __name__ == "__main__":
    main()
