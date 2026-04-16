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

import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import os
from typing import List, Dict, Optional


class LogAnalyzer:
    def __init__(self, log_file_path: str = None):
        """
        初始化日志分析器

        Args:
            log_file_path: 日志文件路径，如果为None则使用示例数据
        """
        self.log_file_path = log_file_path
        self.df = None  # 解析后的数据
        self.request_stats = None  # 请求统计信息
        self.address_stats = None  # 地址统计信息
        self.app_stats = None  # 应用统计信息

    def read_log_file(self, file_path: str = None) -> str:
        """
        读取日志文件

        Args:
            file_path: 日志文件路径，如果为None则使用初始化时指定的路径

        Returns:
            日志文件内容字符串
        """
        if file_path is None:
            file_path = self.log_file_path

        if file_path is None:
            raise ValueError("未提供日志文件路径")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"日志文件不存在: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✓ 成功读取日志文件: {file_path} (大小: {len(content)} 字符)")
            return content
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
                print(f"✓ 成功读取日志文件 (GBK编码): {file_path}")
                return content
            except:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                print(f"✓ 成功读取日志文件 (Latin-1编码): {file_path}")
                return content
        except Exception as e:
            raise Exception(f"读取日志文件失败: {str(e)}")

    def parse_log_line(self, line: str) -> Optional[Dict]:
        """
        解析单行日志

        Args:
            line: 单行日志内容

        Returns:
            解析后的字典，如果解析失败返回None
        """
        # 清理ANSI颜色代码
        line = re.sub(r'\x1b\[[0-9;]*[mK]', '', line)

        # 模式1: 包含时间戳、appID、address、request_id的完整格式
        pattern1 = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\|.*?chat (start|end) time:([\d.]+), appID:([\w-]+), address:([\w.:-]+), request_id:([\w-]+)'

        # 模式2: 更宽松的匹配模式，用于不同格式的日志
        pattern2 = r'chat (start|end) time:([\d.]+).*?appID:([\w-]+).*?address:([\w.:-]+).*?request_id:([\w-]+)'

        # 首先尝试精确匹配
        match = re.search(pattern1, line)
        if not match:
            # 尝试宽松匹配
            match = re.search(pattern2, line)
            if not match:
                return None
            log_time_str = None
            _, timestamp, app_id, address, request_id = match.groups()
            event_type = _.lower()  # start 或 end
        else:
            log_time_str, _, timestamp, app_id, address, request_id = match.groups()
            event_type = _.lower()

        try:
            # 解析日志时间
            if log_time_str:
                log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S,%f')
            else:
                log_time = None

            # 解析事件时间戳（支持秒和毫秒级）
            timestamp_float = float(timestamp)
            event_time = float(timestamp)
            # if timestamp_float > 4102416000:  # 2100年以后的认为是毫秒
            #     event_time = datetime.fromtimestamp(timestamp_float / 1000)
            # else:
            #     event_time = datetime.fromtimestamp(timestamp_float)

            # 提取基础地址（去掉端口号）
            base_address = address.split(':')[0] if ':' in address else address

            return {
                'log_time': log_time,
                'event_type': event_type,
                'event_time': event_time,
                'timestamp': timestamp_float,
                'app_id': app_id,
                'address': address,
                'base_address': base_address,
                'request_id': request_id,
                'raw_line': line[:200]  # 保留原始行前200个字符
            }
        except Exception as e:
            print(f"警告: 解析行失败 - {str(e)}: {line[:100]}...")
            return None

    def parse_log_data(self, log_content: str = None) -> pd.DataFrame:
        """
        解析日志数据

        Args:
            log_content: 日志内容字符串，如果为None则从文件读取

        Returns:
            解析后的DataFrame
        """
        if log_content is None:
            if self.log_file_path:
                log_content = self.read_log_file()
            else:
                raise ValueError("未提供日志内容或文件路径")

        print("开始解析日志数据...")
        records = []
        lines = log_content.strip().split('\n')
        total_lines = len(lines)

        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue

            record = self.parse_log_line(line)
            if record:
                records.append(record)

            # 显示进度
            if i % 100 == 0 or i == total_lines:
                print(f"  进度: {i}/{total_lines} 行 (已解析: {len(records)} 条记录)")

        if not records:
            raise ValueError("未解析到任何有效记录，请检查日志格式")

        self.df = pd.DataFrame(records)
        print(f"✓ 解析完成，共 {len(self.df)} 条有效记录")

        # 按时间排序
        self.df = self.df.sort_values('event_time')

        return self.df

    def calculate_request_statistics(self) -> pd.DataFrame:
        """
        计算请求统计信息

        Returns:
            请求统计DataFrame
        """
        if self.df is None:
            raise ValueError("请先解析日志数据")

        print("计算请求统计信息...")

        # 按request_id分组
        request_groups = {}

        for request_id, group in self.df.groupby('request_id'):
            start_events = group[group['event_type'] == 'start']
            end_events = group[group['event_type'] == 'end']

            if len(start_events) == 0:
                continue

            # 获取基本信息
            start_event = start_events.iloc[0]
            app_id = start_event['app_id']
            address = start_event['address']
            start_time = start_event['event_time']

            # 查找对应的结束事件
            end_time = None
            duration_seconds = None
            status = 'incomplete'

            if len(end_events) > 0:
                # 找到最接近的开始时间之后的结束事件
                for _, end_event in end_events.iterrows():
                    if end_event['event_time'] >= start_time:
                        end_time = end_event['event_time']
                        duration_seconds = end_time - start_time
                        # duration_seconds = (end_time - start_time).total_seconds()
                        status = 'completed'
                        break

            request_groups[request_id] = {
                'app_id': app_id,
                'address': address,
                'base_address': start_event['base_address'],
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration_seconds,
                'duration_ms': duration_seconds * 1000 if duration_seconds else None,
                'status': status,
                'start_timestamp': start_event['timestamp'],
                'has_end_event': len(end_events) > 0,
                'total_events': len(group)
            }

        self.request_stats = pd.DataFrame.from_dict(request_groups, orient='index')
        self.request_stats.index.name = 'request_id'

        print(f"✓ 统计完成，共 {len(self.request_stats)} 个请求")
        return self.request_stats

    def analyze_address_performance(self) -> pd.DataFrame:
        """
        分析地址性能统计

        Returns:
            地址性能统计DataFrame
        """
        if self.request_stats is None:
            self.calculate_request_statistics()

        print("分析地址性能...")

        address_stats = []

        for address, group in self.request_stats.groupby('address'):
            completed_requests = group[group['status'] == 'completed']
            total_requests = len(group)

            if len(completed_requests) > 0:
                avg_duration = completed_requests['duration_ms'].mean()
                min_duration = completed_requests['duration_ms'].min()
                max_duration = completed_requests['duration_ms'].max()
                throughput = len(completed_requests) / (
                        group['start_time'].max() - group['start_time'].min()) if len(
                    group) > 1 else 0
            else:
                avg_duration = min_duration = max_duration = throughput = 0

            address_stats.append({
                'address': address,
                'base_address': group['base_address'].iloc[0],
                'total_requests': total_requests,
                'completed_requests': len(completed_requests),
                'completion_rate': len(completed_requests) / total_requests if total_requests > 0 else 0,
                'avg_duration_ms': avg_duration,
                'min_duration_ms': min_duration,
                'max_duration_ms': max_duration,
                'throughput_rps': throughput,
                'first_request_time': group['start_time'].min(),
                'last_request_time': group['start_time'].max()
            })

        self.address_stats = pd.DataFrame(address_stats)

        # 按请求数排序
        self.address_stats = self.address_stats.sort_values('total_requests', ascending=False)

        print(f"✓ 地址分析完成，共 {len(self.address_stats)} 个地址")
        return self.address_stats

    def analyze_app_distribution(self) -> pd.DataFrame:
        """
        分析应用分布情况

        Returns:
            应用分布统计DataFrame
        """
        if self.request_stats is None:
            self.calculate_request_statistics()

        print("分析应用分布...")

        app_stats = []

        for app_id, group in self.request_stats.groupby('app_id'):
            unique_addresses = group['address'].unique()
            unique_base_addresses = group['base_address'].unique()
            total_requests = len(group)
            completed_requests = len(group[group['status'] == 'completed'])

            # 计算应用的时间跨度
            time_span = (group['start_time'].max() - group['start_time'].min())

            if completed_requests > 0:
                avg_duration = group[group['status'] == 'completed']['duration_ms'].mean()
            else:
                avg_duration = 0

            app_stats.append({
                'app_id': app_id,
                'total_requests': total_requests,
                'completed_requests': completed_requests,
                'completion_rate': completed_requests / total_requests if total_requests > 0 else 0,
                'unique_addresses': len(unique_addresses),
                'unique_base_addresses': len(unique_base_addresses),
                'address_list': list(unique_addresses),
                'base_address_list': list(unique_base_addresses),
                'avg_duration_ms': avg_duration,
                'time_span_seconds': time_span,
                'first_request_time': group['start_time'].min(),
                'last_request_time': group['start_time'].max()
            })

        self.app_stats = pd.DataFrame(app_stats)

        # 按请求数排序
        self.app_stats = self.app_stats.sort_values('total_requests', ascending=False)

        print(f"✓ 应用分析完成，共 {len(self.app_stats)} 个应用")
        return self.app_stats

    def export_results(self, output_dir: str = "log_analysis_results"):
        """
        导出分析结果到文件

        Args:
            output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\n导出分析结果到目录: {output_dir}")

        # 1. 导出解析后的原始数据
        if self.df is not None:
            csv_path = os.path.join(output_dir, "parsed_log_data.csv")
            self.df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ 已保存解析数据到: {csv_path}")

        # 2. 导出请求统计
        if self.request_stats is not None:
            csv_path = os.path.join(output_dir, "request_statistics.csv")
            self.request_stats.to_csv(csv_path, encoding='utf-8')
            print(f"✓ 已保存请求统计数据到: {csv_path}")

        # 3. 导出地址性能统计
        if self.address_stats is not None:
            csv_path = os.path.join(output_dir, "address_performance.csv")
            self.address_stats.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ 已保存地址性能数据到: {csv_path}")

        # 4. 导出应用分布统计
        if self.app_stats is not None:
            csv_path = os.path.join(output_dir, "app_distribution.csv")
            self.app_stats.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ 已保存应用分布数据到: {csv_path}")

        # 5. 导出分析摘要
        self.export_summary_report(output_dir)

        print(f"\n所有结果已保存到目录: {os.path.abspath(output_dir)}")

    def export_summary_report(self, output_dir: str):
        """导出分析摘要报告"""
        report_path = os.path.join(output_dir, "analysis_summary.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("日志数据分析报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            if self.df is not None:
                f.write(f"\n1. 数据概览:\n")
                f.write(f"   总日志记录数: {len(self.df)}\n")
                f.write(f"   时间范围: {self.df['event_time'].min()} 到 {self.df['event_time'].max()}\n")
                f.write(
                    f"   时间跨度: {(self.df['event_time'].max() - self.df['event_time'].min()):.2f} 秒\n")

            if self.request_stats is not None:
                f.write(f"\n2. 请求统计:\n")
                f.write(f"   总请求数: {len(self.request_stats)}\n")
                completed = len(self.request_stats[self.request_stats['status'] == 'completed'])
                f.write(f"   已完成请求: {completed} ({completed / len(self.request_stats) * 100:.1f}%)\n")

                if completed > 0:
                    durations = self.request_stats[self.request_stats['status'] == 'completed']['duration_ms']
                    f.write(f"   平均耗时: {durations.mean():.2f} ms\n")
                    f.write(f"   最小耗时: {durations.min():.2f} ms\n")
                    f.write(f"   最大耗时: {durations.max():.2f} ms\n")

            if self.address_stats is not None:
                f.write(f"\n3. 节点性能排名 (按请求数):\n")
                for i, (_, row) in enumerate(self.address_stats.iterrows(), 1):
                    f.write(f"   {i:2d}. {row['address']:<25} 请求数: {row['total_requests']:3d} "
                            f"完成率: {row['completion_rate'] * 100:5.1f}% "
                            f"平均耗时: {row['avg_duration_ms']:7.1f} ms\n")

            if self.app_stats is not None:
                f.write(f"\n4. 应用分布 (按请求数):\n")
                top_apps = self.app_stats.head(10)  # 显示前10个应用
                for i, (_, row) in enumerate(top_apps.iterrows(), 1):
                    f.write(f"   {i:2d}. {row['app_id'][:30]:<30}... 请求数: {row['total_requests']:3d} "
                            f"节点数: {row['unique_addresses']:2d}\n")

        print(f"✓ 已保存分析摘要到: {report_path}")


def main():
    """主函数"""
    print("日志文件分析工具")
    print("=" * 50)

    # 1. 指定日志文件路径
    log_file_path = r"\xxx\xxx\logs_1765472433858.log"  # 修改为你的日志文件路径
    output_path = r"\xxx\xxx\64token"  # 修改为你的日志文件路径

    # 如果文件不存在，创建一个示例文件
    if not os.path.exists(log_file_path):
        print(f"日志文件不存在: {log_file_path}")
        print("创建示例日志文件...")

    #         example_log = """[36m(AgentExecutor pid=681, ip=172.16.2.219)[0m 2025-12-10 11:58:24,436|INFO|router.py|chat():208|trajectory performance status, chat start time:1765339104.436777, appID:48-183753f1-2011-4a8f-9824-2908e58733ea681391, address:172.16.7.88:60549-4, request_id:48-183753f1-2011-4a8f-9824-2908e58733ea681391--2
    # [36m(AgentExecutor pid=681, ip=172.16.2.219)[0m 2025-12-10 11:58:24,504|INFO|router.py|chat():208|trajectory performance status, chat start time:1765339104.504231, appID:63-3261031d-5cc3-4363-a761-d7d0e95aad4a681505, address:172.16.7.88:60549-4, request_id:63-3261031d-5cc3-4363-a761-d7d0e95aad4a681505--2
    # [36m(AgentExecutor pid=681, ip=172.16.2.219)[0m 2025-12-10 11:58:25,503|INFO|router.py|chat():215|trajectory performance status, chat end time:1765339105.50344, appID:62-d676a49b-fb99-4b67-8920-b771ef356133681503, address:172.16.7.88:60549-5, request_id:62-d676a49b-fb99-4b67-8920-b771ef356133681503--2"""

    # with open(log_file_path, 'w', encoding='utf-8') as f:
    #     f.write(example_log)
    # print(f"✓ 已创建示例日志文件: {log_file_path}")

    # 2. 创建分析器实例
    analyzer = LogAnalyzer(log_file_path)

    try:
        # 3. 解析日志数据
        df = analyzer.parse_log_data()

        # 4. 计算请求统计
        request_stats = analyzer.calculate_request_statistics()

        # 5. 分析地址性能
        address_stats = analyzer.analyze_address_performance()

        # 6. 分析应用分布
        app_stats = analyzer.analyze_app_distribution()

        # 7. 导出结果
        analyzer.export_results(output_path)

        # 8. 显示关键信息
        print("\n" + "=" * 50)
        print("关键分析结果:")
        print("=" * 50)

        print(f"\n1. 最繁忙的节点:")
        for i, (_, row) in enumerate(address_stats.head(3).iterrows(), 1):
            print(f"   {i}. {row['address']}: {row['total_requests']} 请求, "
                  f"平均耗时: {row['avg_duration_ms']:.1f} ms")

        print(f"\n2. 最活跃的应用:")
        for i, (_, row) in enumerate(app_stats.head(3).iterrows(), 1):
            print(f"   {i}. {row['app_id'][:30]}...: {row['total_requests']} 请求, "
                  f"分布在 {row['unique_addresses']} 个节点")

        print(f"\n3. 系统整体性能:")
        completed_requests = request_stats[request_stats['status'] == 'completed']
        if len(completed_requests) > 0:
            avg_duration = completed_requests['duration_ms'].mean()
            total_time = (request_stats['start_time'].max() - request_stats['start_time'].min())
            throughput = len(completed_requests) / total_time if total_time > 0 else 0
            print(f"   平均请求耗时: {avg_duration:.1f} ms")
            print(f"   系统吞吐量: {throughput:.2f} 请求/秒")

        print("\n分析完成！")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()