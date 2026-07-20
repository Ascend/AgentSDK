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
import json
import os
import argparse

def convert_parquet_to_filtered_jsonl(input_parquet, output_jsonl):
    """
    将 Parquet 转换为 JSONL 格式，提取特定字段。
    """
    print(f"正在读取 Parquet 文件: {input_parquet} ...")

    try:
        df = pd.read_parquet(input_parquet)
        records = df.to_dict('records')
    except Exception as e:
        print(f"读取 Parquet 失败: {e}")
        return

    print(f"读取到 {len(records)} 行数据，开始提取字段...")

    success_count = 0
    skipped_count = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for idx, data in enumerate(records):
            try:
                new_data = {
                    "data_source": data.get('data_source'),
                    "question": data['prompt'][0]['content'],
                    "answer": data['reward_model']['ground_truth'],
                    "labels": data['reward_model']['ground_truth']
                }
                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                success_count += 1
            except KeyError as e:
                skipped_count += 1
                print(f"跳过第 {idx} 行: 缺失字段 {e}")
            except Exception as e:
                skipped_count += 1
                print(f"跳过第 {idx} 行: 处理出错 {e}")

    print(f"处理完成！共 {len(records)} 行，成功 {success_count} 条，跳过 {skipped_count} 条，保存至: {output_jsonl}")
    if skipped_count > 0:
        print(f"警告: 共跳过 {skipped_count} 行数据，请检查输入 parquet 文件字段完整性")

def main():
    parser = argparse.ArgumentParser(description="将 Parquet 转换为 JSONL")
    parser.add_argument('--input', type=str, required=True, help='输入的 parquet 文件路径')
    parser.add_argument('--output', type=str, default='output.jsonl', help='输出的 jsonl 文件路径')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件 {args.input}")
        return

    convert_parquet_to_filtered_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()
