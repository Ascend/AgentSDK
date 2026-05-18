#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

"""
Parquet文件转换脚本
功能：读取parquet文件，修改字段名称，给某一列的内容加上前缀
"""

import pandas as pd
import argparse
import sys


def transform_parquet(
    input_file: str,
    output_file: str,
    columns: list = None,
    column_mapping: dict = None,
    prefix_column: str = None,
    prefix_value: str = "",
):
    """
    转换parquet文件

    参数:
        input_file: 输入parquet文件路径
        output_file: 输出parquet文件路径
        columns: 要读取的列名列表，如果指定则只保留这些列（其他字段不保留），如果为None则读取所有列
        column_mapping: 字段名称映射字典，格式：{旧列名: 新列名}
        prefix_column: 需要添加前缀的列名（使用重命名前的原始列名）
        prefix_value: 要添加的前缀值
    """
    try:
        # 读取parquet文件
        print(f"正在读取文件: {input_file}")
        if columns:
            print(f"只读取指定列（其他字段不保留）: {columns}")
            try:
                df = pd.read_parquet(input_file, columns=columns)
            except (KeyError, ValueError) as e:
                # 如果列不存在，读取所有列来显示可用列名
                all_df = pd.read_parquet(input_file)
                missing_columns = [col for col in columns if col not in all_df.columns]
                if missing_columns:
                    print(f"错误: 以下列不存在于文件中: {missing_columns}")
                    print(f"可用列名: {list(all_df.columns)}")
                    sys.exit(1)
                else:
                    raise e
        else:
            df = pd.read_parquet(input_file)
        print(f"成功读取，共 {len(df)} 行，{len(df.columns)} 列")
        print(f"保留的列名: {list(df.columns)}")

        # 验证prefix_column和column_mapping中的列是否在读取的列中
        if columns:
            if prefix_column and prefix_column not in columns:
                print(f"错误: --prefix-column 指定的列 '{prefix_column}' 不在 --columns 列表中")
                sys.exit(1)
            if column_mapping:
                missing_rename_cols = [old_col for old_col in column_mapping.keys() if old_col not in columns]
                if missing_rename_cols:
                    print(f"错误: 重命名中的以下列不在 --columns 列表中: {missing_rename_cols}")
                    sys.exit(1)

        # 给指定列添加前缀（在重命名之前，使用原始列名）
        if prefix_column and prefix_value:
            if prefix_column in df.columns:
                print(f"\n正在给列 '{prefix_column}' 添加前缀 '{prefix_value}'")
                df[prefix_column] = prefix_value + df[prefix_column].astype(str)
                print("前缀添加完成")
            else:
                print(f"警告: 列 '{prefix_column}' 不存在于数据中")
                print(f"可用列名: {list(df.columns)}")

        # 重命名字段（在添加前缀之后）
        if column_mapping:
            print(f"\n正在重命名字段: {column_mapping}")
            df = df.rename(columns=column_mapping)
            print(f"重命名后的列名: {list(df.columns)}")

        # 保存为parquet文件
        print(f"\n正在保存到: {output_file}")
        df.to_parquet(output_file, index=False)
        print("转换完成！")

        return df

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='转换parquet文件：重命名字段并给指定列添加前缀')
    parser.add_argument('input_file', type=str, help='输入parquet文件路径')
    parser.add_argument('output_file', type=str, help='输出parquet文件路径')
    parser.add_argument(
        '--rename',
        type=str,
        nargs='+',
        help='字段重命名，格式：旧列名:新列名，多个用空格分隔，例如：old_col1:new_col1 old_col2:new_col2',
    )
    parser.add_argument(
        '--prefix-column', type=str, dest='prefix_column', help='需要添加前缀的列名（使用重命名前的原始列名）'
    )
    parser.add_argument('--prefix', type=str, default='', help='要添加的前缀值（默认：空字符串）')
    parser.add_argument(
        '--columns',
        type=str,
        nargs='+',
        help='只读取指定的列（其他字段不保留），多个列名用空格分隔，例如：col1 col2 col3',
    )

    args = parser.parse_args()

    # 解析字段映射
    column_mapping = {}
    if args.rename:
        for rename_pair in args.rename:
            if ':' in rename_pair:
                old_name, new_name = rename_pair.split(':', 1)
                column_mapping[old_name] = new_name
            else:
                print(f"警告: 忽略无效的重命名格式 '{rename_pair}'，应使用 旧列名:新列名")

    try:
        result = transform_parquet(
            input_file=args.input_file,
            output_file=args.output_file,
            columns=args.columns if args.columns else None,
            column_mapping=column_mapping if column_mapping else None,
            prefix_column=args.prefix_column,
            prefix_value=args.prefix,
        )
        if result is None:
            print("Error: Transformation failed, no result generated")
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: Exception occurred during transformation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
