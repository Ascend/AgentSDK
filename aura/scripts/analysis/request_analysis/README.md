# 性能分析脚本使用指南

本目录包含三个用于性能分析和可视化的Python脚本，需要按顺序执行以完成完整的分析流程。

## 📋 目录

- [脚本概述](#脚本概述)
- [执行顺序](#执行顺序)
- [详细说明](#详细说明)
    - [步骤1: analysis_for_dp.py](#步骤1-analysis_for_dppy)
    - [步骤2: analysis_for_request.py](#步骤2-analysis_for_requestpy)
    - [步骤3: analysis_for_traceview.py](#步骤3-analysis_for_traceviewpy)
- [依赖环境](#依赖环境)
- [常见问题](#常见问题)

---

## 脚本概述

| 脚本                          | 功能                      | 主要输出                              |
|-----------------------------|-------------------------|-----------------------------------|
| `analysis_for_dp.py`        | 解析日志文件，计算请求统计信息         | `request_statistics.csv`          |
| `analysis_for_request.py`   | 合并VLLM性能数据与请求统计         | `request_statistics_enhanced.csv` |
| `analysis_for_traceview.py` | 生成Chrome Tracing可视化JSON | `trace_view.json`                 |

---

## 执行顺序

**重要：必须按照以下顺序执行脚本**

```text
1. analysis_for_dp.py
   ↓
2. analysis_for_request.py
   ↓
3. analysis_for_traceview.py
```

---

## 详细说明

### 步骤1: analysis_for_dp.py

#### 功能描述

解析日志文件，提取请求的开始和结束时间，计算每个请求的耗时、完成率等统计信息。

#### 输入文件

- **日志文件** (`log_file_path`): 包含请求开始/结束记录的日志文件
    - 格式要求：日志行需包含 `chat start time` 或 `chat end time` 关键字
    - 示例格式：

      ```text
      2025-12-10 11:58:24,436|INFO|router.py|chat():208|trajectory performance status, chat start time:1765339104.436777, appID:48-183753f1-2011-4a8f-9824-2908e58733ea681391, address:0.0.0.0:60549-4, request_id:48-183753f1-2011-4a8f-9824-2908e58733ea681391--2
      ```

#### 输出文件

在指定的 `output_dir` 目录下生成以下文件：

1. **`parsed_log_data.csv`**
    - 解析后的原始日志数据
    - 包含字段：`log_time`, `event_type`, `event_time`, `app_id`, `address`, `request_id` 等

2. **`request_statistics.csv`** ⭐ **关键文件**
    - 每个请求的统计信息
    - 包含字段：`request_id`, `app_id`, `address`, `start_time`, `end_time`, `duration_ms`, `status` 等
    - **此文件是下一步的必需输入**

3. **`address_performance.csv`**
    - 按处理节点（address）的性能统计
    - 包含：总请求数、完成率、平均耗时、吞吐量等

4. **`app_distribution.csv`**
    - 按应用（app_id）的分布统计
    - 包含：请求数、节点数、平均耗时等

5. **`analysis_summary.txt`**
    - 文本格式的分析摘要报告

#### 使用方法

**修改脚本中的路径配置**（在 `main()` 函数中）：

```python
log_file_path = "/xxxx/xxxx/64token/logs_1765472433858.log"
output_path = "/xxxx/xxxx/64token"
```

**执行脚本**：

```bash
python analysis_for_dp.py
```

#### 输出示例

```text
日志文件分析工具
==================================================
✓ 成功读取日志文件: logs_1765472433858.log (大小: 1234567 字符)
开始解析日志数据...
  进度: 100/1000 行 (已解析: 50 条记录)
✓ 解析完成，共 500 条有效记录
✓ 统计完成，共 250 个请求
✓ 地址分析完成，共 5 个地址
✓ 应用分析完成，共 10 个应用
```

---

### 步骤2: analysis_for_request.py

#### 功能描述

加载VLLM调度性能文件（`vllm_schedule_*.json`），与步骤1生成的请求统计数据合并，生成增强的统计数据，包含TPOT（Time Per Output
Token）等性能指标。

#### 输入文件

1. **`request_statistics.csv`** ⭐ **必需**
    - 由步骤1生成
    - 必须包含 `app_id` 或 `request_id` 字段

2. **`vllm_schedule_*.json` 文件** ⭐ **必需**
    - 位于 `schedule_dir` 目录中
    - 文件名格式：`vllm_schedule_*.json`
    - 包含VLLM调度性能数据
    - 示例结构：

      ```json
      {
        "timestamp": 1765339104.436777,
        "request": {
          "chatcmpl-48-183753f1-2011-4a8f-9824-2908e58733ea681391--2": {
            "add_tick": 1765339104.436777,
            "schedule_tick": 1765339104.500000,
            "prefill_done_tick": 1765339105.000000,
            "finish_tick": 1765339106.500000,
            "prompt_len": 100,
            "output_len": 50
          }
        }
      }
      ```

#### 输出文件

1. **`request_statistics_enhanced.csv`** ⭐ **关键文件**
    - 增强的请求统计数据
    - 包含原始统计字段 + VLLM性能字段
    - 新增字段：
        - `add_tick`, `schedule_tick`, `prefill_done_tick`, `finish_tick`
        - `prompt_len`, `output_len`, `total_tokens`
        - `tpot_ms` (Time Per Output Token)
        - `add_to_schedule_ms`, `schedule_to_prefill_ms`, `prefill_to_finish_ms`
        - `total_execution_ms`, `tokens_per_second`
    - **此文件是下一步的必需输入**

2. **`performance_analysis_report.txt`**
    - 详细的性能分析报告
    - 包含TPOT分析、各阶段耗时分析、吞吐量分析等

#### 使用方法

**修改脚本中的路径配置**（在 `main()` 函数中）：

```python
schedule_dir = "/xxxx/xxxx/64token"  # vllm_schedule_*.json文件所在目录
stats_file = "/xxxx/xxxx/64token/request_statistics.csv"  # 请求统计文件
output_file = "/xxxx/xxxx/64token/request_statistics_enhanced.csv"
```

**执行脚本**：

```bash
python analysis_for_request.py
```

#### 输出示例

```text
======================================================================
VLLM性能数据分析工具
======================================================================
开始加载VLLM调度性能文件...
找到 5 个VLLM调度文件
  已处理: vllm_schedule_001.json - 50 个请求
✓ 合并完成，共 250 个VLLM请求记录
✓ 数据清洗完成，新增 15 个计算字段
加载请求统计文件: request_statistics.csv
✓ 加载完成，共 250 条记录
开始合并数据集...
合并结果: 230/250 条记录匹配成功 (92.0%)
✓ 合并完成，最终数据 250 条记录
✓ TPOT统计: 平均 45.23 ms/token, 中位数 42.10 ms/token
```

---

### 步骤3: analysis_for_traceview.py

#### 功能描述

生成Chrome Tracing格式的JSON文件，用于在Chrome浏览器中可视化性能数据。包含细粒度的Prefill和Decode步骤信息。

#### 输入文件

1. **`request_statistics_enhanced.csv`** ⭐ **必需**
    - 由步骤2生成
    - 必须包含以下字段：
        - `address`, `app_id`, `original_request_key`
        - `add_tick`, `schedule_tick`, `prefill_done_tick`, `finish_tick`

2. **`*_statistic.xlsx` 文件** ⭐ **必需**
    - 位于 `xlsx_dir` 目录中
    - 文件名格式：`*_statistic.xlsx`
    - 包含细粒度步骤信息（Prefill和Decode步骤）
    - 文件结构：
        - 第1行第1列：`request_key`（请求键）
        - 第2行第1列：`address_pid`（格式：`IP:pid` 或 `pid=数字`）
        - 第3行：步骤信息（格式：`t1-p-b1 100` 表示步骤1，Prefill类型，耗时100ms）

3. **`*IntegratedWorker*.csv` 文件** ⭐ **可选但推荐**
    - 位于 `xlsx_dir` 目录中
    - 文件名格式：`*IntegratedWorker*.csv`
    - 包含Worker级别的详细性能数据
    - 用于增强步骤数据的详细信息

#### 输出文件

1. **`trace_view.json`** ⭐ **最终目标文件**
    - Chrome Tracing格式的JSON文件
    - 可在Chrome浏览器中打开（访问 `chrome://tracing`）
    - 包含以下事件类型：
        - **Router阶段** (`cat='router'`): 框架总执行阶段
        - **调度阶段** (`cat='schedule'`): 请求调度等待时间
        - **Prefill步骤** (`cat='prefill'`): Prefill细粒度步骤
        - **Decode步骤** (`cat='decode'`): Decode细粒度步骤
        - **总执行阶段** (`cat='total_execution'`): 完整执行时间线

2. **`all_IntegratedWorker.xlsx`**（如果提供了IntegratedWorker文件）
    - 合并后的Worker数据文件

#### 使用方法

**修改脚本中的路径配置**（在 `main()` 函数中）：

```python
csv_file = "xxxx/xxxx/request_statistics_enhanced.csv"
xlsx_dir = "xxxx/xxxx/64token"  # 包含 *_statistic.xlsx 和 *IntegratedWorker*.csv 的目录
output_file = "xxxx/xxxx/trace_view.json"
```

**执行脚本**：

```bash
python analysis_for_traceview.py
```

#### 输出示例

```text
======================================================================
VLLM增强细粒度步骤Trace View生成器
======================================================================
数据字段关系说明
======================================================================
加载增强请求统计文件: request_statistics_enhanced.csv
✓ 加载成功，共 250 条记录

======================================================================
加载XLSX细粒度统计文件
======================================================================
找到 10 个XLSX统计文件
✓ 解析完成，共 5000 个步骤记录
  步骤类型分布:
    P: 2500 个步骤
    D: 2500 个步骤

======================================================================
加载IntegratedWorker文件
======================================================================
找到 5 个IntegratedWorker文件
✓ 解析完成，共 5000 条worker记录

生成阶段分离的Trace事件（包含细粒度步骤）...
✓ 生成 12500 个分离阶段Trace事件
  其中步骤事件: 5000 个

保存阶段分离Trace JSON到: trace_view.json
✓ 已保存 12550 个事件到 trace_view.json
```

#### 可视化方法

1. **打开Chrome浏览器**
2. **访问** `chrome://tracing`
3. **点击 "Load" 按钮**
4. **选择生成的 `trace_view.json` 文件**
5. **查看可视化结果**

可视化界面说明：

- **进程（PID）**: 每个处理节点（address）为一个进程
- **线程（TID）**: 每个应用（app_id）的每个阶段为一个线程
    - `{app_id}_router`: Router阶段
    - `{app_id}_schedule`: 调度阶段
    - `{app_id}_prefill`: Prefill步骤
    - `{app_id}_decode`: Decode步骤
    - `{app_id}_total`: 总执行阶段

---

## 依赖环境

### Python版本

- Python 3.7+

### 必需Python包

```bash
pip install pandas numpy matplotlib openpyxl tqdm
```

### 包说明

- `pandas`: 数据处理和分析
- `numpy`: 数值计算
- `matplotlib`: 绘图（analysis_for_dp.py使用）
- `openpyxl`: 读取Excel文件（trace_view_analysis使用）
- `tqdm`: 进度条显示

---

## 常见问题

### Q1: 执行 analysis_for_dp.py 时提示"未解析到任何有效记录"

**原因**: 日志文件格式不匹配
**解决**:

- 检查日志文件是否包含 `chat start time` 或 `chat end time` 关键字
- 检查日志格式是否符合脚本中的正则表达式模式
- 查看 `parse_log_line()` 方法中的模式定义

### Q2: 执行 analysis_for_request.py 时匹配率为0%

**原因**: app_id或request_id格式不匹配
**解决**:

- 检查 `request_statistics.csv` 中的 `app_id` 或 `request_id` 格式
- 检查 `vllm_schedule_*.json` 中的 `request_key` 格式
- 查看脚本输出的数据示例，确认格式是否一致
- 脚本会自动尝试备用匹配策略

### Q3: 执行 trace_view_analysis 时提示缺少必要字段

**原因**: `request_statistics_enhanced.csv` 缺少必需字段
**解决**:

- 确认已正确执行步骤1和步骤2
- 检查CSV文件是否包含以下字段：
    - `address`, `app_id`, `original_request_key`
    - `add_tick`, `schedule_tick`, `prefill_done_tick`, `finish_tick`

### Q4: XLSX文件解析失败

**原因**: 文件格式不符合预期
**解决**:

- 确认XLSX文件格式：
    - 第1行第1列：request_key
    - 第2行第1列：包含 `pid=` 的地址信息
    - 第3行：步骤信息（格式：`t1-p-b1 100`）

### Q5: 生成的trace_view.json在Chrome中无法正常显示

**原因**: JSON格式或时间戳问题
**解决**:

- 检查JSON文件格式是否正确
- 确认时间戳字段（`ts`, `dur`）为有效的数值
- 查看脚本输出的统计信息，确认事件数量是否正常

### Q6: 执行速度慢

**优化建议**:

- 对于大型数据集，可以考虑：
    - 在 `trace_view_analysis` 中取消注释数据限制行（`self.df = self.df.iloc[:500]`）进行测试
    - 使用更快的存储设备（SSD）
    - 增加系统内存

---

## 数据流程图

```text
日志文件 (logs_*.log)
    ↓
[步骤1: analysis_for_dp.py]
    ↓
request_statistics.csv
    ↓
[步骤2: analysis_for_request.py]
    ↓ (合并 vllm_schedule_*.json)
request_statistics_enhanced.csv
    ↓
[步骤3: analysis_for_traceview.py]
    ↓ (合并 *_statistic.xlsx 和 *IntegratedWorker*.csv)
trace_view.json ⭐ 最终输出
```

---

## 联系与支持

如有问题或建议，请检查：

1. 脚本中的路径配置是否正确
2. 输入文件格式是否符合要求
3. 依赖包是否已正确安装
4. Python版本是否兼容

---

**最后更新**: 2025-12-17
**版本**: 1.0
