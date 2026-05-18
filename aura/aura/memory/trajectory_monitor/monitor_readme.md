# 步骤

- 配置环境变量
    - 异步监控

        ```bash
        export TRAJECTORY_MONITOR_MODE=ASYNC
        ```

    - 串行监控

        ```bash
        export TRAJECTORY_MONITOR_MODE=SYNC
        ```

    - 监控默认关闭，或设置任何其他环境变量值

- 在traj_monitor.json中配置metrics及参数
- 在线轨迹检测在`rollout_worker.py`中实现：

```python
        ################## Trajetrocy Monitor ##################
        from trajectory_online_monitor import online_monitor
        # We use the responses for batched detection (List[List[int]])
        online_monitor.step(responses, iteration_id=self.iteration)
        ########################################################
```

# 监控指标

## 概览

- 每个监控指标独立检测
    - `window`指标计算每个滑动窗口的值，并最终通过每条轨迹的最小窗口值检测阈值
    - `window`指标会对每条估计的所有窗口值分布进行评估，和**最开始的100条轨迹**分布计算Wasserstein distance，并进行阈值检测
    - `global`指标对每条轨迹计算，并进行阈值检测

- 检测方式：
    - 一条轨迹被标记为`outlier`当且仅当它在至少一个指标下超出阈值范围
    - 对被标记为`outlier`的轨迹，计算其在所有指标下的zscore以衡量离群程度，并进行加权和，记为最终异常值
    - 每个Batch，打印所有异常轨迹id及异常值，按照异常值从高到低进行排序

- 指标参数：
    - 当前指标参数依照特定训练场景设定，其他任务按需微调

## 具体监控指标

- 基于n-gram的监测：
  例：对于输出的轨迹序列
  `[151644, 872, 198, 10234, 525, 27412, 3232, 29959, 4204, 537, 2677, 39318, 22823, 30, 5443, 264, 9442,  54046, 3110, 13, 151645, 198, 151644, 77091, 198, 151667, 198, 32313, 11, 773]`
  进行检测时，单个token id包含的信息过少且独立，失去了序列本身的统计学信息。因此通过n-gram的方式组合:

```text
# n=3
[
    [151644, 872, 198],
    [872, 198, 10234],
    [198, 10234, 525],
    [10234, 525, 27412],
    [525, 27412, 3232],
    [27412, 3232, 29959],
    [3232, 29959, 4204],
    ...
    [151667, 198, 32313],
    [198, 32313, 11],
    [32313, 11, 773],
]
```

- `distinct_n`：检测不重复的n-grams占总n-gram的比例
- `compression_ratio`：检测序列的压缩率
- `token_entropy`：检测序列中token频率的信息熵
- `vocab_gini`：检测token序列频率的基尼系数
- `intra_kld`：检测token序列自身的KL散度
- `LZ_complexity`：检测序列的Lempel–Ziv复杂度
