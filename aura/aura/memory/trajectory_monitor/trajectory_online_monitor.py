#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import time
import json
import zlib
import math
import multiprocessing
import numpy as np
from collections import Counter
from scipy.stats import wasserstein_distance
from datetime import datetime

now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
LOG_FILENAME = f"/monitor_events_{now_str}.log"

# --- METRIC REGISTRY & FUNCTIONS (Kept exactly as your offline script) ---
_METRIC_REGISTRY = {}


def register_metric(name: str):
    def decorator(func):
        _METRIC_REGISTRY[name] = func
        return func

    return decorator


@register_metric("distinct_n")
def get_distinct_n(tokens, n=3):
    all_grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []
    return len(set(all_grams)) / len(all_grams) if all_grams else 0.0


@register_metric("compression_ratio")
def get_compression_ratio(tokens):
    if not tokens:
        return 1.0
    raw = (" ".join(map(str, tokens))).encode("utf-8")
    return len(zlib.compress(raw)) / max(1, len(raw))


@register_metric("token_entropy")
def get_token_entropy(tokens):
    counts = Counter(tokens)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs) / (math.log2(total) + 1e-9)


@register_metric("vocab_gini")
def get_vocab_gini(tokens):
    counts = Counter(tokens)
    values = np.sort(np.array(list(counts.values())))
    if len(values) < 2:
        return 0
    n = len(values)
    return (2 * np.sum((np.arange(1, n + 1) * values))) / (n * values.sum()) - (n + 1) / n


@register_metric("intra_kld")
def get_intra_kld(tokens):  # dangerous zones (10,\infity)
    tokens = np.array(tokens)  # Ensure input is numpy array
    mid = len(tokens) // 2
    part1 = tokens[:mid]
    part2 = tokens[mid:]

    # Determine size for bincount based on max token ID
    max_id = max(part1.max(), part2.max())

    counts1 = np.bincount(part1, minlength=max_id + 1)
    counts2 = np.bincount(part2, minlength=max_id + 1)

    mask = (counts1 > 0) | (counts2 > 0)
    p = counts1[mask].astype(np.float64)
    q = counts2[mask].astype(np.float64)

    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / (q + 1e-12) + 1e-12))


@register_metric("LZ_complexity")
def get_LZ_complexity(tokens):
    if not tokens:
        return 0.0

    n = len(tokens)
    substrings = set()

    i = 0
    k = 1
    count = 0

    while i + k <= n:
        sub = tuple(tokens[i : i + k])

        if sub in substrings:
            k += 1
        else:
            substrings.add(sub)
            count += 1
            i += k
            k = 1

    return count / n


def get_ngrams(tokens, n=3):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []


# --- UTILITY FUNCTIONS ---
def compute_metric_wrapper(tokens, func, **kwargs):
    # Extract known metric-specific args, ignore window args here
    n_arg = kwargs.get('n')

    if n_arg is not None:
        return func(tokens, n_arg)
    return func(tokens)


def moving_window_compute(tokens, func, window_size=None, stride=None, **kwargs):
    '''
    Computes metric.
    If window_size is None, computes globally on the whole sequence.
    kwargs captures extra params like 'n' to pass to the metric function.
    '''
    val = []

    # 1. Global Computation
    if window_size is None:
        val.append(compute_metric_wrapper(tokens, func, **kwargs))
        return val

    # 2. Windowed Computation
    if stride is None:
        stride = window_size // 2

    for i in range(0, max(1, len(tokens) - window_size + 1), stride):
        windowed_token = tokens[i : i + window_size]

        val.append(compute_metric_wrapper(windowed_token, func, **kwargs))

    return val


# Remove the segements of `<tool_response>` as they are not generated directly by the model
def remove_segments(data: list) -> list:
    START_VAL, END_VAL = 151665, 151666
    result = []
    in_seg = False
    for item in data:
        if item == START_VAL:
            in_seg = True
        elif item == END_VAL:
            in_seg = False
        elif not in_seg:
            result.append(item)
    return result


# =========================================
##### Vanilla Online Monitor Class #####
# =========================================
class TrajectoryOnlineMonitor:
    def __init__(self, config_path="traj_monitor.json"):
        self.config = self._load_config(config_path)
        self.reference_samples = {}
        self.initialized = False
        self.outliers = {}
        self.results = {key: [] for key in self.config["metrics"]}
        # Initialize wasserstein distance lists for windowed metrics
        for key in self.config["metrics"]:
            if "window" in key:
                self.results[key + "_wd"] = []

    def _load_config(self, path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {"metrics": {}}

    def step(self, batch_trajectories, iteration_id=0):
        if not self.config.get("metrics"):
            return

        # Pre-processing to remove tool response
        if self.config.get("remove_tool_response", False):
            batch_trajectories = [remove_segments(t) for t in batch_trajectories]

        # Compute Metrics
        batch_results = {}
        batch_scalar = {}
        for unique_name, params in self.config["metrics"].items():
            metric_type = params.get("type", unique_name)
            if metric_type not in _METRIC_REGISTRY:
                continue

            func = _METRIC_REGISTRY[metric_type]

            # Iterate through every trajectory and apply the metric (windowed or global)
            batch_results[unique_name] = [self._compute_single_traj(t, func, params) for t in batch_trajectories]

            # Save to global results
            batch_scalar[unique_name] = [item.min() for item in batch_results[unique_name]]
            self.results[unique_name] += batch_scalar[unique_name]

        # Initialize Reference (First Batch)
        if not self.initialized:
            print(f"[Monitor] Iter {iteration_id}: INITIALIZING REFERENCE (100 samples).")
            self.reference_samples = {
                key: np.concatenate(value[:100]) for key, value in batch_results.items() if "window" in key
            }
            self.initialized = True

        # Detect Outliers
        self._detect_and_report(batch_scalar, batch_results, iteration_id)

    def _compute_single_traj(self, tokens, func, params):
        """
        Smart dispatcher that handles Global vs Windowed logic
        and filters kwargs to prevent TypeError.
        """
        # Filter Arguments
        meta_keys = {'type', 'thresholds', 'wd', 'window_size', 'stride'}
        func_kwargs = {k: v for k, v in params.items() if k not in meta_keys}

        # Check Windowing
        w_size = params.get("window_size")
        stride = params.get("stride")

        if w_size and stride:
            # --- Windowed Logic ---
            vals = []
            for i in range(0, max(1, len(tokens) - w_size + 1), stride):
                window = tokens[i : i + w_size]
                vals.append(func(window, **func_kwargs))

            return np.array(vals)
        else:
            # --- Global Logic ---
            return np.array(func(tokens, **func_kwargs))

    def _detect_windowed_outliers(self, outliers, name, values):
        wd_limit = self.config["metrics"][name].get("wd")
        ref_dist = self.reference_samples.get(name)
        for idx in range(len(values)):
            if len(values[idx]) > 0:
                wd_val = wasserstein_distance(values[idx], ref_dist)
            else:
                wd_val = np.nan
            # Save WD to results
            self.results[name + "_wd"].append(wd_val)
            if wd_val > wd_limit:
                outliers[idx] = 0
        return outliers

    def _detect_and_report(self, batch_scalar, batch_results, iteration_id):
        print(f"\n--- Monitor Report (Iter {iteration_id}) ---")
        has_issues = False
        outliers = {}
        for name, values in batch_results.items():
            params = self.config["metrics"][name]
            thresholds = params.get("thresholds", {})

            # Threshold Check
            min_t = thresholds.get("min", -float('inf'))
            max_t = thresholds.get("max", float('inf'))

            outlier_ids = np.where((np.array(batch_scalar[name]) < min_t) | (np.array(batch_scalar[name]) > max_t))[0]

            if len(outliers) > 0:
                for idx in outlier_ids:
                    outliers[idx] = 0

            # Wasserstein Check
            if "window" in name:
                outliers = self._detect_windowed_outliers(outliers, name, values)

        if len(outliers) == 0:
            print("All metrics within limits.")
            return

        for idx_in_batch in outliers:
            for name in self.results:
                idx_global = idx_in_batch + iteration_id * self.config["bs"]
                if "wd" in name:
                    outliers[idx_in_batch] += abs(
                        (self.results[name][idx_global] - np.mean(self.results[name])) / np.std(self.results[name]) / 5
                    )
                else:
                    outliers[idx_in_batch] += abs(
                        (self.results[name][idx_global] - np.mean(self.results[name])) / np.std(self.results[name])
                    )
                self.outliers[idx_global] = outliers[idx_in_batch]
        for outlier, zscore in outliers.items():
            idx_global = outlier + iteration_id * self.config["bs"]
            print(
                f"[OUTLIER][Iteration {iteration_id:3}][Global idx {idx_global:5}][Batch idx {outlier:3}] Abnormality Score: {zscore:.4f}"
            )


# =========================================
##### Monitor Logic Handler #####
# =========================================


class _MonitorLogicHandler:
    def __init__(self, config_path="traj_config.json"):
        self.config = self._load_config(config_path)
        self.reference_samples = {}
        self.initialized = False
        self.outliers = {}
        self.results = {key: [] for key in self.config["metrics"]}
        # Initialize wasserstein distance lists for windowed metrics
        for key in self.config["metrics"]:
            if "window" in key:
                self.results[key + "_wd"] = []

    def _load_config(self, path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {"metrics": {}}

    def _log(self, message, mode="a"):
        """Appends to the log file created by the main process."""
        print(message)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(LOG_FILENAME, mode) as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception:
            pass  # Never crash the worker

    def process_batch(self, batch_trajectories, iteration_id=0):
        if not self.config.get("metrics"):
            return

        # Pre-processing to remove tool response
        if self.config.get("remove_tool_response", False):
            batch_trajectories = [remove_segments(t) for t in batch_trajectories]

        # Compute Metrics
        batch_results = {}
        batch_scalar = {}
        for unique_name, params in self.config["metrics"].items():
            metric_type = params.get("type", unique_name)
            if metric_type not in _METRIC_REGISTRY:
                continue

            func = _METRIC_REGISTRY[metric_type]

            # Iterate through every trajectory and apply the metric (windowed or global)
            batch_results[unique_name] = [self._compute_single_traj(t, func, params) for t in batch_trajectories]

            # Save to global results
            batch_scalar[unique_name] = [item.min() for item in batch_results[unique_name]]
            self.results[unique_name] += batch_scalar[unique_name]

        # Initialize Reference (First Batch)
        if not self.initialized:
            self._log(f"[Monitor] Iter {iteration_id}: INITIALIZING REFERENCE (100 samples).", "w")
            self.reference_samples = {
                key: np.concatenate(value[:100]) for key, value in batch_results.items() if "window" in key
            }
            self.initialized = True

        # Detect Outliers
        self._detect_and_report(batch_scalar, batch_results, iteration_id)

    def _compute_single_traj(self, tokens, func, params):
        """
        Smart dispatcher that handles Global vs Windowed logic
        and filters kwargs to prevent TypeError.
        """
        # Filter Arguments
        meta_keys = {'type', 'thresholds', 'wd', 'window_size', 'stride'}
        func_kwargs = {k: v for k, v in params.items() if k not in meta_keys}

        # Check Windowing
        w_size = params.get("window_size")
        stride = params.get("stride")

        if w_size and stride:
            # --- Windowed Logic ---
            vals = []
            for i in range(0, max(1, len(tokens) - w_size + 1), stride):
                window = tokens[i : i + w_size]
                vals.append(func(window, **func_kwargs))

            return np.array(vals)
        else:
            # --- Global Logic ---
            return np.array(func(tokens, **func_kwargs))

    def _detect_windowed_outliers(self, outliers, name, values):
        wd_limit = self.config["metrics"][name].get("wd")
        ref_dist = self.reference_samples.get(name)
        for idx in range(len(values)):
            if len(values[idx]) > 0:
                wd_val = wasserstein_distance(values[idx], ref_dist)
            else:
                wd_val = np.nan
            # Save WD to results
            self.results[name + "_wd"].append(wd_val)
            if wd_val > wd_limit:
                outliers[idx] = 0
        return outliers

    def _detect_and_report(self, batch_scalar, batch_results, iteration_id):
        self._log(f"\n--- Monitor Report (Iter {iteration_id}) ---")
        has_issues = False
        outliers = {}
        for name, values in batch_results.items():
            params = self.config["metrics"][name]
            thresholds = params.get("thresholds", {})

            # Threshold Check
            min_t = thresholds.get("min", -float('inf'))
            max_t = thresholds.get("max", float('inf'))

            outlier_ids = np.where((np.array(batch_scalar[name]) < min_t) | (np.array(batch_scalar[name]) > max_t))[0]

            if len(outliers) > 0:
                for idx in outlier_ids:
                    outliers[idx] = 0

            # Wasserstein Check
            if "window" in name:
                outliers = self._detect_windowed_outliers(outliers, name, values)

        if len(outliers) == 0:
            self._log("  All metrics within limits.")
            return

        for idx_in_batch in outliers:
            for name in self.results:
                idx_global = idx_in_batch + iteration_id * self.config["bs"]
                if "wd" in name:
                    outliers[idx_in_batch] += abs(
                        (self.results[name][idx_global] - np.mean(self.results[name])) / np.std(self.results[name]) / 5
                    )
                else:
                    outliers[idx_in_batch] += abs(
                        (self.results[name][idx_global] - np.mean(self.results[name])) / np.std(self.results[name])
                    )
            self.outliers[idx_global] = outliers[idx_in_batch]
        outliers = dict(sorted(outliers.items(), key=lambda item: item[1], reverse=True))
        for outlier, zscore in outliers.items():
            idx_global = outlier + iteration_id * self.config["bs"]
            self._log(
                f"[OUTLIER][Iteration {iteration_id:3}][Global idx {idx_global:5}][Batch idx {outlier:3}] Abnormality Score: {zscore:.4f}"
            )


# =========================================
##### Individual Monitors #####
# =========================================


# Synchronised
class SynchronousMonitor:
    def __init__(self, config_path="../../memory/trajectory_monitor/traj_monitor.json"):
        self.logic = _MonitorLogicHandler(config_path)

    def step(self, batch_trajectories, iteration_id=0):
        self.logic.process_batch(batch_trajectories, iteration_id)


# Asynchronised
class AsyncMonitor:
    def __init__(self, config_path="../../memory/trajectory_monitor/traj_monitor.json"):
        self.config_path = config_path
        self.queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=self._background_entry_point, args=(self.queue, self.config_path), daemon=True
        )
        self.process.start()
        print(f"[System] Async Monitor started. PID: {self.process.pid}")
        print(f"[System] Watching logs at: {os.path.abspath(LOG_FILENAME)}")

    def step(self, batch_trajectories, iteration_id=0):
        try:
            # We assume batch_trajectories is serializable (List of Lists of ints)
            self.queue.put_nowait((batch_trajectories, iteration_id))
        except Exception as e:
            print(f"[AsyncMonitor] Queue push failed: {e}")

    @staticmethod
    def _background_entry_point(queue, config_path):
        # This runs in the separate process
        try:
            worker_logic = _MonitorLogicHandler(config_path)
            worker_logic._log("--- Background Worker Started ---")

            while True:
                data = queue.get()  # Block until data arrives
                worker_logic.process_batch(data[0], data[1])
        except KeyboardInterrupt:
            pass
        except Exception as e:
            # Last resort logging
            with open("monitor_crash.log", "w") as f:
                f.write(str(e))


# Do nothing
class DummyMonitor:
    def step(self, batch_trajectories, iteration_id=0):
        pass


# =========================================
##### Mode Selector #####
# =========================================


def _get_monitor_instance():
    mode = os.getenv("TRAJECTORY_MONITOR_MODE", "NONE").upper()
    print(f"[System] Trajectory Monitor Mode: {mode}")

    if mode == "ASYNC":
        return AsyncMonitor()
    elif mode == "SYNC":
        return SynchronousMonitor()
    else:
        return DummyMonitor()


online_monitor = _get_monitor_instance()
