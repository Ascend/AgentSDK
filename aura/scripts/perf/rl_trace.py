#!/usr/bin/env python3
"""
将训练日志解析为 Chrome Trace 格式，以便可视化关键阶段耗时。

用法示例：

    python perf.py --log test.log --config perf.yaml
    python perf.py --log log1.log log2.log --config perf.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from datetime import datetime

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - 运行时报错更友好
    raise SystemExit("缺少 PyYAML 依赖，请执行 pip install pyyaml") from exc

LOGGER = logging.getLogger("perf")


@dataclass
class EventRule:
    name: str
    start_pattern: re.Pattern
    end_pattern: re.Pattern
    time_group: int
    id_group: Optional[int]
    pid: Optional[int]
    tid: Optional[int]


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if "trace" not in config:
        raise ValueError("配置文件缺少 trace 节点")
    return config["trace"]


def compile_rules(trace_cfg: Dict) -> List[EventRule]:
    events_cfg = trace_cfg.get("events", [])
    if not events_cfg:
        raise ValueError("trace.events 不能为空")

    compiled: List[EventRule] = []
    for raw in events_cfg:
        try:
            name = raw["name"]
            start_regex = re.compile(raw["start_pattern"])
            end_regex = re.compile(raw["end_pattern"])
            time_group = int(raw.get("time_group", 1))
        except KeyError as exc:
            raise ValueError(f"事件配置缺少字段: {exc}") from exc
        id_group = raw.get("id_group")
        compiled.append(
            EventRule(
                name=name,
                start_pattern=start_regex,
                end_pattern=end_regex,
                time_group=time_group,
                id_group=int(id_group) if id_group is not None else None,
                pid=raw.get("pid"),
                tid=raw.get("tid"),
            )
        )
    return compiled


def _convert_timestamp(ts_str: str, unit: str, dt_format: Optional[str]) -> float:
    if unit == "datetime":
        if not dt_format:
            raise ValueError("timestamp_unit 为 datetime 时必须提供 timestamp_format")
        dt = datetime.strptime(ts_str, dt_format)
        return dt.timestamp() * 1_000_000.0
    value = float(ts_str)
    if unit == "unix_ms":
        return value * 1000.0
    if unit == "unix_s":
        return value * 1_000_000.0
    if unit == "relative_ms":
        return value * 1000.0
    raise ValueError(f"不支持的时间单位: {unit}")


def build_trace_from_log(
    log_path: Path,
    rules: Iterable[EventRule],
    pid_default: int,
    tid_default: int,
    timestamp_unit: str,
    timestamp_format: Optional[str],
) -> List[Dict]:
    trace_events: List[Dict] = []
    active_with_id: Dict[Tuple[str, str], Dict] = {}
    active_no_id: Dict[str, List[Dict]] = {}
    auto_id_counters: Dict[str, int] = {}

    with log_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            text = line.rstrip("\n")
            matched = False
            for rule in rules:
                start_match = rule.start_pattern.search(text)
                if start_match:
                    matched = True
                    if rule.id_group is not None:
                        event_id = start_match.group(rule.id_group)
                        key = (rule.name, event_id)
                        if key in active_with_id:
                            LOGGER.warning(
                                "Line %d: %s 已有未结束事件（id=%s），将被覆盖",
                                line_no,
                                rule.name,
                                event_id,
                            )
                        target_store = active_with_id
                    else:
                        counter = auto_id_counters.get(rule.name, 0) + 1
                        auto_id_counters[rule.name] = counter
                        event_id = str(counter)
                        target_store = active_no_id.setdefault(rule.name, [])

                    ts = _convert_timestamp(
                        start_match.group(rule.time_group),
                        timestamp_unit,
                        timestamp_format,
                    )
                    record = {"ts": ts, "line": line_no, "id": event_id}
                    if rule.id_group is not None:
                        target_store[(rule.name, event_id)] = record
                    else:
                        target_store.append(record)
                    continue

                end_match = rule.end_pattern.search(text)
                if end_match:
                    matched = True
                    if rule.id_group is not None:
                        event_id = end_match.group(rule.id_group)
                        key = (rule.name, event_id)
                        start_info = active_with_id.pop(key, None)
                    else:
                        queue = active_no_id.get(rule.name)
                        start_info = queue.pop(0) if queue else None
                        if queue is not None and not queue:
                            active_no_id.pop(rule.name, None)
                        event_id = start_info["id"] if start_info else None
                    if not start_info:
                        LOGGER.warning(
                            "Line %d: 找到 %s 结束但没有匹配的开始（id=%s）",
                            line_no,
                            rule.name,
                            event_id,
                        )
                        continue
                    end_ts = _convert_timestamp(
                        end_match.group(rule.time_group),
                        timestamp_unit,
                        timestamp_format,
                    )
                    duration = max(0.0, end_ts - start_info["ts"])
                    trace_events.append(
                        {
                            "name": rule.name,
                            "cat": rule.name,
                            "ph": "X",
                            "ts": start_info["ts"],
                            "dur": duration,
                            "pid": rule.pid or pid_default,
                            "tid": rule.tid or tid_default,
                            "args": {
                                "id": event_id,
                                "start_line": start_info["line"],
                                "end_line": line_no,
                            },
                        }
                    )
                    continue
            if not matched:
                LOGGER.debug("Line %d: 无匹配事件 -> %s", line_no, text)

    for (name, event_id), info in active_with_id.items():
        LOGGER.warning("事件 %s (id=%s) 在行 %d 开始但未找到结束", name, event_id, info["line"])
    for name, queue in active_no_id.items():
        for info in queue:
            LOGGER.warning(
                "事件 %s (auto_id=%s) 在行 %d 开始但未找到结束",
                name,
                info["id"],
                info["line"],
            )
    return trace_events


def save_trace(output: Path, trace_events: List[Dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"traceEvents": trace_events}
    with output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    LOGGER.info("已写入 trace 文件：%s", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练日志转 trace 工具")
    parser.add_argument(
        "--log",
        type=Path,
        nargs="+",
        required=True,
        help="训练日志路径（可指定多个文件）",
    )
    parser.add_argument("--config", type=Path, default=Path("perf.yaml"), help="配置文件路径")
    parser.add_argument("--output", type=Path, help="覆盖配置里的 trace.output，若提供则优先生效")
    parser.add_argument("--verbose", action="store_true", help="打印更多调试信息（用于排查匹配问题）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    trace_cfg = load_config(args.config)
    rules = compile_rules(trace_cfg)
    pid_default = int(trace_cfg.get("pid", 0))
    tid_default = int(trace_cfg.get("tid", 0))
    timestamp_unit = trace_cfg.get("timestamp_unit", "unix_ms")
    timestamp_format = trace_cfg.get("timestamp_format")

    all_events: List[Dict] = []
    for log_path in args.log:
        if not log_path.exists():
            LOGGER.warning("日志文件不存在，跳过: %s", log_path)
            continue
        LOGGER.info("解析日志 %s", log_path)
        events = build_trace_from_log(
            log_path=log_path,
            rules=rules,
            pid_default=pid_default,
            tid_default=tid_default,
            timestamp_unit=timestamp_unit,
            timestamp_format=timestamp_format,
        )
        all_events.extend(events)
        LOGGER.info("从 %s 提取了 %d 个事件", log_path, len(events))

    LOGGER.info("总共提取了 %d 个事件", len(all_events))
    output_path = args.output or Path(trace_cfg.get("output", "trace.json"))
    save_trace(output_path, all_events)


if __name__ == "__main__":
    main()
