"""``/btw`` 使用统计 (F-122-I).

记录用户每次 ``/btw`` 调用的次数与最近一次问题文本，存储于
``$CLAWCODEX_DATA_DIR/btw_stats.json``（默认 ``~/.clawcodex/btw_stats.json``），
与 sidechain transcript 共用同一根目录约定。

设计动机
--------

规划文档 §1.3 / §2.3 Phase 7 指出，F-122-I 是 P3 可选特性，用于「记录
``/btw`` 使用次数（类似 TS 的 ``btwUseCount`` config）」。该计数可在
后续 /settings 面板、usage report 或 telemetry 仪表盘中展示，让用户
直观看到自己侧边问答的使用频率。

存储格式
--------

单个 JSON 文件::

    {
      "use_count":        5,
      "first_used":       "2026-07-02T12:34:56",
      "first_used_epoch": 1751475296.123,
      "last_used":        "2026-07-02T13:00:00",
      "last_used_epoch":  1751478000.456,
      "last_question":    "what is X?"
    }

字段语义：

* ``use_count`` — 累计调用次数（含失败调用；只要 ``/btw`` 触发即 +1）
* ``first_used`` / ``first_used_epoch`` — 首次调用的时间戳（秒精度 + epoch）
* ``last_used`` / ``last_used_epoch`` — 最近一次调用的时间戳
* ``last_question`` — 最近一次问题文本的前 80 字符（截断以避免巨型字段）

原子写
------

read-modify-write 通过 ``<file>.tmp`` + ``os.replace`` 实现原子替换。OS 层面
保证看到的是旧文件或新文件，不会出现半写状态。与 sidechain transcript
的 O_APPEND 不同，本模块每次写入都是完整文件，因此使用替换策略而非追加。

失败语义
--------

fire-and-forget —— 任何 IO / JSON 错误仅记录 WARNING，绝不向上抛出。
``/btw`` 用户流程必须永远观察不到统计模块的副作用。

环境变量
--------

* ``CLAWCODEX_DISABLE_BTW_STATS=1`` (或 ``true`` / ``yes`` / ``on``) —
  禁用统计；所有写入变为 no-op，``get_btw_stats()`` 仍返回零值快照。
* ``CLAWCODEX_DATA_DIR=/path/to/root`` — 覆盖统计文件根目录。
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_FILENAME = "btw_stats.json"
_TMP_SUFFIX = ".tmp"

_DISABLE_ENV_VAR = "CLAWCODEX_DISABLE_BTW_STATS"
_DATA_DIR_ENV_VAR = "CLAWCODEX_DATA_DIR"

_LAST_QUESTION_MAX_LEN = 80


_DEFAULT_STATS: dict[str, Any] = {
    "use_count": 0,
    "first_used": None,
    "first_used_epoch": None,
    "last_used": None,
    "last_used_epoch": None,
    "last_question": None,
}


def _is_env_truthy(value: str | None) -> bool:
    """Standard env-truthy test (matches sidechain/paths convention)."""
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def is_btw_stats_enabled() -> bool:
    """Whether ``/btw`` usage statistics are currently being recorded.

    Default is *enabled*. Set ``CLAWCODEX_DISABLE_BTW_STATS=1`` (or any
    truthy value) to opt out.
    """
    return not _is_env_truthy(os.environ.get(_DISABLE_ENV_VAR))


def get_btw_stats_path() -> Path:
    """Return the JSON file path used to persist ``/btw`` stats.

    Resolution order:
      1. ``$CLAWCODEX_DATA_DIR/btw_stats.json`` if the env var is set.
      2. ``~/.clawcodex/btw_stats.json`` as fallback.

    Note: the parent directory is **not** created here — that happens on
    the first increment, so sessions that never call ``/btw`` leave no
    trace on disk.
    """
    override = os.environ.get(_DATA_DIR_ENV_VAR)
    if override:
        root = Path(override).expanduser()
    else:
        root = Path.home() / ".clawcodex"
    return root / _FILENAME


def _truncate_question(question: str | None) -> str | None:
    """Trim *question* to a bounded length so the stats file stays small."""
    if not question:
        return None
    text = question.strip()
    if not text:
        return None
    if len(text) <= _LAST_QUESTION_MAX_LEN:
        return text
    return text[:_LAST_QUESTION_MAX_LEN] + "…"


def _load_existing_stats(path: Path) -> dict[str, Any]:
    """Load the existing stats file, returning a default-zero snapshot on
    any read/parse failure (corrupted file, permission error, etc.).

    Failure here is *not* fatal — ``increment_btw_use_count`` should still
    succeed by starting from zero.
    """
    if not path.exists():
        return dict(_DEFAULT_STATS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "F-122-I: failed to read existing btw stats at %s (%s); starting from zero",
            path,
            exc,
        )
        return dict(_DEFAULT_STATS)
    if not isinstance(data, dict):
        logger.warning(
            "F-122-I: existing btw stats at %s is not a JSON object; starting from zero",
            path,
        )
        return dict(_DEFAULT_STATS)
    # Coerce each known field to its expected type so an externally-edited
    # file can't crash the increment path with a TypeError on next write.
    coerced = dict(_DEFAULT_STATS)
    raw_count = data.get("use_count", 0)
    try:
        coerced["use_count"] = int(raw_count)
    except (TypeError, ValueError):
        coerced["use_count"] = 0
    for str_field in ("first_used", "last_used", "last_question"):
        val = data.get(str_field)
        coerced[str_field] = val if isinstance(val, str) or val is None else str(val)
    for num_field in ("first_used_epoch", "last_used_epoch"):
        val = data.get(num_field)
        if val is None:
            coerced[num_field] = None
        else:
            try:
                coerced[num_field] = float(val)
            except (TypeError, ValueError):
                coerced[num_field] = None
    return coerced


def increment_btw_use_count(*, question: str | None = None) -> dict[str, Any] | None:
    """Increment the ``/btw`` use count and persist the updated stats.

    Fire-and-forget: any IO / parse failure is logged at WARNING and
    swallowed. Returns the updated stats dict on success, ``None`` when
    recording is skipped (disabled) or unrecoverably failed.

    F-122-I: this is the single source of truth for the ``/btw`` use
    counter. The increment happens at the **command layer** (every UI path
    — REPL, TUI, headless — flows through ``btw_command_run``) so the
    counter is incremented exactly once per invocation regardless of
    whether the underlying side question succeeds or fails.
    """
    if not is_btw_stats_enabled():
        return None

    path = get_btw_stats_path()
    now_epoch = time.time()
    now_iso = datetime.now().isoformat(timespec="seconds")
    truncated_question = _truncate_question(question)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = _load_existing_stats(path)
        new_count = int(stats.get("use_count", 0)) + 1
        is_first_use = new_count == 1 or not stats.get("first_used")

        stats["use_count"] = new_count
        stats["last_used"] = now_iso
        stats["last_used_epoch"] = now_epoch
        stats["last_question"] = truncated_question
        if is_first_use:
            stats["first_used"] = now_iso
            stats["first_used_epoch"] = now_epoch

        tmp_path = path.with_name(path.name + _TMP_SUFFIX)
        # ``O_WRONLY | O_CREAT | O_TRUNC`` so a stale .tmp from a crashed
        # previous run is overwritten cleanly. ``0o600`` keeps the stats
        # file readable by the current user only — it can carry recent
        # question text which is mildly sensitive.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(str(tmp_path), flags, 0o600)
        try:
            payload = json.dumps(stats, ensure_ascii=False, indent=2)
            os.write(fd, payload.encode("utf-8"))
            os.write(fd, b"\n")
        finally:
            os.close(fd)
        os.replace(tmp_path, path)
        return stats
    except Exception:
        logger.warning(
            "F-122-I: failed to record /btw usage stat (question=%r)",
            (question or "")[:60],
            exc_info=True,
        )
        return None


def get_btw_stats() -> dict[str, Any]:
    """Read the current ``/btw`` stats snapshot.

    Returns the persisted stats dict on success, or a zero-valued copy
    of :data:`_DEFAULT_STATS` if the file does not exist / is unreadable
    / recording is disabled. Never raises — safe to call from any code
    path including tests and the main user flow.
    """
    if not is_btw_stats_enabled():
        return dict(_DEFAULT_STATS)
    path = get_btw_stats_path()
    if not path.exists():
        return dict(_DEFAULT_STATS)
    return _load_existing_stats(path)


def reset_btw_stats() -> None:
    """Remove the persisted stats file (if any). Intended for tests only."""
    path = get_btw_stats_path()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # missing_ok=True already swallows FileNotFoundError; any other
        # OSError (perm denied, etc.) is a test-setup problem we don't
        # want to mask as an exception in the calling test.
        pass


__all__ = [
    "increment_btw_use_count",
    "get_btw_stats",
    "get_btw_stats_path",
    "is_btw_stats_enabled",
    "reset_btw_stats",
]
