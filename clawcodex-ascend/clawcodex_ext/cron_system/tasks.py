"""File-backed Cron task storage."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, MutableMapping

from .jitter import jittered_next_cron_run_ms, one_shot_jittered_next_cron_run_ms
from .lock import acquire_cron_storage_lock
from .models import (
    DEFAULT_RECURRING_MAX_AGE_MS,
    SCHEDULED_TASKS_RELATIVE_PATH,
    CronJitterConfig,
    CronTask,
    validate_jitter_config,
)
from .parser import parse_cron_expression


# F-22-H: mtime-based incremental read cache
# Entry shape: (cached_at_monotonic, cached_file_mtime, tasks).
# Both timestamps must be tracked separately: the monotonic value drives
# the TTL gate (clock-independent of file-system mtime changes); the file
# mtime drives the change-detection gate (new mtime → re-read regardless
# of TTL).
_MtimeCache: dict[Path, tuple[float, float, list[CronTask]]] = {}
_MTIME_CACHE_TTL: float = 1.0  # 1 second cooldown

# Serialise all session_store mutations (add / mark-fired / prune) so the
# clear()+rewrite pattern cannot lose entries to a concurrent writer.
_session_store_lock = threading.Lock()


def tasks_file_path(workspace_root: Path) -> Path:
    return workspace_root / SCHEDULED_TASKS_RELATIVE_PATH


def now_ms() -> int:
    return int(time.time() * 1000)


def read_cron_tasks(workspace_root: Path) -> list[CronTask]:
    path = tasks_file_path(workspace_root)
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if not exists:
        # Backward compat: check legacy .claude/scheduled_tasks.json
        legacy = workspace_root / ".claude" / "scheduled_tasks.json"
        try:
            legacy_exists = legacy.exists()
        except OSError:
            legacy_exists = False
        if not legacy_exists:
            return []
        path = legacy
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    entries = raw.get("tasks", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return []

    tasks: list[CronTask] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        task = CronTask.from_dict(entry)
        if task is not None and parse_cron_expression(task.cron) is not None:
            tasks.append(task)
    tasks.sort(key=lambda task: task.id)
    return tasks


def read_cron_tasks_cached(workspace_root: Path) -> list[CronTask]:
    """F-22-H: mtime-based incremental read of durable cron tasks.

    Skips full file re-read when either:

    1. The cached entry is fresh (within ``_MTIME_CACHE_TTL`` seconds of
       monotonic clock at cache time), **or**
    2. The cached entry's recorded file mtime still matches the on-disk
       file mtime (cross-tick dedup when nothing has changed).

    Falls back to :func:`read_cron_tasks` on cache miss or IO error.
    """
    path = tasks_file_path(workspace_root)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return read_cron_tasks(workspace_root)  # fallback
    cached = _MtimeCache.get(path)
    if cached is not None:
        cached_at, cached_mtime, cached_tasks = cached
        if (time.monotonic() - cached_at) < _MTIME_CACHE_TTL:
            return cached_tasks
        if cached_mtime == mtime:
            # Refresh the TTL window so subsequent in-window reads skip
            # the stat call entirely.
            _MtimeCache[path] = (time.monotonic(), cached_mtime, cached_tasks)
            return cached_tasks
    tasks = read_cron_tasks(workspace_root)
    _MtimeCache[path] = (time.monotonic(), mtime, tasks)
    return tasks


def write_cron_tasks(workspace_root: Path, tasks: Iterable[CronTask]) -> None:
    path = tasks_file_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "tasks": [task.to_dict() for task in sorted(tasks, key=lambda item: item.id)],
    }
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_session_cron_tasks(
    session_store: MutableMapping[str, CronTask | dict] | None,
) -> list[CronTask]:
    if session_store is None:
        return []

    tasks: list[CronTask] = []
    stale_ids: list[str] = []
    for task_id, value in session_store.items():
        if isinstance(value, CronTask):
            task = value
        elif isinstance(value, dict):
            task = CronTask.from_dict(value)
        else:
            task = None
        if task is None or parse_cron_expression(task.cron) is None:
            stale_ids.append(task_id)
            continue
        tasks.append(task)

    for task_id in stale_ids:
        session_store.pop(task_id, None)

    tasks.sort(key=lambda task: task.id)
    return tasks


def write_session_cron_tasks(session_store: MutableMapping[str, CronTask | dict], tasks: Iterable[CronTask]) -> None:
    session_store.clear()
    for task in sorted(tasks, key=lambda item: item.id):
        session_store[task.id] = task


def read_all_cron_tasks(
    workspace_root: Path,
    session_store: MutableMapping[str, CronTask | dict] | None = None,
) -> list[CronTask]:
    tasks = [*read_session_cron_tasks(session_store), *read_cron_tasks(workspace_root)]
    tasks.sort(key=lambda task: task.id)
    return tasks


def has_cron_tasks_sync(workspace_root: Path) -> bool:
    return bool(read_cron_tasks(workspace_root))


def add_cron_task(
    workspace_root: Path,
    *,
    cron: str,
    prompt: str,
    recurring: bool = True,
    durable: bool = True,
    jitter: CronJitterConfig | None = None,
    created_at: int | None = None,
    session_store: MutableMapping[str, CronTask | dict] | None = None,
    agent_id: str | None = None,  # F-22-F: agent ownership
) -> CronTask:
    fields = parse_cron_expression(cron)
    if fields is None:
        raise ValueError("cron must be a valid five-field expression")

    timestamp = created_at if created_at is not None else now_ms()
    task_id = uuid.uuid4().hex[:8]
    task = CronTask(
        id=task_id,
        cron=cron,
        prompt=prompt,
        recurring=recurring,
        durable=durable,
        agent_id=agent_id,
        created_at=timestamp,
        updated_at=timestamp,
        expires_at=timestamp + DEFAULT_RECURRING_MAX_AGE_MS if recurring else None,
        jitter=jitter or CronJitterConfig(),
    )
    if recurring:
        next_fire_at = jittered_next_cron_run_ms(
            task.id,
            fields,
            _datetime_from_ms(timestamp),
            task.jitter,
        )
    else:
        next_fire_at = one_shot_jittered_next_cron_run_ms(
            task.id,
            fields,
            _datetime_from_ms(timestamp),
            task.jitter,
        )
    task = replace(
        task,
        next_fire_at=next_fire_at,
    )
    if not durable:
        if session_store is None:
            raise ValueError("session_store is required for session cron tasks")
        with _session_store_lock:
            tasks = read_session_cron_tasks(session_store)
            tasks.append(task)
            write_session_cron_tasks(session_store, tasks)
        return task

    with acquire_cron_storage_lock(workspace_root, f"add-{task_id}"):
        tasks = read_cron_tasks(workspace_root)
        tasks.append(task)
        write_cron_tasks(workspace_root, tasks)
    return task


def write_permanent_task_if_missing(
    workspace_root: Path,
    *,
    cron: str,
    prompt: str,
    recurring: bool = True,
    jitter: CronJitterConfig | None = None,
    created_at: int | None = None,
    task_id: str | None = None,
) -> tuple[CronTask, bool]:
    """F-22-G4 installer entry point.

    Idempotent: writes the task only if no existing task matches the
    ``cron`` expression AND ``prompt`` (case-insensitive trimmed match).
    Returns ``(task, created)`` so the installer can log "skipped" on
    re-install.

    Raises ``PermissionError`` if any pre-existing task has
    ``permanent=True`` but a different ``cron`` or ``prompt`` — this guards
    against the installer accidentally overwriting a system task.
    """
    fields = parse_cron_expression(cron)
    if fields is None:
        raise ValueError("cron must be a valid five-field expression")

    normalized_cron = cron.strip().casefold()
    normalized_prompt = prompt.strip().casefold()
    with acquire_cron_storage_lock(workspace_root, f"permanent-{task_id or 'install'}"):
        existing = read_cron_tasks(workspace_root)
        for task in existing:
            if task.permanent:
                same_schedule = task.cron.strip().casefold() == normalized_cron
                same_prompt = task.prompt.strip().casefold() == normalized_prompt
                if same_schedule and same_prompt:
                    return task, False
                if same_schedule or same_prompt:
                    raise PermissionError(
                        f"refusing to overwrite permanent task {task.id!r} "
                        f"(cron={task.cron!r}, prompt={task.prompt[:40]!r})"
                    )

        # If a non-permanent task with the same shape exists, replace it
        # with the permanent version. Installers should converge to a
        # permanent record on first run.
        filtered = [
            t
            for t in existing
            if not (t.cron.strip().casefold() == normalized_cron and t.prompt.strip().casefold() == normalized_prompt)
        ]
        timestamp = created_at if created_at is not None else now_ms()
        new_id = task_id or uuid.uuid4().hex[:8]
        permanent_task_kwargs: dict[str, Any] = dict(
            id=new_id,
            cron=normalized_cron,
            prompt=normalized_prompt,
            recurring=recurring,
            durable=True,
            created_at=timestamp,
            updated_at=timestamp,
            # Permanent tasks intentionally never auto-expire.
            expires_at=None,
            permanent=True,
        )
        if jitter is not None:
            permanent_task_kwargs["jitter"] = validate_jitter_config(jitter)
        permanent_task = CronTask(**permanent_task_kwargs)
        # Compute next_fire_at with the (recurring) jitter path.
        permanent_task = replace(
            permanent_task,
            next_fire_at=jittered_next_cron_run_ms(
                permanent_task.id,
                fields,
                _datetime_from_ms(timestamp),
                permanent_task.jitter,
            ),
        )
        filtered.append(permanent_task)
        write_cron_tasks(workspace_root, filtered)
        return permanent_task, True


def remove_cron_tasks(
    workspace_root: Path,
    task_id: str,
    session_store: MutableMapping[str, CronTask | dict] | None = None,
) -> bool:
    if session_store is not None and task_id in session_store:
        session_store.pop(task_id, None)
        return True

    with acquire_cron_storage_lock(workspace_root, f"remove-{task_id}"):
        tasks = read_cron_tasks(workspace_root)
        remaining = [task for task in tasks if task.id != task_id]
        if len(remaining) == len(tasks):
            return False
        write_cron_tasks(workspace_root, remaining)
    return True


def mark_cron_tasks_fired(
    workspace_root: Path,
    fired: Iterable[CronTask],
    fired_at: int | None = None,
    *,
    session_store: MutableMapping[str, CronTask | dict] | None = None,
) -> list[CronTask]:
    """Mark tasks as fired and reschedule (recurring) or remove (one-shot).

    Phase A-1.3: handle both file-backed (durable) and session-backed
    (durable=False) tasks. Session-store mutations are performed *before*
    the storage lock is acquired to avoid blocking it on in-memory work
    and to keep the storage lock's 10-second deadline free for disk I/O.
    """
    timestamp = fired_at if fired_at is not None else now_ms()
    fired_by_id = {task.id: task for task in fired}

    if session_store is not None:
        with _session_store_lock:
            next_session: list[CronTask] = []
            for key, value in list(session_store.items()):
                if not isinstance(key, str):
                    if isinstance(value, CronTask):
                        next_session.append(value)
                    continue
                task_obj = value if isinstance(value, CronTask) else CronTask.from_dict(value)
                if task_obj is None:
                    continue
                if task_obj.id not in fired_by_id:
                    next_session.append(task_obj)
                    continue
                if not task_obj.recurring:
                    # one-shot session task: drop it
                    continue
                fields = parse_cron_expression(task_obj.cron)
                new_next = (
                    jittered_next_cron_run_ms(
                        task_obj.id,
                        fields,
                        _datetime_from_ms(timestamp),
                        task_obj.jitter,
                    )
                    if fields is not None
                    else None
                )
                next_session.append(
                    replace(
                        task_obj,
                        last_fired_at=timestamp,
                        next_fire_at=new_next,
                        updated_at=timestamp,
                    )
                )
            write_session_cron_tasks(session_store, next_session)

    with acquire_cron_storage_lock(workspace_root, f"mark-fired-{uuid.uuid4().hex}"):
        current = read_cron_tasks(workspace_root)
        updated: list[CronTask] = []
        result: list[CronTask] = []
        for task in current:
            if task.id not in fired_by_id:
                updated.append(task)
                continue
            if not task.recurring:
                result.append(task)
                continue
            fields = parse_cron_expression(task.cron)
            if fields is None:
                next_fire_at = None
            else:
                next_fire_at = jittered_next_cron_run_ms(
                    task.id,
                    fields,
                    _datetime_from_ms(timestamp),
                    task.jitter,
                )
            new_task = replace(
                task,
                last_fired_at=timestamp,
                next_fire_at=next_fire_at,
                updated_at=timestamp,
            )
            updated.append(new_task)
            result.append(new_task)
        write_cron_tasks(workspace_root, updated)
    return result


def find_due_tasks(
    workspace_root: Path,
    at_ms: int | None = None,
    *,
    session_store: MutableMapping[str, CronTask | dict] | None = None,
) -> list[CronTask]:
    """Return tasks whose ``next_fire_at`` is due at ``at_ms``.

    Phase A-1.1: when ``session_store`` is provided, durable=False tasks
    stored in memory are returned alongside file-backed ones.
    """
    timestamp = at_ms if at_ms is not None else now_ms()
    return [
        task
        for task in read_all_cron_tasks(workspace_root, session_store)
        if task.next_fire_at is not None and task.next_fire_at <= timestamp
    ]


def find_missed_tasks(
    workspace_root: Path,
    at_ms: int | None = None,
    *,
    session_store: MutableMapping[str, CronTask | dict] | None = None,
) -> list[CronTask]:
    """Return non-recurring tasks whose ``next_fire_at`` is in the past.

    Phase A-1.2: also consults ``session_store`` so durable=False
    one-shots are reported when the scheduler was inactive.
    """
    timestamp = at_ms if at_ms is not None else now_ms()
    return [
        task
        for task in read_all_cron_tasks(workspace_root, session_store)
        if not task.recurring and task.next_fire_at is not None and task.next_fire_at < timestamp
    ]


def remove_missed_tasks(
    workspace_root: Path,
    missed: Iterable[CronTask],
    *,
    session_store: MutableMapping[str, CronTask | dict] | None = None,
) -> None:
    """Remove one-shot tasks that were missed while the scheduler was inactive.

    Phase C-3: ``session_store`` tasks are removed in memory (no lock) before
    the disk-side removal runs inside the storage lock.
    """
    missed_ids = {task.id for task in missed}
    if not missed_ids:
        return
    if session_store is not None:
        with _session_store_lock:
            for tid in missed_ids:
                session_store.pop(tid, None)
    with acquire_cron_storage_lock(workspace_root, f"remove-missed-{uuid.uuid4().hex}"):
        tasks = read_cron_tasks(workspace_root)
        remaining = [task for task in tasks if task.id not in missed_ids]
        if len(remaining) != len(tasks):
            write_cron_tasks(workspace_root, remaining)


def prune_expired_recurring_tasks(
    workspace_root: Path,
    at_ms: int | None = None,
    *,
    max_age_ms: int | None = None,
    session_store: MutableMapping[str, CronTask | dict] | None = None,
) -> list[CronTask]:
    """Remove recurring tasks past their expiry.

    ``max_age_ms`` (F-22-G2) lets the scheduler pass a live config value
    so an operator tightening ``recurringMaxAgeMs`` mid-session prunes
    stale tasks immediately. When None, the per-task ``expires_at`` is
    used (the value baked at creation time). ``max_age_ms == 0`` disables
    age-based pruning entirely (matches CCB recurringMaxAgeMs=0).

    Phase A-1.4: ``session_store`` is pruned in memory *before* the
    storage lock is taken, mirroring the mark-fired ordering.
    """
    timestamp = at_ms if at_ms is not None else now_ms()

    def _is_kept(task: CronTask) -> bool:
        if task.permanent or not task.recurring:
            return True
        if max_age_ms is not None:
            if max_age_ms == 0:
                return True
            return task.created_at + max_age_ms > timestamp
        if task.expires_at is None:
            return True
        return task.expires_at > timestamp

    if session_store is not None:
        with _session_store_lock:
            kept_session: list[CronTask] = []
            removed_session: list[CronTask] = []
            for key, value in list(session_store.items()):
                if not isinstance(key, str):
                    if isinstance(value, CronTask):
                        kept_session.append(value)
                    continue
                task_obj = value if isinstance(value, CronTask) else CronTask.from_dict(value)
                if task_obj is None:
                    continue
                if _is_kept(task_obj):
                    kept_session.append(task_obj)
                else:
                    removed_session.append(task_obj)
            write_session_cron_tasks(session_store, kept_session)
    else:
        removed_session = []

    with acquire_cron_storage_lock(workspace_root, f"prune-{uuid.uuid4().hex}"):
        tasks = read_cron_tasks(workspace_root)
        # F-22-G4: permanent tasks are exempt from auto-expiry. The
        # assistant-mode installer writes them directly; they must survive
        # restart loops (write_if_missing skips existing files, so a
        # deleted permanent task cannot be recreated).
        kept = [task for task in tasks if _is_kept(task)]
        removed = [task for task in tasks if task not in kept]
        if removed:
            write_cron_tasks(workspace_root, kept)
    return removed_session + removed


def cleanup_orphaned_tasks(
    workspace_root: Path,
    active_agents: set[str],
    *,
    session_store: MutableMapping[str, CronTask | dict] | None = None,
) -> list[CronTask]:
    """F-22-F-5: Find and mark tasks whose owning agent is no longer active.

    Tasks with ``agent_id`` not in ``active_agents`` are returned as
    orphaned. The caller decides whether to delete them (default behaviour:
    list only, no automatic deletion).
    """
    orphaned: list[CronTask] = []
    all_tasks = read_all_cron_tasks(workspace_root, session_store)
    for task in all_tasks:
        if task.agent_id is not None and task.agent_id not in active_agents:
            orphaned.append(task)
    return orphaned


def _datetime_from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000)
