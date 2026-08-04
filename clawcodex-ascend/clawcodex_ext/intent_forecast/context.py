#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Context collection for Intent Forecast."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clawcodex_ext.intent_forecast.config import IntentForecastConfig
from clawcodex_ext.intent_forecast.focus import compute_workspace_focuses
from clawcodex_ext.intent_forecast.learning import read_recent_feedback
from clawcodex_ext.intent_forecast.session_retrieval import rank_session_rows
from clawcodex_ext.intent_forecast.task_state import build_task_state, classify_intent_stage
from clawcodex_ext.session_intelligence.queue import enqueue_summary_job


MEMORY_FILENAMES = ("CLAUDE.md", "CLAUDE.local.md", "AGENTS.md", "CLUDE.md")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastContext:
    cwd: str
    current_messages: list[dict[str, str]] = field(default_factory=list)
    user_intent: dict[str, Any] = field(default_factory=dict)
    sessions: list[dict[str, Any]] = field(default_factory=list)
    memory_files: list[dict[str, str]] = field(default_factory=list)
    workspace: dict[str, Any] = field(default_factory=dict)
    task_state: dict[str, Any] = field(default_factory=dict)
    intent_stage: str = "explore"
    feedback: list[dict[str, Any]] = field(default_factory=list)
    response_language: str = "English"
    intent_strategy: str = "user"
    fingerprint: str = ""

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "cwd": self.cwd,
            "current_messages": self.current_messages,
            "user_intent": self.user_intent,
            "task_state": self.task_state,
            "intent_stage": self.intent_stage,
            "sessions": self.sessions,
            "memory_files": self.memory_files,
            "workspace": self.workspace,
            "feedback": self.feedback,
            "response_language": self.response_language,
            "intent_strategy": self.intent_strategy,
        }


class IntentForecastContextBuilder:
    def __init__(
        self,
        *,
        conversation: Any | None,
        workspace_root: Path,
        config: IntentForecastConfig | None = None,
        sessions_dir: Path | None = None,
        feedback_base_dir: Path | None = None,
    ) -> None:
        self.conversation = conversation
        self.workspace_root = Path(workspace_root)
        self.config = config or IntentForecastConfig()
        self.sessions_dir = sessions_dir
        self.feedback_base_dir = feedback_base_dir

    def build(self) -> ForecastContext:
        current = self._current_messages()
        user_intent = build_user_intent(current)
        memory = self._memory_files()
        workspace = self._workspace_signals()
        workspace["focuses"] = compute_workspace_focuses(
            changed_files=[str(path) for path in workspace.get("changed_files") or []],
            recent_messages=[msg for msg in current if msg.get("role") == "user"],
        )
        sessions = self._sessions(current_messages=current, workspace=workspace)
        task_state = build_task_state(
            current_messages=current,
            sessions=sessions,
            workspace=workspace,
            user_intent=user_intent,
        )
        intent_stage = classify_intent_stage(
            current_messages=current,
            task_state=task_state,
            workspace=workspace,
            user_intent=user_intent,
        )
        feedback = read_recent_feedback(limit=30, base_dir=self.feedback_base_dir)
        response_language = (
            self.config.response_language
            if self.config.response_language in {"Chinese", "English"}
            else infer_response_language(current, sessions)
        )
        raw = json.dumps(
            {
                "cwd": str(self.workspace_root),
                "current": current,
                "user_intent": user_intent,
                "sessions": sessions,
                "memory": memory,
                "workspace": workspace,
                "task_state": task_state,
                "intent_stage": intent_stage,
                "response_language": response_language,
                "intent_strategy": self.config.intent_strategy,
            },
            sort_keys=True,
            default=str,
        )
        return ForecastContext(
            cwd=str(self.workspace_root),
            current_messages=current,
            user_intent=user_intent,
            sessions=sessions,
            memory_files=memory,
            workspace=workspace,
            task_state=task_state,
            intent_stage=intent_stage,
            feedback=feedback,
            response_language=response_language,
            intent_strategy=self.config.intent_strategy,
            fingerprint=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
        )

    def _current_messages(self) -> list[dict[str, str]]:
        messages = list(getattr(self.conversation, "messages", []) or [])
        out: list[dict[str, str]] = []
        for msg in messages[-self.config.max_transcript_tail_messages :]:
            role = str(getattr(msg, "role", "") or "")
            if role not in {"user", "assistant", "system"}:
                continue
            text = _flatten_content(getattr(msg, "content", "")).strip()
            if text:
                limit = 1600 if role == "user" else 360
                out.append({"role": role, "content": text[:limit]})
        return out

    def _sessions(
        self,
        *,
        current_messages: list[dict[str, str]],
        workspace: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            from clawcodex_ext.services.session_storage import SessionStorage

            metas = SessionStorage.list_sessions(
                self.sessions_dir,
                limit=max(self.config.max_sessions * 3, self.config.max_sessions),
            )
        except Exception:
            return []
        rows: list[dict[str, Any]] = []
        for meta in metas:
            row = {
                "session_id": getattr(meta, "session_id", ""),
                "title": getattr(meta, "title", ""),
                "model": getattr(meta, "model", ""),
                "last_user_input": getattr(meta, "last_user_input", ""),
                "last_updated": getattr(meta, "last_updated", 0),
                "cwd": getattr(meta, "cwd", ""),
                "tags": list(getattr(meta, "tags", []) or []),
            }
            summary = self._load_session_summary(str(row["session_id"]))
            if summary:
                row["summary"] = summary
            else:
                tail = self._load_transcript_tail(str(row["session_id"]))
                if tail:
                    row["transcript_tail"] = tail
                if self.config.summary_lazy_generate:
                    try:
                        enqueue_summary_job(str(row["session_id"]), cwd=self.workspace_root)
                    except Exception:
                        logger.debug("Failed to enqueue lazy session summary", exc_info=True)
            rows.append(row)
        recent_text = "\n".join(
            str(msg.get("content") or "") for msg in current_messages[-8:] if msg.get("role") == "user"
        )
        return rank_session_rows(
            rows,
            cwd=self.workspace_root,
            changed_files=[str(path) for path in workspace.get("changed_files") or []],
            recent_text=recent_text,
            limit=self.config.max_sessions,
            strategy=self.config.intent_strategy,
        )

    def _load_session_summary(self, session_id: str) -> dict[str, Any] | None:
        if not session_id:
            return None
        try:
            from clawcodex_ext.services.session_storage import SESSIONS_DIR

            base = self.sessions_dir or SESSIONS_DIR
            path = Path(base) / session_id / "summary.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and int(data.get("schema_version", 0)) >= 1:
                return {
                    k: data.get(k)
                    for k in (
                        "title",
                        "goals",
                        "open_threads",
                        "next_action_candidates",
                        "files_touched",
                        "commands_seen",
                    )
                }
        except Exception:
            return None
        return None

    def _load_transcript_tail(self, session_id: str) -> list[dict[str, str]]:
        if not session_id:
            return []
        try:
            from clawcodex_ext.services.session_storage import SESSIONS_DIR

            base = self.sessions_dir or SESSIONS_DIR
            path = Path(base) / session_id / "transcript.jsonl"
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return []
        rows: list[dict[str, str]] = []
        for line in lines[-self.config.max_transcript_tail_messages :]:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            role = str(data.get("role") or "")
            if role not in {"user", "assistant"}:
                continue
            text = _flatten_content(data.get("content", "")).strip()
            if text:
                rows.append({"role": role, "content": text[:1000]})
        return rows

    def _memory_files(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for name in MEMORY_FILENAMES:
            path = self.workspace_root / name
            if not path.is_file():
                continue
            try:
                rows.append(
                    {
                        "path": name,
                        "content": path.read_text(encoding="utf-8", errors="replace")[:4000],
                    }
                )
            except OSError:
                continue
        dot = self.workspace_root / ".clawcodex"
        if dot.is_dir():
            for path in sorted(dot.glob("*.md"))[:5]:
                try:
                    rows.append(
                        {
                            "path": str(path.relative_to(self.workspace_root)),
                            "content": path.read_text(encoding="utf-8", errors="replace")[:2000],
                        }
                    )
                except OSError:
                    continue
        return rows

    def _workspace_signals(self) -> dict[str, Any]:
        git_status = _run_git(self.workspace_root, ["status", "--short"])[:4000]
        diff_names = _git_lines(_run_git(self.workspace_root, ["diff", "--name-only"]))
        changed_files = _changed_files_from_status(git_status)
        if not changed_files:
            changed_files = diff_names
        last_command = _last_command_sidecar(self.workspace_root)
        return {
            "git_status": git_status,
            "git_branch": _run_git(self.workspace_root, ["branch", "--show-current"])[:200],
            "git_diff_stat": _run_git(self.workspace_root, ["diff", "--stat"])[:4000],
            "git_diff_names": diff_names[:100],
            "changed_files": changed_files,
            "untracked_files": _untracked_files_from_status(git_status),
            "changed_test_mapping": _changed_test_mapping(changed_files),
            "diff_hunks_summary": _diff_hunks_summary(self.workspace_root, diff_names),
            "recent_commits": _recent_commits_from_log(self.workspace_root, limit=10),
            "last_command": last_command.get("command", ""),
            "last_command_exit": last_command.get("exit_code", None),
            "last_test_failures": _last_test_failures(last_command),
            "project_files": _project_files(self.workspace_root),
            "permission_mode": _permission_mode(self.workspace_root),
        }


def _run_git(cwd: Path, args: list[str]) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        return ""
    try:
        proc = subprocess.run(
            [git_executable, *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return ""
    return (proc.stdout or proc.stderr or "").strip()


def _recent_commits_from_log(cwd: Path, *, limit: int = 10) -> list[dict[str, str]]:
    """Collect the most recent commits as a structured signal for Intent Forecast.

    Returns a list of ``{hash, short_hash, subject, author, timestamp}`` rows.
    Empty list when cwd is not a git repository or git log fails — callers
    should treat that as ``no commit signal``.

    Each field is fetched in its own ``git log`` call so a ``|`` (or any other
    separator) inside the subject or author name cannot corrupt the other
    fields. Field misalignment on a single line is impossible because the
    lists are aligned by index, not parsed from a single line.
    """

    n = max(1, limit)
    hashes = _git_log_field(cwd, n, "%H")
    if not hashes:
        return []
    short_hashes = _git_log_field(cwd, n, "%h")
    subjects = _git_log_field(cwd, n, "%s")
    authors = _git_log_field(cwd, n, "%an")
    timestamps = _git_log_field(cwd, n, "%at")

    rows: list[dict[str, str]] = []
    for idx, hash_ in enumerate(hashes[:n]):
        if not hash_:
            continue
        subject = subjects[idx] if idx < len(subjects) else ""
        if not subject.strip():
            continue
        short_hash = short_hashes[idx] if idx < len(short_hashes) else hash_[:7]
        author = authors[idx] if idx < len(authors) else ""
        timestamp = timestamps[idx] if idx < len(timestamps) else ""
        rows.append(
            {
                "hash": hash_,
                "short_hash": short_hash or hash_[:7],
                "subject": subject[:240],
                "author": author[:120],
                "timestamp": timestamp,
            }
        )
    return rows


def _git_log_field(cwd: Path, limit: int, fmt: str) -> list[str]:
    """Run ``git log`` with a single-field format and return the trimmed values.

    ``_run_git`` returns ``""`` on failure (non-git repo, timeout, missing
    binary), so the caller can short-circuit when the hash list is empty.
    """

    raw = _run_git(
        cwd,
        [
            "log",
            f"-{max(1, limit)}",
            "--no-color",
            "--no-decorate",
            f"--format={fmt}",
        ],
    )
    if not raw:
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _project_files(cwd: Path) -> list[str]:
    names = ("README.md", "pyproject.toml", "package.json", "CLAUDE.md", "AGENTS.md")
    return [name for name in names if (cwd / name).exists()]


def _changed_files_from_status(status: str) -> list[str]:
    files: list[str] = []
    for line in status.splitlines():
        text = line.strip()
        if not text:
            continue
        path = text[2:].strip() if len(text) > 2 else text
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        if path:
            files.append(path.replace("\\", "/"))
    return files[:50]


def _git_lines(raw: str) -> list[str]:
    return [line.strip().replace("\\", "/") for line in raw.splitlines() if line.strip()]


def _untracked_files_from_status(status: str) -> list[str]:
    files: list[str] = []
    for line in status.splitlines():
        if line.startswith("?? "):
            files.append(line[3:].strip().replace("\\", "/"))
    return files[:50]


def _changed_test_mapping(changed_files: list[str]) -> list[str]:
    hints: list[str] = []
    for path in changed_files:
        normalized = path.replace("\\", "/")
        lower = normalized.lower()
        if lower.startswith("tests/") or "/tests/" in f"/{lower}" or "test_" in lower:
            hints.append(normalized)
        elif lower.startswith("clawcodex_ext/intent_forecast/"):
            hints.append("tests/intent_forecast")
        elif lower.startswith("clawcodex_ext/tui/"):
            hints.append("tests/tui")
        elif lower.startswith("extensions/orchestrator/"):
            hints.append("tests/orchestrator")
    out: list[str] = []
    for hint in hints:
        if hint not in out:
            out.append(hint)
    return out[:12]


def _diff_hunks_summary(cwd: Path, diff_names: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in diff_names[:8]:
        stat = _run_git(cwd, ["diff", "--shortstat", "--", name])
        if stat:
            rows.append({"path": name, "summary": stat[:240]})
    return rows


def _last_command_sidecar(cwd: Path) -> dict[str, Any]:
    paths = [
        cwd / ".clawcodex" / "last_command.json",
        Path.home() / ".clawcodex" / "last_command.json",
    ]
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _last_test_failures(last_command: dict[str, Any]) -> list[str]:
    command = str(last_command.get("command") or "")
    try:
        exit_code = int(last_command.get("exit_code"))
    except (TypeError, ValueError):
        return []
    if exit_code == 0 or ("pytest" not in command.lower() and "test" not in command.lower()):
        return []
    output = str(last_command.get("output") or last_command.get("stderr") or last_command.get("stdout") or "")
    failures: list[str] = []
    for line in output.splitlines():
        lower = line.lower()
        if "failed" in lower or "error" in lower or "assertion" in lower:
            failures.append(line.strip()[:240])
        if len(failures) >= 6:
            break
    return failures or [f"{command} failed with exit code {exit_code}"]


def _permission_mode(cwd: Path) -> str:
    try:
        from clawcodex_ext.settings.settings import load_settings  # pylint: disable=no-name-in-module

        settings = load_settings(cwd=cwd)
        structured = getattr(getattr(settings, "permissions", None), "default_mode", None)
        legacy = getattr(settings, "permission_mode", "") or ""
        return str(structured or legacy or "default")
    except Exception:
        return "unknown"


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_flatten_content(item) for item in content)
    if isinstance(content, dict):
        if content.get("type") in (None, "text"):
            return str(content.get("text") or content.get("content") or "")
        if content.get("type") == "tool_use":
            return f"[tool:{content.get('name') or ''}]"
        if content.get("type") == "tool_result":
            return str(content.get("content") or "")
    text = getattr(content, "text", None)
    if text is not None:
        return str(text)
    return str(content)


def infer_response_language(
    current_messages: list[dict[str, str]],
    sessions: list[dict[str, Any]] | None = None,
) -> str:
    """Infer whether forecast suggestions should be Chinese or English."""

    samples: list[str] = []
    for msg in reversed(current_messages):
        if msg.get("role") == "user":
            samples.append(str(msg.get("content") or ""))
        if len(samples) >= 6:
            break
    if not samples:
        for session in sessions or []:
            title = str(session.get("title") or "")
            if title:
                samples.append(title)
            last = str(session.get("last_user_input") or "")
            if last:
                samples.append(last)
            summary = session.get("summary")
            if isinstance(summary, dict):
                samples.extend(_summary_language_samples(summary))
            tail = session.get("transcript_tail")
            if isinstance(tail, list):
                for item in reversed(tail):
                    if isinstance(item, dict) and item.get("role") == "user":
                        samples.append(str(item.get("content") or ""))
                        break
            if len(samples) >= 6:
                break

    text = "\n".join(samples)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    if cjk >= 3 and cjk >= latin * 0.25:
        return "Chinese"
    return "English"


def build_user_intent(current_messages: list[dict[str, str]]) -> dict[str, Any]:
    """Extract user-owned intent signals from the active conversation."""

    user_turns = [
        str(msg.get("content") or "").strip()
        for msg in current_messages
        if msg.get("role") == "user" and str(msg.get("content") or "").strip()
    ]
    initial = user_turns[0] if user_turns else ""
    latest = user_turns[-1] if user_turns else ""
    previous = user_turns[-6:-1] if len(user_turns) > 1 else []
    return {
        "initial_user_input": initial[:1600],
        "latest_user_input": latest[:1600],
        "previous_user_inputs": [item[:800] for item in previous],
        "user_turn_count": len(user_turns),
        "explicit_preferences": _explicit_user_preferences(user_turns),
    }


def _explicit_user_preferences(user_turns: list[str]) -> list[str]:
    markers = (
        "prefer",
        "don't",
        "do not",
        "use ",
        "keep",
        "应该",
        "不要",
        "别",
        "优先",
        "使用",
        "保持",
        "改成",
    )
    prefs: list[str] = []
    for turn in user_turns[-8:]:
        lowered = turn.lower()
        if any(marker in lowered for marker in markers):
            prefs.append(turn[:240])
    return prefs[-5:]


def _summary_language_samples(summary: dict[str, Any]) -> list[str]:
    samples: list[str] = []
    for key in ("title", "goals", "open_threads", "next_action_candidates"):
        value = summary.get(key)
        if isinstance(value, str):
            samples.append(value)
        elif isinstance(value, list):
            samples.extend(str(item) for item in value[:6])
    return samples
