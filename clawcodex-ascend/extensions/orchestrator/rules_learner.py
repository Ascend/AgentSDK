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

"""Rule extraction, storage, and retrieval from PR review feedback.

Pipeline:
  - RuleStore: read/write workflow.rules.yaml with atomic write
  - RuleEngine.extract(): parse agent reply for ## Extracted Rules section
  - BatchedLLMJudge: batched LLM-based dedup / merge / conflict detection
    (subprocess `clawcodex-dev -p`), replacing the earlier TF-IDF/embedding
    approach
  - RuleEngine.score(): 5-dimension quality scoring
  - RuleEngine.prune(): auto-prune when over max_rules limit
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from ._file_utils import read_json, read_text_utf8, write_text_utf8

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RuleJudge protocol — batched LLM-based dedup / merge / conflict detection
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    """LLM 对单个 candidate 规则的去重/合并/冲突判定结果。"""

    action: Literal["duplicate", "merge", "conflict", "new"]
    target_idx: int | None = None
    """当 action 为 duplicate/merge/conflict 时，指向 existing 中对应的索引。"""


# ---------------------------------------------------------------------------
# ExtractTracker — 追踪已提取规则的 commit，保证幂等性
# ---------------------------------------------------------------------------


class ExtractTracker:
    """管理 ``.clawcodex_extracted.json``，记录已提取的 commit SHA。

    与 ``workflow.rules.yaml`` 同级存放，确保每次 ``extract``
    命令不会重复提取同一个 commit。
    """

    FILENAME = ".clawcodex_extracted.json"

    def __init__(self, rules_path: str) -> None:
        self._path = Path(rules_path).parent / self.FILENAME

    def load(self) -> set[str]:
        """读取已处理的 commit SHA 集合。"""
        if not self._path.exists():
            return set()
        try:
            data = read_json(self._path)
            return set(data.get("processed_commits", []))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to load extract tracker %s: %s", self._path, exc)
            return set()

    def save(self, processed: set[str]) -> None:
        """写入已处理的 commit SHA 集合（原子写入）。"""
        data = {
            "version": 1,
            "processed_commits": sorted(processed),
            "last_extracted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        tmp = self._path.with_suffix(".json.tmp")
        try:
            write_text_utf8(tmp, json.dumps(data, indent=2, ensure_ascii=False))
            tmp.replace(self._path)
        except OSError as exc:
            logger.error("Failed to save extract tracker %s: %s", self._path, exc)
            raise


# ---------------------------------------------------------------------------
# BatchedLLMJudge — 生产实现
# ---------------------------------------------------------------------------


class BatchedLLMJudge:
    """通过子进程调用 `clawcodex-dev -p` 做批量判定。

    复用与 `clawcodex-dev -p --provider --model` 完全相同的
    provider/model 解析链路，确保 LLM judge 与编排器主线 agent
    配置一致。
    """

    _JUDGE_SYSTEM = (
        "You are a coding-convention analysis assistant. "
        "Given a set of existing rules and one or more new candidate rules, "
        "determine for each candidate whether it duplicates, should be merged "
        "with, semantically conflicts with, or is entirely new relative to the "
        "existing rules.  Reply ONLY with the exact format shown."
    )

    def __init__(self, provider_name: str | None = None, model: str | None = None) -> None:
        self._provider_name = provider_name
        self._model = model

    async def judge(self, candidates: list[dict], existing: list[dict]) -> list[JudgeResult]:
        if not candidates or not existing:
            return [JudgeResult(action="new") for _ in candidates]

        prompt = self._build_prompt(candidates, existing)
        # 将 system prompt 拼入 prompt 开头（-p 模式不支持 --system-prompt）
        full_prompt = f"{self._JUDGE_SYSTEM}\n\n{prompt}"
        reply = await self._run_clawcodex(full_prompt)
        return self._parse_reply(reply, len(candidates))

    # ------------------------------------------------------------------
    # Subprocess invocation
    # ------------------------------------------------------------------

    _NOISE_MARKER = "\nResume this session with:"

    @staticmethod
    def _resolve_clawcodex_dev() -> str:
        """定位 clawcodex-dev 可执行文件。"""
        import shutil
        import sys

        exe = shutil.which("clawcodex-dev")
        if exe:
            return exe
        # venv 回退：与当前 Python 同目录
        return str(Path(sys.executable).parent / "clawcodex-dev")

    async def _run_clawcodex(self, prompt: str) -> str:
        """子进程执行 ``clawcodex-dev -p <prompt> --provider P --model M``。

        解析 stdout 中的 LLM 回复，过滤尾部 ``Resume this session...`` 噪音。
        """
        exe = self._resolve_clawcodex_dev()
        cmd = [exe, "-p", prompt]
        if self._provider_name:
            cmd.extend(["--provider", self._provider_name])
        if self._model:
            cmd.extend(["--model", self._model])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError("clawcodex-dev timed out after 120s")

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"clawcodex-dev exited {proc.returncode}: {stderr_text}")

        result = stdout.decode("utf-8", errors="replace").strip()

        # 过滤尾部噪音："\nResume this session with: clawcodex --resume <id>"
        noise_pos = result.find(self._NOISE_MARKER)
        if noise_pos != -1:
            result = result[:noise_pos].strip()

        return result

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(candidates: list[dict], existing: list[dict]) -> str:
        lines: list[str] = ["Existing rules:"]
        for i, existing_rule in enumerate(existing, start=1):
            lines.append(f"{i}. [{existing_rule.get('category', '?')}] {existing_rule.get('summary', '')}")
            body = (existing_rule.get("body") or "").strip()
            if body:
                truncated = body[:120] + "..." if len(body) > 120 else body
                lines.append(f"   Body: {truncated}")

        lines.append("")
        lines.append("Candidate rules:")
        for candidate_idx, candidate in enumerate(candidates):
            label = f"C{candidate_idx + 1}"
            lines.append(f"{label}. [{candidate.get('category', '?')}] {candidate.get('summary', '')}")
            body = (candidate.get("body") or "").strip()
            if body:
                truncated = body[:120] + "..." if len(body) > 120 else body
                lines.append(f"   Body: {truncated}")

        lines.append("")
        lines.append("For each candidate reply EXACTLY one line in this format:")
        lines.append("C1: DUPLICATE <existing_id>")
        lines.append("C1: MERGE <existing_id>")
        lines.append("C1: CONFLICT <existing_id>")
        lines.append("C1: NEW")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reply parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_reply(reply: str, num_candidates: int) -> list[JudgeResult]:
        results: list[JudgeResult] = []
        reply_lower = reply.strip().lower()

        for candidate_idx in range(num_candidates):
            label = f"C{candidate_idx + 1}"
            pattern = re.compile(
                rf"{re.escape(label)}\s*:\s*(duplicate|merge|conflict|new)\s*(\d+)?",
                re.IGNORECASE,
            )
            match = pattern.search(reply_lower)
            if match:
                action = match.group(1)
                target_str = match.group(2)
                target_idx = int(target_str) - 1 if target_str is not None else None
                results.append(JudgeResult(action=action, target_idx=target_idx))  # type: ignore[arg-type]
            else:
                logger.warning("Failed to parse judge result for candidate %s, defaulting to new", label)
                results.append(JudgeResult(action="new"))

        return results


# Regex to find the ## Extracted Rules section in an agent reply.
_RULES_SECTION_RE = re.compile(
    r"^##\s+Extracted\s+Rules\s*\n(.*?)(?=\n##\s|\Z)",
    re.DOTALL | re.MULTILINE,
)

# Regex to parse individual rule items inside the section — strict form.
# Matches the canonical template format: `- [category] summary` + optional
# `Body: ...` on the next indented line. Kept for backward compatibility
# with the prompt template's documented format.
#
# The lookahead accepts the start of ANY next list item (strict `[cat]`,
# loose `**bold**`, or bare `\w` summary) so a strict item does not
# greedily swallow a following loose-format item (regex fix).
#
# The Body capture tolerates an optional leading list marker (`- ` or
# `* `) since LLMs frequently write `  - Body: ...` instead of `  Body:`
# (regex fix).
_RULE_ITEM_RE = re.compile(
    r"^\s*[-*]\s+\[([^\]]+)\]\s+(.+?)(?:\n\s*[-*]?\s*Body:\s*(.+?))?"
    r"(?=\n\s*[-*]\s+(?:\[|\*\*|\w)|\n\s*$|\Z)",
    re.DOTALL | re.MULTILINE,
)

# loose fallback regex for LLM output that deviates from the
# template. Tolerates three common deviations observed in practice:
#   (a) bold title instead of `[category]` — `- **Quote Style** — desc`
#   (b) Body line with a leading list marker — `  - Body: ...`
#   (c) bare summary with no category / title — `- Use double quotes`
# The category capture group is empty when no `[category]` prefix is
# present; `_infer_category()` then maps summary/body keywords to a
# canonical category (or falls back to `other`).
# Group layout: (1)=category-or-empty (2)=summary (3)=body-or-None
_RULE_ITEM_LOOSE_RE = re.compile(
    r"^\s*[-*]\s+(?:\[([^\]]*)\]\s+|\*\*([^*]*)\*\*\s*[-\u2014]\s+)?(.+?)"
    r"(?:\n\s*[-*]?\s*Body:\s*(.+?))?"
    r"(?=\n\s*[-*]\s+(?:\[|\*\*|\w)|\n\s*$|\Z)",
    re.DOTALL | re.MULTILINE,
)

# canonical category enum. Used by `_infer_category()` to map
# free-form summary/body text to a category when the agent omits `[cat]`.
_RULE_CATEGORIES = (
    "naming",
    "error_handling",
    "testing",
    "import_style",
    "code_style",
    "type_annotation",
    "architecture",
    "boilerplate",
    "security",
    "performance",
    "other",
)

# Keyword → category inference map (checked in order; first hit wins).
# Keys are lowercased substrings; a rule whose summary or body contains
# any keyword is assigned the corresponding category.
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("error_handling", ("except", "exception", "error", "raise", "try", "catch", "异常")),
    ("testing", ("test", "pytest", "assert", "fixture", "mock", "测试")),
    ("import_style", ("import", "导入", "排序")),
    ("type_annotation", ("type", "annotation", "typing", "mypy", "类型")),
    ("naming", ("name", "naming", "variable", "function", "class", "命名")),
    ("architecture", ("layer", "module", "depend", "分层", "依赖", "架构")),
    ("boilerplate", ("license", "header", "docstring", "doc", "注释", "版权")),
    ("security", ("security", "auth", "token", "secret", "安全", "密钥")),
    ("performance", ("perf", "performance", "speed", "cache", "性能", "缓存")),
    (
        "code_style",
        (
            "quote",
            "引号",
            "双引号",
            "单引号",
            "indent",
            "缩进",
            "space",
            "空格",
            "format",
            "格式",
            "style",
            "风格",
            "bracket",
            "括号",
        ),
    ),
]


def _infer_category(text: str) -> str:
    """Map free-form rule text to a canonical category by keyword match.

    Returns the first matching category from ``_CATEGORY_KEYWORDS``, or
    ``'other'`` if no keyword hits. Used when the agent omits the
    ``[category]`` prefix (fix for loose LLM output).
    """
    lowered = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(kw.lower() in lowered for kw in keywords):
            return category
    return "other"


_AUTO_MANAGED_COMMENT = (
    "# workflow.rules.yaml \u2014 \u7531 clawcodex orchestrator \u81ea\u52a8\u7ba1\u7406\n"
    "# \u89c4\u5219\u662f\u4ece PR review feedback \u4e2d\u5f52\u7eb3\u51fa\u7684\u53c2\u8003\u7ea6\u5b9a\uff0c\u975e\u5f3a\u5236\u7ea6\u675f\u3002\n"
    "# Agent \u5728\u9002\u5f53\u65f6\u673a\u4ee5 Read() \u67e5\u9605\u3002"
)

# Default quality-score weights.
_W_SUPPORT = 0.30
_W_SPECIFICITY = 0.25
_W_RECENCY = 0.10
_W_AUTHORITY = 0.15
_W_CRITICALITY = 0.20


# ---------------------------------------------------------------------------
# RuleStore
# ---------------------------------------------------------------------------


class RuleStore:
    """Read/write ``workflow.rules.yaml`` with atomic write safety."""

    DEFAULT_FILENAME = "workflow.rules.yaml"

    @staticmethod
    def resolve_path(workflow_path: str, rules_path: str) -> str:
        if rules_path:
            rules_file = Path(rules_path)
            if rules_file.is_absolute():
                return str(rules_file)
            return str((Path(workflow_path).parent / rules_path).resolve())
        return str((Path(workflow_path).parent / RuleStore.DEFAULT_FILENAME).resolve())

    @staticmethod
    def load(path: str) -> dict:
        """Load rules file.

        Returns ``{"version": 1, "rules": []}`` when the file does not
        exist (normal first-run state).  Raises ``yaml.YAMLError`` or
        ``OSError`` when the file exists but is corrupted, so callers
        can refuse to overwrite history rules.
        """
        rules_file = Path(path)
        if not rules_file.exists():
            return {"version": 1, "rules": []}
        raw = read_text_utf8(rules_file)
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise yaml.YAMLError(f"Rules file {path} is not a valid YAML mapping")
        data.setdefault("version", 1)
        data.setdefault("rules", [])
        return data

    @staticmethod
    def save(path: str, rules: list[dict], version: int = 1) -> None:
        rules_file = Path(path)
        content = yaml.dump(
            {"version": version, "rules": rules},
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        full_content = f"{_AUTO_MANAGED_COMMENT}\n{content}"
        tmp = rules_file.with_suffix(".yaml.tmp")
        try:
            write_text_utf8(tmp, full_content)
            tmp.replace(rules_file)
        except OSError as exc:
            logger.error("Failed to save rules file %s: %s", path, exc)
            raise

    @staticmethod
    def is_user_managed(path: str) -> bool:
        rules_file = Path(path)
        if not rules_file.exists():
            return False
        try:
            first = read_text_utf8(rules_file).splitlines()[0] if rules_file.stat().st_size > 0 else ""
            # 只检测首行（兼容多行 header 的扩展）
            marker = _AUTO_MANAGED_COMMENT.splitlines()[0]
            return marker not in first
        except (OSError, IndexError):
            return False

    @staticmethod
    def ensure_file(path: str) -> None:
        rules_file = Path(path)
        if rules_file.exists():
            return
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        RuleStore.save(path, [], version=1)


# ---------------------------------------------------------------------------
# RuleEngine
# ---------------------------------------------------------------------------


class RuleEngine:
    """Rule extraction, deduplication, merge, quality scoring, pruning."""

    def __init__(self, store: RuleStore | None = None) -> None:
        self.store = store or RuleStore()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract(agent_reply: str) -> list[dict]:
        match = _RULES_SECTION_RE.search(agent_reply)
        if not match:
            return []
        section = match.group(1).strip()
        rules: list[dict] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Track char spans already consumed by the strict regex so the
        # loose fallback does not double-count the same rule. Spans are
        # half-open [start, end) over the stripped section text.
        strict_spans: list[tuple[int, int]] = []

        for rule_match in _RULE_ITEM_RE.finditer(section):
            category = rule_match.group(1).strip()
            summary = rule_match.group(2).strip()
            body = rule_match.group(3).strip() if rule_match.group(3) else ""
            if not body and ":" in summary:
                parts = summary.split(":", 1)
                summary = parts[0].strip()
                body = parts[1].strip()
            rules.append(
                {
                    "category": category,
                    "summary": summary,
                    "body": body,
                    "source": "",
                    "support_count": 1,
                    "confidence": "medium",
                    "created_at": now,
                    "updated_at": now,
                    "conflict_with": [],
                    "last_applied": now,
                }
            )
            strict_spans.append((rule_match.start(), rule_match.end()))

        # loose fallback for LLM output that deviates from the
        # canonical `- [category] summary` template. Common deviations:
        # bold-title instead of `[cat]`, Body line with a leading `- `,
        # or a bare summary with no category prefix at all. We skip any
        # loose match whose start falls inside a strict span to avoid
        # duplicating rules the strict pass already captured.
        def _in_strict_span(pos: int) -> bool:
            for span_start, span_end in strict_spans:
                if span_start <= pos < span_end:
                    return True
            return False

        for rule_match in _RULE_ITEM_LOOSE_RE.finditer(section):
            if _in_strict_span(rule_match.start()):
                continue
            # Loose group layout: (1)=category-or-None (2)=bold-title-or-None
            # (3)=summary (4)=body-or-None. Exactly one of (1)/(2) is set.
            category = (rule_match.group(1) or "").strip()
            bold_title = (rule_match.group(2) or "").strip()
            summary = (rule_match.group(3) or "").strip()
            body = (rule_match.group(4).strip() if rule_match.group(4) else "").strip()
            if not summary:
                continue
            # If the agent used a bold title instead of [category], fold
            # the title into the summary (it is usually the rule name).
            if bold_title and not category:
                summary = f"{bold_title} — {summary}" if summary else bold_title
            if not body and ":" in summary:
                parts = summary.split(":", 1)
                summary = parts[0].strip()
                body = parts[1].strip()
            if not category or category not in _RULE_CATEGORIES:
                category = _infer_category(f"{summary} {body}")
            rules.append(
                {
                    "category": category,
                    "summary": summary,
                    "body": body,
                    "source": "",
                    "support_count": 1,
                    "confidence": "medium",
                    "created_at": now,
                    "updated_at": now,
                    "conflict_with": [],
                    "last_applied": now,
                }
            )
        return rules

    # ------------------------------------------------------------------
    # Semantic dedup + merge (Phase 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_and_merge(
        candidates: list[dict],
        existing: list[dict],
        _judge_results: list[JudgeResult] | None = None,
    ) -> list[dict]:
        """Phase 2 dedup + merge.

        When ``_judge_results`` is provided (LLM path), its decisions
        override the default all-new behaviour.  Exact dedup (same
        summary text, case-insensitive) is always applied as a fast
        path regardless of the judge.
        """
        # --- fast path: exact dedup ---------------------------------------------------
        merged = list(existing)
        existing_map: dict[str, dict] = {}
        for rule in merged:
            key = rule.get("summary", "").strip().lower()
            if key:
                existing_map[key] = rule

        remaining: list[tuple[int, dict]] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for orig_idx, candidate in enumerate(candidates):
            key = candidate.get("summary", "").strip().lower()
            if key and key in existing_map:
                existing_map[key]["support_count"] = existing_map[key].get("support_count", 1) + 1
                existing_map[key]["updated_at"] = now
            else:
                remaining.append((orig_idx, candidate))

        if not remaining:
            return merged

        # --- decide: LLM judge path or all-new fallback -------------------------------
        if _judge_results is not None:
            return _apply_judge_results(merged, remaining, _judge_results, now)

        # --- all-new fallback (LLM unavailable) ----------------------------------------
        # 不做语义去重/合并/冲突检测，所有 candidate 作为新规则追加。
        # max_rules + prune 会自动控制规则库大小，避免无限膨胀。
        next_id = max((rule.get("id", 0) for rule in merged), default=0) + 1
        for _orig_idx, candidate in remaining:
            next_id = _append_new_rule(merged, candidate, next_id, now)
        return merged

    # ------------------------------------------------------------------
    # Quality scoring (Phase 2)
    # ------------------------------------------------------------------

    @staticmethod
    def score(rule: dict) -> float:
        """Four-dimension quality score used for auto-pruning.

        Dimensions:
          1. ``support_count_norm``  — capped at 5
          2. ``specificity``          — has body text?
          3. ``recency``              — days since creation (90-day linear decay)
          4. ``authority``            — derived from rule ``confidence`` field
             (also drives ``criticality``; both use the same confidence map)

        Returns a float in [0.0, 1.0].
        """
        now = datetime.now(timezone.utc)

        # 1. Support count (capped at 5)
        support = rule.get("support_count", 1)
        support_norm = min(support, 5) / 5.0

        # 2. Specificity: body text → 1.0, only summary → 0.3
        body = rule.get("body", "") or ""
        specificity = 1.0 if len(body.strip()) > 20 else 0.3

        # 3. Recency: linear decay over 90 days
        created_str = rule.get("created_at", "")
        days = 999.0
        if created_str:
            try:
                created = datetime.fromisoformat(created_str)
                days = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                pass
        recency = max(0.0, 1.0 - days / 90.0)

        # 4. Authority / criticality derived from confidence field
        _conf_weight = {"high": 0.9, "medium": 0.7, "low": 0.5}
        authority = _conf_weight.get(rule.get("confidence", "medium"), 0.7)
        criticality = authority

        return (
            _W_SUPPORT * support_norm
            + _W_SPECIFICITY * specificity
            + _W_RECENCY * recency
            + _W_AUTHORITY * authority
            + _W_CRITICALITY * criticality
        )

    @staticmethod
    def prune(rules: list[dict], max_rules: int) -> list[dict]:
        """Drop lowest-scoring rules when ``len(rules) > max_rules``.

        Returns a trimmed list (never exceeds ``max_rules``).
        """
        if max_rules <= 0 or len(rules) <= max_rules:
            return list(rules)

        scored = [(RuleEngine.score(rule), rule) for rule in rules]
        # Sort descending by score, keep top max_rules
        scored.sort(key=lambda item: item[0], reverse=True)
        kept = [rule for _, rule in scored[:max_rules]]

        dropped = len(rules) - len(kept)
        if dropped > 0:
            logger.info("Pruned %d low-quality rule(s), kept %d", dropped, len(kept))

        return kept

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_rules_path(config: Any, workflow_path: str | None) -> str | None:
        rules_config = getattr(config, "rules", None)
        if not rules_config or not getattr(rules_config, "enabled", False):
            return None
        rules_path = getattr(rules_config, "path", "") or ""
        if not workflow_path:
            return None
        return RuleStore.resolve_path(workflow_path, rules_path)


# ---------------------------------------------------------------------------
# LLM judge result applier
# ---------------------------------------------------------------------------


def _append_new_rule(merged: list[dict], candidate: dict, next_id: int, now: str) -> int:
    """Append *candidate* as a new rule, assigning id and updated_at.

    Returns the next available id (incremented).
    """
    candidate["id"] = next_id
    candidate["updated_at"] = now
    merged.append(candidate)
    return next_id + 1


def _apply_judge_results(
    merged: list[dict],
    remaining: list[tuple[int, dict]],
    judge_results: list[JudgeResult],
    now: str,
) -> list[dict]:
    """Apply ``JudgeResult`` list from LLM to produce the final merged list.

    Each element in *judge_results* corresponds to the **original**
    candidate index (``orig_idx``).  The method processes *remaining*
    candidates that survived exact dedup.
    """
    next_id = max((rule.get("id", 0) for rule in merged), default=0) + 1
    for orig_idx, candidate in remaining:
        jr = judge_results[orig_idx] if orig_idx < len(judge_results) else None
        if jr is None or jr.action == "new":
            # New rule — append
            next_id = _append_new_rule(merged, candidate, next_id, now)
        elif jr.action == "duplicate" and jr.target_idx is not None:
            # Duplicate — increment support_count on the target
            target_idx = jr.target_idx
            if target_idx < len(merged):
                merged[target_idx]["support_count"] = merged[target_idx].get("support_count", 1) + 1
                merged[target_idx]["updated_at"] = now
            else:
                # Fallback: target out of range, treat as new
                next_id = _append_new_rule(merged, candidate, next_id, now)
        elif jr.action == "merge" and jr.target_idx is not None:
            # Merge — combine candidate into the target existing rule
            target_idx = jr.target_idx
            if target_idx < len(merged):
                merged[target_idx] = _merge_two_rules(merged[target_idx], candidate)
            else:
                next_id = _append_new_rule(merged, candidate, next_id, now)
        elif jr.action == "conflict" and jr.target_idx is not None:
            # Conflict — keep separate, mark with index refs
            target_idx = jr.target_idx
            if target_idx < len(merged):
                candidate["_conflict_with_idx"] = target_idx
                merged[target_idx].setdefault("_conflict_with_idx", []).append(len(merged))
                candidate["updated_at"] = now
                merged.append(candidate)
                next_id += 1
            else:
                next_id = _append_new_rule(merged, candidate, next_id, now)
        else:
            next_id = _append_new_rule(merged, candidate, next_id, now)
    return merged


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def _merge_two_rules(existing_rule: dict, candidate_rule: dict) -> dict:
    """Merge two similar rules into a single enriched rule.

    ``existing_rule`` is the existing (higher-confidence) rule;
    ``candidate_rule`` is the candidate.  The merge picks the best of
    each field.
    """
    # Summary: pick the longer / more specific one
    existing_summary = (existing_rule.get("summary") or "").strip()
    candidate_summary = (candidate_rule.get("summary") or "").strip()
    merged_summary = existing_summary if len(existing_summary) >= len(candidate_summary) else candidate_summary

    # Body: pick the longer one, or concatenate if both present
    existing_body = (existing_rule.get("body") or "").strip()
    candidate_body = (candidate_rule.get("body") or "").strip()
    if existing_body and candidate_body:
        merged_body = existing_body if len(existing_body) >= len(candidate_body) else candidate_body
    else:
        merged_body = existing_body or candidate_body

    # Category: prefer the more specific one (longer string), or 'multi' if incompatible
    existing_category = (existing_rule.get("category") or "").strip()
    candidate_category = (candidate_rule.get("category") or "").strip()
    if existing_category and candidate_category and existing_category != candidate_category:
        merged_category = "multi"
    else:
        merged_category = existing_category or candidate_category or "other"

    # Support count: sum
    support = (existing_rule.get("support_count") or 1) + (candidate_rule.get("support_count") or 1)

    # Source: append if different
    existing_source = (existing_rule.get("source") or "").strip()
    candidate_source = (candidate_rule.get("source") or "").strip()
    merged_source = existing_source
    if candidate_source and candidate_source not in merged_source:
        merged_source = f"{merged_source}; {candidate_source}" if merged_source else candidate_source

    # Confidence: take the higher one
    conf_order = {"low": 0, "medium": 1, "high": 2}
    existing_conf = conf_order.get(existing_rule.get("confidence", "medium"), 1)
    candidate_conf = conf_order.get(candidate_rule.get("confidence", "medium"), 1)
    merged_confidence = (
        "high"
        if max(existing_conf, candidate_conf) >= 2
        else "medium"
        if max(existing_conf, candidate_conf) >= 1
        else "low"
    )

    # Timestamps
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    existing_created = existing_rule.get("created_at", "") or now
    candidate_created = candidate_rule.get("created_at", "") or now
    merged_created = min(existing_created, candidate_created)

    return {
        "summary": merged_summary,
        "body": merged_body,
        "category": merged_category,
        "support_count": support,
        "source": merged_source,
        "confidence": merged_confidence,
        "conflict_with": list(set(existing_rule.get("conflict_with", []) + candidate_rule.get("conflict_with", []))),
        "created_at": merged_created,
        "updated_at": now,
        "last_applied": now,
    }
