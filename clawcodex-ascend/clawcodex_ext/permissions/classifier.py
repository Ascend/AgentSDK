#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Auto Mode LLM Classifier.

This module extends the rule-based auto_mode_classify with LLM-based
classification for uncertain scenarios. When the rule-based classifier
returns an uncertain result, the LLM classifier can provide additional
context-aware judgment.

Features:
- LLM-based classification for complex/uncertain tool calls
- TTL-based cache to avoid repeated LLM calls
- Confidence threshold for auto-approval decisions
- Integration with existing rule-based classifier
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.permissions.check import AutoModeDecision

from clawcodex_ext.permissions.types import ToolPermissionContext

from .danger_detector import detect_dangerous_tool_call

log = logging.getLogger(__name__)

_THINKING_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\"decision\"[^{}]*\}", re.DOTALL)


def _extract_json_from_response(content: str) -> dict[str, Any] | None:
    """Extract JSON object from LLM response, handling thinking tags and markdown."""
    cleaned = _THINKING_TAG_RE.sub("", content).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass  # Invalid candidate; continue with the surrounding fallback.

    match = _JSON_OBJECT_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass  # Invalid candidate; continue with the surrounding fallback.

    brace_start = cleaned.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


@dataclass(frozen=True)
class LLMClassificationResult:
    decision: str  # "AUTO_ALLOW" | "AUTO_DENY" | "ASK_USER"
    reasoning: str
    confidence: float  # 0.0-1.0
    cache_key: str


@dataclass
class ClassificationCache:
    entries: dict[str, tuple[LLMClassificationResult, float]] = field(default_factory=dict)
    ttl_seconds: float = 600.0  # 10 minutes
    max_entries: int = 1000

    def get(self, cache_key: str) -> LLMClassificationResult | None:
        if cache_key not in self.entries:
            return None
        result, timestamp = self.entries[cache_key]
        if time.time() - timestamp > self.ttl_seconds:
            self.entries.pop(cache_key, None)
            return None
        return result

    def set(self, cache_key: str, result: LLMClassificationResult) -> None:
        if len(self.entries) >= self.max_entries:
            oldest_key = min(self.entries.keys(), key=lambda k: self.entries[k][1])
            self.entries.pop(oldest_key, None)
        self.entries[cache_key] = (result, time.time())

    def clear(self) -> None:
        self.entries.clear()


_cache = ClassificationCache()


def get_cache() -> ClassificationCache:
    return _cache


AUTO_MODE_LLM_PROMPT = """你是一个安全分类器，判断以下工具调用是否可以在 auto 模式下自动执行。

工具: {tool_name}
命令/参数: {command}
当前目录: {cwd}
文件类型: {file_type}
最近操作: {recent_operations}

考虑因素:
1. 工具类型：Read/Glob/Grep 通常安全，Write/Edit/Bash 需谨慎
2. 命令风险：是否包含 rm -rf、sudo、chmod 777 等危险操作
3. 目标路径：是否在 .git、.vscode、.clawcodex 等保护目录内
4. 重复模式：如果是第 N 次执行类似操作，可放宽限制

输出 JSON 格式:
{{"decision": "AUTO_ALLOW" | "AUTO_DENY" | "ASK_USER", "reasoning": "简短说明（1-2 句）", "confidence": 0.0-1.0}}

注意：
- AUTO_ALLOW：确认安全，可直接执行
- AUTO_DENY：明确危险，应拒绝
- ASK_USER：不确定，需要人工确认"""


def _compute_cache_key(
    tool_name: str,
    tool_input: dict[str, Any],
    context: ToolPermissionContext,
) -> str:
    key_data = {
        "tool": tool_name,
        "input_hash": hashlib.md5(json.dumps(tool_input, sort_keys=True).encode()).hexdigest()[:16],  # nosec B324
        "cwd": str(context.cwd) if hasattr(context, "cwd") else "",
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()  # nosec B324


def _build_prompt(
    tool_name: str,
    tool_input: dict[str, Any],
    context: ToolPermissionContext,
    recent_operations: list[str] | None = None,
) -> str:
    command = ""
    if tool_name == "Bash":
        command = tool_input.get("command", "")
    elif tool_name in ("Write", "Edit"):
        command = tool_input.get("file_path", "")
    else:
        command = json.dumps(tool_input, indent=2)[:200]

    cwd = str(context.cwd) if hasattr(context, "cwd") else "unknown"
    file_type = ""
    if "file_path" in tool_input:
        path = tool_input["file_path"]
        if "." in path:
            file_type = path.rsplit(".", 1)[-1]

    recent = ""
    if recent_operations:
        recent = "\n".join(f"- {op}" for op in recent_operations[:5])

    return AUTO_MODE_LLM_PROMPT.format(
        tool_name=tool_name,
        command=command,
        cwd=cwd,
        file_type=file_type or "unknown",
        recent_operations=recent or "none",
    )


def llm_classify_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    context: ToolPermissionContext,
    provider: Any | None = None,
    recent_operations: list[str] | None = None,
) -> LLMClassificationResult:
    cache_key = _compute_cache_key(tool_name, tool_input, context)
    cached = _cache.get(cache_key)
    if cached is not None:
        log.debug("LLM classifier cache hit for %s", tool_name)
        return cached

    if provider is None:
        try:
            from clawcodex_ext.providers.runtime import build_provider_from_config

            provider = build_provider_from_config("openai")
        except Exception as e:
            log.warning("Failed to get provider for LLM classifier: %s", e)
            return LLMClassificationResult(
                decision="ASK_USER",
                reasoning="No provider available for classification",
                confidence=0.0,
                cache_key=cache_key,
            )

    prompt = _build_prompt(tool_name, tool_input, context, recent_operations)

    try:
        messages = [{"role": "user", "content": prompt}]
        response = provider.chat(messages)
        content = response.content.strip()
        log.info("LLM classifier response received (chars=%d)", len(content))

        result_data = _extract_json_from_response(content)

        if result_data is not None:
            decision = result_data.get("decision", "ASK_USER")
            reasoning = result_data.get("reasoning", "")
            confidence = float(result_data.get("confidence", 0.5))
        else:
            log.warning("LLM classifier: no JSON found, falling back to keyword match")
            if "AUTO_ALLOW" in content:
                decision = "AUTO_ALLOW"
            elif "AUTO_DENY" in content:
                decision = "AUTO_DENY"
            else:
                decision = "ASK_USER"
            reasoning = content[:100]
            confidence = 0.5

        log.info(
            "LLM classifier parsed: decision=%s, confidence=%s, reasoning=%s",
            decision,
            confidence,
            reasoning,
        )

        result = LLMClassificationResult(
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            cache_key=cache_key,
        )

        if decision != "AUTO_DENY":
            _cache.set(cache_key, result)

        return result

    except Exception as e:
        log.warning("LLM classifier error: %s", e)
        return LLMClassificationResult(
            decision="ASK_USER",
            reasoning=f"Classification error: {e}",
            confidence=0.0,
            cache_key=cache_key,
        )


def auto_mode_classify_with_llm(
    tool_name: str,
    tool_input: dict[str, Any],
    context: ToolPermissionContext,
    provider: Any | None = None,
    recent_operations: list[str] | None = None,
    use_llm_for_uncertain: bool = True,
    _original_classify: Any | None = None,
) -> AutoModeDecision:
    if _original_classify is None:
        from src.permissions.check import auto_mode_classify as _original_classify

    rule_result = _original_classify(tool_name, tool_input, context)
    log.info(
        "Rule classifier: tool=%s, allow=%s, reason=%s",
        tool_name,
        rule_result.allow,
        rule_result.reason,
    )

    if rule_result.allow:
        return rule_result

    _HARD_DENY_REASONS = frozenset(
        {
            "empty command",
            "complex command structure",
            "MCP tools require explicit approval",
        }
    )

    if rule_result.reason in _HARD_DENY_REASONS:
        return rule_result

    if not use_llm_for_uncertain:
        return rule_result

    is_danger, danger_reason = detect_dangerous_tool_call(tool_name, tool_input)
    if is_danger:
        log.info("Hard-coded danger detected: %s", danger_reason)
        return rule_result

    log.info(
        "Calling LLM classifier for uncertain command: tool=%s, reason=%s",
        tool_name,
        rule_result.reason,
    )
    llm_result = llm_classify_tool_call(tool_name, tool_input, context, provider, recent_operations)
    log.info(
        "LLM result: decision=%s, confidence=%s",
        llm_result.decision,
        llm_result.confidence,
    )

    if llm_result.decision == "AUTO_ALLOW" and llm_result.confidence >= 0.8:
        log.info("LLM override: ALLOW")
        # Re-import at call site so pylint can see the binding
        from src.permissions.check import AutoModeDecision as _AutoModeDecision

        return _AutoModeDecision(
            allow=True,
            reason=f"LLM override: {llm_result.reasoning}",
        )

    if llm_result.decision == "AUTO_DENY":
        # Re-import at call site so pylint can see the binding
        from src.permissions.check import AutoModeDecision as _AutoModeDecision

        return _AutoModeDecision(
            allow=False,
            reason=f"LLM confirmed danger: {llm_result.reasoning}",
        )

    return rule_result


__all__ = [
    "AUTO_MODE_LLM_PROMPT",
    "ClassificationCache",
    "LLMClassificationResult",
    "auto_mode_classify_with_llm",
    "get_cache",
    "llm_classify_tool_call",
]
