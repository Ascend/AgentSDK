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

"""Simple dict-based i18n for community radar reports.

Deliberately not using gettext — the string count (~50 entries) doesn't
warrant the complexity. All user-facing strings in reporter.py and the
Jinja2 templates flow through :func:`get_text` so adding a new language
is a matter of extending :data:`STRINGS`.
"""

from __future__ import annotations

import logging
from typing import Literal

Language = Literal["zh", "en"]

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# String table — semantic_key → {zh: ..., en: ...}
# ---------------------------------------------------------------------------


STRINGS: dict[str, dict[str, str]] = {
    # ── Period labels ────────────────────────────────────────────────
    "period_weekly": {
        "zh": "周报",
        "en": "Weekly Digest",
    },
    "period_monthly": {
        "zh": "月报",
        "en": "Monthly Digest",
    },
    "period_full": {
        "zh": "总报",
        "en": "Full Report",
    },
    # ── Report chrome ────────────────────────────────────────────────
    "report_title": {
        "zh": "ClawCodex 社区动态报告",
        "en": "ClawCodex Community Radar Report",
    },
    "generated_at": {
        "zh": "生成时间",
        "en": "Generated at",
    },
    "coverage_label": {
        "zh": "覆盖范围",
        "en": "Coverage",
    },
    "coverage_projects": {
        "zh": "个项目",
        "en": " projects",
    },
    "coverage_versions": {
        "zh": "个版本",
        "en": " versions",
    },
    "coverage_features": {
        "zh": "条特性记录",
        "en": " feature records",
    },
    "filtered_info": {
        "zh": "已过滤",
        "en": "Filtered",
    },
    "llm_assisted": {
        "zh": "LLM 辅助: 本期分类/摘要经模型精炼",
        "en": "LLM-assisted: classification & summary refined by model",
    },
    # ── Section headings ─────────────────────────────────────────────
    "section_highlights": {
        "zh": "本期重点",
        "en": "Highlights",
    },
    "highlights_desc": {
        "zh": "以下为{period_word}（{date_range}）评分较高、属于核心模块的重要特性更新。",
        "en": "Major feature updates ({date_range}) with high scores and core-module impact.",
    },
    "highlights_desc_full": {
        "zh": "以下为{period_word}评分较高、属于核心模块的重要特性更新。",
        "en": "All major feature updates with high scores and core-module impact.",
    },
    "section_detail_table": {
        "zh": "新增特性详表",
        "en": "New Feature Details",
    },
    "section_summary": {
        "zh": "摘要",
        "en": "Summary",
    },
    "section_new_features": {
        "zh": "新增候选特性",
        "en": "New Candidate Features",
    },
    "section_breaking": {
        "zh": "破坏性变更预警",
        "en": "Breaking Change Alerts",
    },
    "section_distribution": {
        "zh": "分类分布",
        "en": "Category Distribution",
    },
    "section_by_domain": {
        "zh": "按领域",
        "en": "By Domain",
    },
    "section_by_subcategory": {
        "zh": "按子分类",
        "en": "By Subcategory",
    },
    "section_errors": {
        "zh": "抓取错误",
        "en": "Fetch Errors",
    },
    "section_comparison": {
        "zh": "变化对比 (vs 上期)",
        "en": "Changes (vs Previous)",
    },
    "comparison_baseline": {
        "zh": "对比基准",
        "en": "Baseline",
    },
    "comparison_new": {
        "zh": "新增特性 ({n})",
        "en": "New Features ({n})",
    },
    "comparison_disappeared": {
        "zh": "消失特性 ({n})",
        "en": "Disappeared Features ({n})",
    },
    "comparison_score": {
        "zh": "评分变化 ({n})",
        "en": "Score Changes ({n})",
    },
    "source_from": {
        "zh": "来源",
        "en": "Source",
    },
    # ── Table headers ────────────────────────────────────────────────
    "th_feature": {
        "zh": "特性",
        "en": "Feature",
    },
    "th_source": {
        "zh": "来源",
        "en": "Source",
    },
    "th_score": {
        "zh": "评分",
        "en": "Score",
    },
    "th_category": {
        "zh": "分类",
        "en": "Category",
    },
    "th_desc": {
        "zh": "简述",
        "en": "Description",
    },
    "th_type": {
        "zh": "类型",
        "en": "Type",
    },
    "th_new": {
        "zh": "新增",
        "en": "New",
    },
    "th_project": {
        "zh": "项目",
        "en": "Project",
    },
    "th_impact": {
        "zh": "影响评估",
        "en": "Impact Assessment",
    },
    "th_old_score": {
        "zh": "旧评分",
        "en": "Old Score",
    },
    "th_new_score": {
        "zh": "新评分",
        "en": "New Score",
    },
    "th_change": {
        "zh": "变化",
        "en": "Change",
    },
    # ── Placeholder / empty ──────────────────────────────────────────
    "none": {
        "zh": "（无）",
        "en": "(none)",
    },
    "no_activity": {
        "zh": "（本期暂无显著动态。）",
        "en": "(No significant activity this period.)",
    },
    "no_new_features_summary": {
        "zh": "本期没有新的候选特性。",
        "en": "No new candidate features this period.",
    },
    "no_new_features": {
        "zh": "（无新增特性）",
        "en": "(No new features)",
    },
    "no_disappeared": {
        "zh": "（无消失特性）",
        "en": "(No disappeared features)",
    },
    "no_score_changes": {
        "zh": "（无显著评分变化）",
        "en": "(No significant score changes)",
    },
    "no_breaking": {
        "zh": "（本月暂无）",
        "en": "(None this month)",
    },
    # ── Monthly-specific ─────────────────────────────────────────────
    "monthly_top10": {
        "zh": "月度 Top-10 高分候选",
        "en": "Monthly Top-10 Candidates",
    },
    "trend_category": {
        "zh": "趋势分类",
        "en": "Trend Categories",
    },
    "focus_projects": {
        "zh": "重点关注项目",
        "en": "Focus Projects",
    },
    "monthly_breaking_summary": {
        "zh": "月度破坏性变更汇总",
        "en": "Monthly Breaking Change Summary",
    },
    "counter_items": {
        "zh": "条",
        "en": "items",
    },
    "counter_candidates": {
        "zh": "条候选",
        "en": "candidates",
    },
    "counter_filtered": {
        "zh": "条已过滤",
        "en": "items filtered",
    },
    # ── Feature labeling ─────────────────────────────────────────────
    "this_project_only": {
        "zh": "（仅此项目）",
        "en": "(this project only)",
    },
    "also_in": {
        "zh": "同时出现于",
        "en": "Also in",
    },
    # ── Impact assessments ───────────────────────────────────────────
    "impact_multi_project": {
        "zh": "中—多项目已采纳，需评估兼容",
        "en": "Medium — adopted by multiple projects; compatibility review needed",
    },
    "impact_migration": {
        "zh": "中—需评估迁移成本",
        "en": "Medium — migration cost evaluation needed",
    },
    # ── Summary template ─────────────────────────────────────────────
    "summary_template": {
        "zh": "本期共发现 {n} 条候选特性，主要集中在 {cats} 方向。",
        "en": "This period found {n} candidate features, mainly in {cats}.",
    },
    # ── Highlight entry template ─────────────────────────────────────
    "highlight_score": {
        "zh": "评分",
        "en": "Score",
    },
    "highlight_category": {
        "zh": "分类",
        "en": "Category",
    },
    "highlight_source": {
        "zh": "来源",
        "en": "Source",
    },
    "view_detail_link": {
        "zh": "查看详情",
        "en": "View",
    },
    # ── Filtered summary ─────────────────────────────────────────────
    "filtered_summary": {
        "zh": "{n} 条 bugfix/文档更新已过滤",
        "en": "{n} bugfix/doc items filtered out",
    },
    # ── LLM prompt for importance classification (language-neutral) ──
    # The zh prompt is always used so MAJOR/MINOR decisions are identical
    # regardless of the report language, and title_zh/desc_zh translations
    # are generated in the same LLM call.  The en prompt is kept as a
    # reference / fallback but is not currently selected by the pipeline.
    "llm_importance_prompt": {
        "zh": (
            "你是一个开源社区分析助手。以下是本周从各开源项目收集到的 {n} 条特性更新。\n"
            "请完成三件事：\n"
            "1. 判断每条特性属于「重大更新」(MAJOR)还是「小更新」(MINOR)。\n"
            "   请用以下绝对标准独立判断每条特性，不要与列表中的其他特性比较：\n"
            "   MAJOR（重大更新）——满足以下至少 2 条：\n"
            "     a. 引入了一个全新的架构层、子系统或核心能力（不只是改进已有功能）\n"
            "     b. 改变了 Agent/工具/模型的使用范式或工作流（不只是增加配置选项）\n"
            "     c. 被至少 2 个不同的知名项目独立采纳或实现（跨项目趋势信号）\n"
            "     d. 解决了 AI 编程/智能体领域一个公认的痛点（不只是修了一个 bug）\n"
            "   MINOR（小更新）——不满足以上任何一条，或只满足一条。\n"
            "   校准规则（非常重要）：\n"
            "     - 用相同的绝对尺度评判每条特性，不受当前批次质量高低的影响。\n"
            "     - 一个批次中 MAJOR 通常不超过 10%，如果远超此比例请重新逐条审视。\n"
            "     - 标题含 'fix'、'chore'、'docs'、'refactor'、'test' 的通常是 MINOR。\n"
            "     - 纯 CLI 参数调整、日志改进、内部重构、依赖升级一律是 MINOR。\n"
            "2. 对每条 MAJOR 特性，写一句 50-100 字的中文介绍，概括它的核心价值和影响面。\n"
            "3. 将所有特性的 title 翻译为中文填入 title_zh，将 description 翻译为中文（60字以内）填入 desc_zh。\n\n"
            "返回一个 JSON 数组（不要包含 markdown 代码块或其他文字）：\n"
            '[{{"id": "特性ID", "level": "MAJOR", "highlight": "介绍文字", "title_zh": "中文标题", "desc_zh": "中文简述"}}, ...]\n\n'
            "特性列表：\n{features_json}"
        ),
        "en": (
            "You are an open-source community analyst. Below are {n} feature updates "
            "collected from various open-source projects this period.\n"
            "Please do three things:\n"
            "1. Classify each feature as MAJOR or MINOR.\n"
            "   Use these absolute criteria for each feature independently:\n"
            "   MAJOR — meets at least 2 of:\n"
            "     a. Introduces a new architectural layer, subsystem, or core capability (not just improvements)\n"
            "     b. Changes the paradigm or workflow for agents/tools/models (not just a config option)\n"
            "     c. Adopted or implemented independently by at least 2 well-known projects\n"
            "     d. Solves a recognized pain point in AI coding/agents (not just a bugfix)\n"
            "   MINOR — meets 0 or 1 of the above.\n"
            "   Calibration rules (important):\n"
            "     - Judge every feature by the same absolute scale, regardless of batch quality.\n"
            "     - MAJOR should typically not exceed 10% of the batch. Re-check if much higher.\n"
            "     - Titles with 'fix', 'chore', 'docs', 'refactor', 'test' are usually MINOR.\n"
            "     - CLI flag tweaks, logging changes, internal refactors, dep bumps are always MINOR.\n"
            "2. For each MAJOR, write a one-sentence introduction (50-100 chars) summarizing its value.\n"
            "3. Translate every title and description to English (title_en, desc_en).\n\n"
            "Return a JSON array (no markdown blocks):\n"
            '[{{"id": "feature_id", "level": "MAJOR", "highlight": "intro text", "title_en": "English title", "desc_en": "English description"}}, ...]\n\n'
            "Feature list:\n{features_json}"
        ),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_text(key: str, lang: str = "zh", **kwargs: object) -> str:
    """Return the translated string for *key* in *lang*.

    Supports ``**kwargs`` for ``str.format()`` substitution, e.g.
    ``get_text("comparison_new", lang="en", n=5)`` → ``"New Features (5)"``.

    Falls back to ``zh`` when *lang* or *key* is unknown, and returns the
    raw key (wrapped in ``??key??``) as a last resort so a missing
    translation is visually obvious in the rendered report.
    """
    lang = lang if lang in ("zh", "en") else "zh"
    entry = STRINGS.get(key)
    if entry is None:
        return f"??{key}??"
    text = entry.get(lang) or entry.get("zh", f"??{key}??")
    if kwargs:
        try:
            text = text.format(**{k: str(v) for k, v in kwargs.items()})
        except (KeyError, ValueError):
            _log.warning("get_text(%r, %r): format placeholders not satisfied; rendering raw template", key, lang)
    return text


def _format_date_range(period_start: str) -> str:
    """Format a date range from *period_start* to now.

    ``period_start`` is ISO-8601 (e.g. ``2026-07-03T00:00:00Z``).
    Returns ``YYYY-MM-DD ~ YYYY-MM-DD``, or ``??`` when missing.
    """
    if not period_start:
        return "?? ~ ??"
    start_date = period_start[:10]
    from datetime import datetime, timezone

    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{start_date} ~ {end_date}"


def build_template_labels(lang: str = "zh", *, period: str = "weekly", period_start: str = "") -> dict[str, str]:
    """Return a flat ``labels`` dict for Jinja2 template rendering.

    Every label key used by ``weekly_digest.md.j2`` / ``monthly_digest.md.j2``
    must appear here so templates never hardcode Chinese strings.

    This is intentionally a separate function (not auto-generated from
    :data:`STRINGS`) so the template variable names can differ from the
    semantic keys used by the Python-side reporter.
    """
    period_word = get_text(f"period_{period}", lang)
    date_range = _format_date_range(period_start)
    return {
        # Chrome
        "generated_at": get_text("generated_at", lang),
        "coverage_label": get_text("coverage_label", lang),
        "coverage_projects": get_text("coverage_projects", lang),
        "coverage_versions": get_text("coverage_versions", lang),
        "coverage_features": get_text("coverage_features", lang),
        "filtered_info": get_text("filtered_info", lang),
        "llm_assisted": get_text("llm_assisted", lang),
        # Sections
        "section_highlights": get_text("section_highlights", lang),
        "highlights_desc": get_text(
            "highlights_desc_full" if period == "full" else "highlights_desc",
            lang,
            period_word=period_word,
            date_range=date_range,
        ),
        "section_detail_table": get_text("section_detail_table", lang),
        "section_summary": get_text("section_summary", lang),
        "section_new_features": get_text("section_new_features", lang),
        "section_breaking": get_text("section_breaking", lang),
        "section_distribution": get_text("section_distribution", lang),
        "section_by_domain": get_text("section_by_domain", lang),
        "section_by_subcategory": get_text("section_by_subcategory", lang),
        "section_errors": get_text("section_errors", lang),
        "section_comparison": get_text("section_comparison", lang),
        "comparison_baseline": get_text("comparison_baseline", lang),
        "comparison_new": get_text("comparison_new", lang),
        "comparison_disappeared": get_text("comparison_disappeared", lang),
        "comparison_score": get_text("comparison_score", lang),
        "source_from": get_text("source_from", lang),
        # Table headers
        "th_feature": get_text("th_feature", lang),
        "th_source": get_text("th_source", lang),
        "th_score": get_text("th_score", lang),
        "th_category": get_text("th_category", lang),
        "th_desc": get_text("th_desc", lang),
        "th_type": get_text("th_type", lang),
        "th_new": get_text("th_new", lang),
        "th_project": get_text("th_project", lang),
        "th_impact": get_text("th_impact", lang),
        "th_old_score": get_text("th_old_score", lang),
        "th_new_score": get_text("th_new_score", lang),
        "th_change": get_text("th_change", lang),
        # Empty / placeholder
        "none": get_text("none", lang),
        "no_activity": get_text("no_activity", lang),
        "no_new_features": get_text("no_new_features", lang),
        "no_disappeared": get_text("no_disappeared", lang),
        "no_score_changes": get_text("no_score_changes", lang),
        "no_breaking": get_text("no_breaking", lang),
        # Monthly-specific
        "monthly_top10": get_text("monthly_top10", lang),
        "trend_category": get_text("trend_category", lang),
        "focus_projects": get_text("focus_projects", lang),
        "monthly_breaking_summary": get_text("monthly_breaking_summary", lang),
        "counter_items": get_text("counter_items", lang),
        "counter_candidates": get_text("counter_candidates", lang),
        "counter_filtered": get_text("counter_filtered", lang),
        # Feature labeling
        "this_project_only": get_text("this_project_only", lang),
        "also_in": get_text("also_in", lang),
        # Highlight labels
        "highlight_score": get_text("highlight_score", lang),
        "highlight_category": get_text("highlight_category", lang),
        "highlight_source": get_text("highlight_source", lang),
        # URL link
        "view_detail_link": get_text("view_detail_link", lang),
    }
