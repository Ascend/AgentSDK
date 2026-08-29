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

# pylint: disable=relative-beyond-top-level
"""Community digest generator — Markdown/JSON/Jinja2 output + Phase 4 proposals."""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RadarConfig
from .i18n import _format_date_range, build_template_labels, get_text
from .models import (
    CommunityDigest,
    DigestStats,
    FeatureRecord,
    FeatureScore,
    FeatureType,
    HistoryComparison,
    ScoredFeature,
    get_root,
    utc_now_iso,
)

_log = logging.getLogger(__name__)


def _short_desc(text: str, *, limit: int = 120) -> str:
    cleaned = (text or "").strip().replace("\n", " ")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f9ff"  # Misc symbols, emoticons, supplementals
    "\U0001fa00-\U0001faff"  # Chess symbols, symbols extended-A
    "\U00002600-\U000027bf"  # Misc symbols (dingbats, etc.)
    "\U00002b50"  # ⭐ (white medium star)
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"^\*\*\s*\*\*\s*", "", cleaned)
    return cleaned.strip()


def _escape(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def _impact_for(record: FeatureRecord, lang: str = "zh") -> str:
    if record.related_projects:
        return get_text("impact_multi_project", lang)
    return get_text("impact_migration", lang)


def _t_title(record: FeatureRecord, llm_info: dict[str, str], lang: str) -> str:
    if lang == "zh" and llm_info.get("title_zh"):
        return _strip_emoji(llm_info["title_zh"])
    return _strip_emoji(record.title)


def _t_desc(record: FeatureRecord, llm_info: dict[str, str], lang: str) -> str:
    if lang == "zh":
        hl = llm_info.get("highlight", "")
        if hl:
            return _strip_emoji(hl)
        desc_zh = llm_info.get("desc_zh", "")
        if desc_zh:
            return _strip_emoji(desc_zh)
    return _strip_emoji(_short_desc(record.description))


def _make_url_suffix(url: str, lang: str) -> str:
    if not url:
        return ""
    label = get_text("view_detail_link", lang)
    return f" — [{label}]({url})"


def _title_link(record: FeatureRecord, llm_info: dict[str, str], lang: str) -> str:
    title = _escape(_t_title(record, llm_info, lang))
    if record.url:
        return f"[{title}]({record.url})"
    return title


def _feature_row_fields(
    item: ScoredFeature,
    digest: CommunityDigest,
    *,
    lang: str,
    with_marker: bool = False,
    also_in_label: str = "",
    this_project_only_label: str = "",
) -> dict[str, Any]:
    """Build the row fields shared by the inline and Jinja renderers.

    Both markdown code paths present the same per-item data (localized
    title/description, source list, score, category, and the optional
    new-feature marker); keeping the transformation here means the two
    renderers cannot drift apart.
    """
    record = item.record
    llm_info = digest.llm_importance.get(record.id, {})
    row: dict[str, Any] = {
        "record": record,
        "llm_info": llm_info,
        "title": _t_title(record, llm_info, lang),
        "desc": _t_desc(record, llm_info, lang),
        "sources": " + ".join([record.source, *record.related_projects]),
        "score": item.score.overall,
        "category": record.category.value,
        "url": record.url or "",
    }
    if with_marker:
        row["new_marker"] = (
            f"{also_in_label}: " + ", ".join(record.related_projects)
            if record.related_projects
            else this_project_only_label
        )
        row["feature_type"] = record.feature_type.value
    return row


def _render_inline_markdown(
    digest: CommunityDigest,
    *,
    comparison: HistoryComparison | None = None,
    lang: str = "zh",
) -> str:
    lines: list[str] = []
    period_label = get_text(f"period_{digest.period}", lang)
    period_word = get_text(f"period_{digest.period}", lang)
    date_range = _format_date_range(digest.period_start)
    lines.append(f"# {get_text('report_title', lang)} ({period_label})")
    lines.append("")
    lines.append(f"> {get_text('generated_at', lang)}: {digest.generated_at}")
    _sources = ", ".join(digest.sources_used) if digest.sources_used else get_text("none", lang)
    lines.append(
        f"> {get_text('coverage_label', lang)}: "
        f"{len(digest.sources_used)} {get_text('coverage_projects', lang)} · "
        f"{digest.stats.total_versions} {get_text('coverage_versions', lang)} · "
        f"{digest.stats.total_features} {get_text('coverage_features', lang)}"
    )
    # Show filtered count when non-zero
    if digest.stats.filtered_count > 0:
        lines.append(
            f"> {get_text('filtered_info', lang)}: {get_text('filtered_summary', lang, n=digest.stats.filtered_count)}"
        )
    lines.append("")

    # History comparison section (before summary when present)
    if comparison is not None:
        lines.append(_render_comparison_section(comparison, lang=lang))

    # ── Highlights (major features in prose) ──
    lines.append(f"## {get_text('section_highlights', lang)}")
    lines.append("")
    if digest.highlights:
        hl_desc_key = "highlights_desc_full" if digest.period == "full" else "highlights_desc"
        lines.append(f"> {get_text(hl_desc_key, lang, period_word=period_word, date_range=date_range)}")
        lines.append("")
        for idx, item in enumerate(digest.highlights, 1):
            fields = _feature_row_fields(item, digest, lang=lang)
            url_suffix = _make_url_suffix(fields["url"], lang)
            lines.append(
                f"{idx}. **{_escape(fields['title'])}** — {_escape(fields['desc'])} "
                f"({get_text('highlight_score', lang)} {fields['score']:.1f} · "
                f"{get_text('highlight_source', lang)}: {fields['sources']} · "
                f"{get_text('highlight_category', lang)}: {fields['category']})"
                f"{url_suffix}"
            )
            lines.append("")
    else:
        lines.append(get_text("no_activity", lang))
        lines.append("")

    # ── Detail table (all trending features, with new-feature markers) ──
    lines.append(f"## {get_text('section_detail_table', lang)}")
    lines.append("")
    if digest.trending:
        lines.append(
            f"| {get_text('th_feature', lang)} | {get_text('th_source', lang)} | "
            f"{get_text('th_score', lang)} | {get_text('th_category', lang)} | "
            f"{get_text('th_type', lang)} | {get_text('th_new', lang)} | "
            f"{get_text('th_desc', lang)} |"
        )
        lines.append("|------|------|:----:|------|:----:|------|------|")
        for item in digest.trending:
            fields = _feature_row_fields(
                item,
                digest,
                lang=lang,
                with_marker=True,
                also_in_label=get_text("also_in", lang),
                this_project_only_label=get_text("this_project_only", lang),
            )
            lines.append(
                f"| {_title_link(fields['record'], fields['llm_info'], lang)} | "
                f"{fields['sources']} | {fields['score']:.1f} | {fields['category']} | "
                f"{fields['feature_type']} | {fields['new_marker']} | "
                f"{_escape(fields['desc'])} |"
            )
    else:
        lines.append(get_text("none", lang))
    lines.append("")

    # ── Summary ──
    lines.append(f"## {get_text('section_summary', lang)}")
    lines.append("")
    lines.append(digest.summary.strip() or get_text("no_activity", lang))
    lines.append("")

    # ── Breaking changes ──
    lines.append(f"## {get_text('section_breaking', lang)}")
    lines.append("")
    if digest.breaking_changes:
        lines.append(
            f"| {get_text('th_project', lang)} | {get_text('th_feature', lang)} | {get_text('th_impact', lang)} |"
        )
        lines.append("|------|------|---------|")
        for record in digest.breaking_changes:
            lines.append(
                f"| {record.source} | {_title_link(record, digest.llm_importance.get(record.id, {}), lang)} | {_impact_for(record, lang)} |"
            )
    else:
        lines.append(get_text("none", lang))
    lines.append("")

    # ── Category distribution ──
    lines.append(f"## {get_text('section_distribution', lang)}")
    lines.append("")
    if digest.stats.by_root_category:
        lines.append(f"### {get_text('section_by_domain', lang)}")
        lines.append("")
        for parent_name, count in sorted(digest.stats.by_root_category.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {parent_name}: {count}")
        lines.append("")
        lines.append(f"### {get_text('section_by_subcategory', lang)}")
        lines.append("")
    if digest.stats.by_category:
        for category, count in sorted(digest.stats.by_category.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {category}: {count}")
    else:
        lines.append(get_text("none", lang))
    lines.append("")

    if digest.errors:
        lines.append(f"## {get_text('section_errors', lang)}")
        lines.append("")
        for err in digest.errors:
            lines.append(f"- {err}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _resolve_template_dir(template_dir: Path | str | None) -> Path | None:
    candidates: list[Path] = []
    if template_dir is not None:
        candidates.append(Path(template_dir))
    # Default location relative to this file.
    candidates.append(Path(__file__).parent / "templates")
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _render_jinja_markdown(
    digest: CommunityDigest,
    template_dir: Path,
    *,
    comparison: HistoryComparison | None = None,
    lang: str = "zh",
) -> str | None:
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _log.debug("jinja2 not available (%s); falling back to inline renderer", exc)
        return None

    period_to_template = {
        "weekly": "weekly_digest.md.j2",
        "monthly": "monthly_digest.md.j2",
        "full": "weekly_digest.md.j2",
    }
    template_name = period_to_template.get(digest.period)
    if template_name is None:
        return None
    template_path = template_dir / template_name
    if not template_path.exists():
        _log.debug("template %s missing; falling back to inline renderer", template_path)
        return None

    try:
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template(template_name)
    except Exception as exc:  # noqa: BLE001
        _log.warning("failed to load %s: %s; falling back", template_path, exc)
        return None

    labels = build_template_labels(lang, period=digest.period, period_start=digest.period_start)
    period_label = get_text(f"period_{digest.period}", lang)
    llm_assisted = digest.summary.endswith("(LLM-assisted)") or get_text("llm_assisted", lang) in digest.summary

    by_category = sorted(digest.stats.by_category.items(), key=lambda kv: -kv[1])
    by_root_category = sorted(digest.stats.by_root_category.items(), key=lambda kv: -kv[1])

    # Highlights rows — prefer LLM-generated text when available
    highlight_rows = [_feature_row_fields(item, digest, lang=lang) for item in digest.highlights]

    # Detail table rows (merged trending + new-feature markers)
    trending_rows = [
        _feature_row_fields(
            item,
            digest,
            lang=lang,
            with_marker=True,
            also_in_label=labels["also_in"],
            this_project_only_label=labels["this_project_only"],
        )
        for item in digest.trending
    ]

    breaking_rows = [
        {
            "source": record.source,
            "title": _t_title(record, digest.llm_importance.get(record.id, {}), lang),
            "impact": _impact_for(record, lang),
            "url": record.url or "",
        }
        for record in digest.breaking_changes
    ]

    comparison_data: dict[str, Any] | None = None
    if comparison is not None:
        cmp_new_rows = [
            {
                "title": rec.title,
                "desc": _short_desc(rec.description),
                "category": rec.category.value if rec.category else "unknown",
                "source": rec.source,
            }
            for rec in comparison.new_features
        ]
        cmp_disappeared_rows = [
            {
                "title": rec.title,
                "desc": _short_desc(rec.description),
                "category": rec.category.value if rec.category else "unknown",
                "source": rec.source,
            }
            for rec in comparison.disappeared_features
        ]
        comparison_data = {
            "has_comparison": True,
            "previous_period": comparison.previous_period,
            "previous_generated_at": comparison.previous_generated_at,
            "new_count": comparison.new_count,
            "disappeared_count": comparison.disappeared_count,
            "score_changed": comparison.score_changed,
            "new_features": cmp_new_rows,
            "disappeared_features": cmp_disappeared_rows,
            "show_new_features": comparison.new_count > 0,
            "show_disappeared": comparison.disappeared_count > 0,
            "show_score_changes": len(comparison.score_changed) > 0,
        }

    try:
        return template.render(
            title=get_text("report_title", lang),
            period_label=period_label,
            generated_at=digest.generated_at,
            sources_count=len(digest.sources_used),
            total_versions=digest.stats.total_versions,
            total_features=digest.stats.total_features,
            filtered_count=digest.stats.filtered_count,
            llm_assisted=llm_assisted,
            summary=digest.summary.strip() or get_text("no_activity", lang),
            highlights=highlight_rows,
            trending=trending_rows,
            breaking_changes=breaking_rows,
            by_category=by_category,
            by_root_category=by_root_category,
            top_projects=digest.stats.top_projects,
            errors=list(digest.errors),
            comparison=comparison_data or {},
            labels=labels,
        )
    except Exception as exc:  # noqa: BLE001 — StrictUndefined or bad var
        _log.warning("template render failed (%s): %s; falling back", template_name, exc)
        return None


def _build_stats(
    new_features: Iterable[FeatureRecord],
    versions_total: int,
    *,
    filtered_count: int = 0,
    major_count: int = 0,
    minor_count: int = 0,
) -> DigestStats:
    new_list = new_features.copy()
    by_cat: Counter[str] = Counter()
    by_parent: Counter[str] = Counter()
    project_counts: Counter[str] = Counter()
    for record in new_list:
        by_cat[record.category.value] += 1
        by_parent[get_root(record.category).value] += 1
        project_counts[record.source] += 1
        for related in record.related_projects:
            project_counts[related] += 1
    top_projects = project_counts.most_common(5)
    return DigestStats(
        total_versions=versions_total,
        total_features=len(new_list),
        filtered_count=filtered_count,
        major_count=major_count,
        minor_count=minor_count,
        by_category=dict(by_cat),
        by_root_category=dict(by_parent),
        top_projects=top_projects,
    )


def _split_breaking(features: list[FeatureRecord]) -> list[FeatureRecord]:
    return [r for r in features if r.feature_type == FeatureType.BREAKING]


def _summarise(features: list[FeatureRecord], lang: str = "zh") -> str:
    if not features:
        return get_text("no_new_features_summary", lang)
    counts = Counter(r.category.value for r in features)
    top_categories = [name for name, _ in counts.most_common(3)]
    bullet = "、".join(top_categories) if top_categories else get_text("none", lang)
    return get_text("summary_template", lang, n=len(features), cats=bullet)


def find_previous_digest_path(
    output_dir: Path | str,
    period: str,
    current_stem: str,
) -> Path | None:
    """Return the most recent previous JSON digest in *output_dir*, or None."""
    out = Path(output_dir)
    if not out.is_dir():
        return None
    pattern = f"community-digest-{period}-????????T??????Z.json"
    candidates = sorted(out.glob(pattern), reverse=True)
    for path in candidates:
        if path.stem != current_stem:
            return path
    return None


def _load_digest_from_json(path: Path) -> CommunityDigest | None:
    """Rehydrate a CommunityDigest from a JSON file, or return None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "period" not in data:
        return None
    try:
        new_features = [FeatureRecord.from_dict(r) for r in data.get("new_features", []) or []]
        trending_raw = data.get("trending", []) or []
        trending = []
        for item in trending_raw:
            record = FeatureRecord.from_dict(item.get("record", {}))
            s = item.get("score", {})
            score_dims = s.get("dimensions", s)
            score = FeatureScore(
                record_id=s.get("record_id", record.id),
                overall=float(s.get("overall", 0)),
                popularity=float(score_dims.get("popularity", 0)),
                maturity=float(score_dims.get("maturity", 0)),
                adaptation_cost=float(score_dims.get("adaptation_cost", 0)),
                strategic_value=float(score_dims.get("strategic_value", 0)),
                architecture_fit=float(score_dims.get("architecture_fit", 0)),
            )
            trending.append(ScoredFeature(record=record, score=score))
        breaking = [FeatureRecord.from_dict(r) for r in data.get("breaking_changes", []) or []]
        stats_raw = data.get("stats", {})
        stats = DigestStats(
            total_versions=int(stats_raw.get("total_versions", 0)),
            total_features=int(stats_raw.get("total_features", 0)),
            filtered_count=int(stats_raw.get("filtered_count", 0)),
            major_count=int(stats_raw.get("major_count", 0)),
            minor_count=int(stats_raw.get("minor_count", 0)),
            by_category=stats_raw.get("by_category", {}),
            by_root_category=stats_raw.get("by_root_category", {}),
            top_projects=[tuple(item) for item in stats_raw.get("top_projects", [])],
        )
        highlights_raw = data.get("highlights", []) or []
        highlights = []
        for item in highlights_raw:
            record = FeatureRecord.from_dict(item.get("record", {}))
            s = item.get("score", {})
            score_dims = s.get("dimensions", s)
            score = FeatureScore(
                record_id=s.get("record_id", record.id),
                overall=float(s.get("overall", 0)),
                popularity=float(score_dims.get("popularity", 0)),
                maturity=float(score_dims.get("maturity", 0)),
                adaptation_cost=float(score_dims.get("adaptation_cost", 0)),
                strategic_value=float(score_dims.get("strategic_value", 0)),
                architecture_fit=float(score_dims.get("architecture_fit", 0)),
            )
            highlights.append(ScoredFeature(record=record, score=score))
        llm_importance_raw = data.get("llm_importance", {}) or {}
        return CommunityDigest(
            period=str(data["period"]),
            generated_at=str(data.get("generated_at") or ""),
            period_start=str(data.get("period_start") or ""),
            summary=str(data.get("summary", "")),
            new_features=new_features,
            trending=trending,
            highlights=highlights,
            llm_importance=llm_importance_raw,
            breaking_changes=breaking,
            stats=stats,
            sources_used=data.get("sources_used", []),
            errors=data.get("errors", []),
        )
    except Exception:  # noqa: BLE001
        return None


def compare_digests(
    previous: CommunityDigest,
    current: CommunityDigest,
    *,
    score_delta_threshold: float = 5.0,
) -> HistoryComparison:
    """Produce a delta between previous and current digests by matching feature IDs."""
    prev_ids: dict[str, FeatureRecord] = {}
    prev_scores: dict[str, float] = {}
    for f in previous.new_features:
        prev_ids[f.id] = f
    for sf in previous.trending:
        prev_ids[sf.record.id] = sf.record
        prev_scores[sf.record.id] = sf.score.overall

    curr_ids: dict[str, FeatureRecord] = {}
    curr_scores: dict[str, float] = {}
    for f in current.new_features:
        curr_ids[f.id] = f
    for sf in current.trending:
        curr_scores[sf.record.id] = sf.score.overall
        if sf.record.id not in curr_ids:
            curr_ids[sf.record.id] = sf.record

    # pylint: disable=consider-using-dict-items
    new = [curr_ids[fid] for fid in curr_ids if fid not in prev_ids]
    disappeared = [prev_ids[fid] for fid in prev_ids if fid not in curr_ids]

    score_changed: list[dict[str, Any]] = []
    for fid in set(prev_scores) & set(curr_scores):
        delta = curr_scores[fid] - prev_scores[fid]
        if abs(delta) >= score_delta_threshold:
            rec = curr_ids.get(fid) or prev_ids.get(fid)
            score_changed.append(
                {
                    "id": fid,
                    "title": rec.title if rec else fid,
                    "old_score": round(prev_scores[fid], 1),
                    "new_score": round(curr_scores[fid], 1),
                    "delta": round(delta, 1),
                }
            )

    return HistoryComparison(
        previous_period=previous.period,
        previous_generated_at=previous.generated_at,
        previous_stem="",
        new_count=len(new),
        disappeared_count=len(disappeared),
        score_changed=score_changed,
        new_features=new,
        disappeared_features=disappeared,
    )


def _render_comparison_section(comparison: HistoryComparison, lang: str = "zh") -> str:
    lines: list[str] = []
    prev_label = get_text(f"period_{comparison.previous_period}", lang)
    lines.append(f"## {get_text('section_comparison', lang)}")
    lines.append("")
    lines.append(f"> {get_text('comparison_baseline', lang)}: {comparison.previous_generated_at} ({prev_label})")
    lines.append("")

    # New features
    lines.append(f"### {get_text('comparison_new', lang, n=comparison.new_count)}")
    lines.append("")
    if comparison.new_features:
        for rec in comparison.new_features:
            cat = rec.category.value if rec.category else "unknown"
            lines.append(
                f"- **{_escape(rec.title)}** — "
                f"{_escape(_short_desc(rec.description))} "
                f"({cat}, {get_text('source_from', lang)}: {rec.source})"
            )
    else:
        lines.append(get_text("no_new_features", lang))
    lines.append("")

    # Disappeared features
    lines.append(f"### {get_text('comparison_disappeared', lang, n=comparison.disappeared_count)}")
    lines.append("")
    if comparison.disappeared_features:
        for rec in comparison.disappeared_features:
            cat = rec.category.value if rec.category else "unknown"
            lines.append(
                f"- ~~**{_escape(rec.title)}**~~ — "
                f"{_escape(_short_desc(rec.description))} "
                f"({cat}, {get_text('source_from', lang)}: {rec.source})"
            )
    else:
        lines.append(get_text("no_disappeared", lang))
    lines.append("")

    # Score changes
    lines.append(f"### {get_text('comparison_score', lang, n=len(comparison.score_changed))}")
    lines.append("")
    if comparison.score_changed:
        lines.append(
            f"| {get_text('th_feature', lang)} | {get_text('th_old_score', lang)} | "
            f"{get_text('th_new_score', lang)} | {get_text('th_change', lang)} |"
        )
        lines.append("|------|:------:|:------:|:----:|")
        for item in comparison.score_changed:
            direction = "↑" if item["delta"] > 0 else "↓"
            lines.append(
                f"| {_escape(item['title'])} "
                f"| {item['old_score']:.1f} "
                f"| {item['new_score']:.1f} "
                f"| {direction} {abs(item['delta']):.1f} |"
            )
    else:
        lines.append(get_text("no_score_changes", lang))
    lines.append("")

    return "\n".join(lines)


@dataclass
class DigestWriteResult:
    """Paths produced by :meth:`CommunityReporter.write`."""

    markdown_path: Path
    json_path: Path
    proposals_path: Path | None = None


class CommunityReporter:
    def __init__(
        self,
        config: RadarConfig | None = None,
        *,
        template_dir: Path | str | None = None,
    ) -> None:
        self.config = config or RadarConfig()
        self._explicit_template_dir = Path(template_dir) if template_dir is not None else None

    def build_digest(
        self,
        *,
        period: str,
        features: list[FeatureRecord],
        scored: list[ScoredFeature],
        sources_used: Iterable[str],
        errors: Iterable[str] | None = None,
        summary: str | None = None,
        generated_at: str | None = None,
        versions_total: int = 0,
        filtered_count: int = 0,
        lang: str = "zh",
        llm_importance: dict[str, dict[str, str]] | None = None,
    ) -> CommunityDigest:
        ranked = sorted(scored, key=lambda s: s.score.overall, reverse=True)
        max_features = max(1, int(self.config.max_features_per_report))
        trending = ranked[:max_features]
        new_features = [s.record for s in trending]

        hl_categories_lower = {c.lower() for c in self.config.highlight_categories}
        hl_min_score = float(self.config.highlight_min_score)
        llm_imp = llm_importance or {}
        highlights: list[ScoredFeature] = []
        highlight_ids: set[str] = set()
        rest: list[ScoredFeature] = []
        for item in trending:
            rid = item.record.id
            if rid in llm_imp and llm_imp[rid].get("level") == "MAJOR":
                highlights.append(item)
                highlight_ids.add(rid)
            elif (
                item.score.overall >= hl_min_score
                and item.record.category.value.lower() in hl_categories_lower
                and rid not in llm_imp  # only rule-classify when LLM didn't classify
            ):
                highlights.append(item)
            else:
                rest.append(item)

        if llm_imp:
            scored_by_id: dict[str, ScoredFeature] = {s.record.id: s for s in scored}
            llm_major_outside = [
                scored_by_id[rid]
                for rid, v in llm_imp.items()
                if v.get("level") == "MAJOR"
                and rid not in highlight_ids
                and rid in scored_by_id
                and scored_by_id[rid].score.overall >= hl_min_score
            ]
            llm_major_outside.sort(key=lambda s: s.score.overall, reverse=True)
            highlights.extend(llm_major_outside)

        grouped: dict[tuple[str, str], list[ScoredFeature]] = {}
        for sf in highlights:
            key = (sf.record.source, sf.record.url)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(sf)

        merged_highlights: list[ScoredFeature] = []
        for key, group in grouped.items():
            if len(group) == 1:
                merged_highlights.append(group[0])
            else:
                group.sort(key=lambda s: s.score.overall, reverse=True)
                best = group[0]
                titles: list[str] = []
                titles_zh: list[str] = []
                highlight_texts: list[str] = []
                for sf in group:
                    info = llm_imp.get(sf.record.id, {})
                    titles.append(sf.record.title)
                    titles_zh.append(info.get("title_zh") or sf.record.title)
                    highlight_texts.append(info.get("highlight") or sf.record.description[:120])
                combined_title = "；".join(titles)
                combined_title_zh = "；".join(titles_zh)
                combined_desc = "；".join(highlight_texts)

                import hashlib as _hl

                merged_id = "merged:" + _hl.sha256(f"{key[0]}|{key[1]}".encode()).hexdigest()[:12]

                merged_record = FeatureRecord(
                    id=merged_id,
                    source=best.record.source,
                    title=combined_title,
                    description=combined_desc,
                    category=best.record.category,
                    feature_type=best.record.feature_type,
                    url=best.record.url,
                    related_projects=list(best.record.related_projects),
                    tags=list(best.record.tags),
                )
                merged_feature = ScoredFeature(
                    record=merged_record,
                    score=FeatureScore(
                        record_id=merged_id,
                        overall=best.score.overall,
                        popularity=best.score.popularity,
                        maturity=best.score.maturity,
                        adaptation_cost=best.score.adaptation_cost,
                        strategic_value=best.score.strategic_value,
                        architecture_fit=best.score.architecture_fit,
                    ),
                )
                merged_highlights.append(merged_feature)
                llm_imp[merged_id] = {
                    "level": "MAJOR",
                    "highlight": combined_desc,
                    "title_zh": combined_title_zh,
                    "desc_zh": combined_desc,
                }

        highlights = merged_highlights

        rest.sort(key=lambda s: s.score.overall, reverse=True)
        while len(highlights) < 5 and rest:
            highlights.append(rest.pop(0))
        highlights = highlights[:15]
        highlight_ids = {s.record.id for s in highlights}

        breaking = _split_breaking(features)
        stats = _build_stats(
            features,
            versions_total,
            filtered_count=filtered_count,
            major_count=len(highlights),
            minor_count=max(0, len(trending) - len(highlights)),
        )
        digest = CommunityDigest(
            period=period,
            generated_at=generated_at or utc_now_iso(),
            summary=summary if summary is not None else _summarise(features, lang),
            new_features=new_features,
            trending=trending,
            highlights=highlights,
            llm_importance=llm_imp,
            breaking_changes=breaking,
            stats=stats,
            sources_used=list(sources_used),
            errors=list(errors or []),
        )
        return digest

    def write(
        self,
        digest: CommunityDigest,
        output_dir: Path | str,
        *,
        write_proposals: bool = True,
        compare: bool = False,
    ) -> DigestWriteResult:
        """Write digest as Markdown + JSON + optional proposals to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        timestamp = _file_timestamp(digest.generated_at)
        stem = f"community-digest-{digest.period}-{timestamp}"
        md_path = out / f"{stem}.md"
        json_path = out / f"{stem}.json"

        # History comparison
        comparison: HistoryComparison | None = None
        if compare:
            prev_path = find_previous_digest_path(out, digest.period, stem)
            if prev_path is not None:
                prev_digest = _load_digest_from_json(prev_path)
                if prev_digest is not None:
                    comparison = compare_digests(prev_digest, digest)
                    comparison.previous_stem = prev_path.stem

        # Prefer Jinja2 templates (Phase 3); gracefully fall back to the
        # inline renderer when templates or jinja2 itself are missing.
        template_dir = _resolve_template_dir(self._explicit_template_dir)
        rendered: str | None = None
        lang = self.config.language
        if template_dir is not None:
            rendered = _render_jinja_markdown(digest, template_dir, comparison=comparison, lang=lang)
        if rendered is None:
            rendered = _render_inline_markdown(digest, comparison=comparison, lang=lang)
        md_path.write_text(rendered, encoding="utf-8")

        json_path.write_text(
            json.dumps(digest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        proposals_path: Path | None = None
        if write_proposals:
            proposals_path = out / f"{stem}.proposals.json"
            proposals_path.write_text(
                json.dumps(render_proposals(digest), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        _log.info(
            "community digest written: %s (%d features, %d trending)",
            md_path,
            digest.stats.total_features,
            len(digest.trending),
        )
        return DigestWriteResult(
            markdown_path=md_path,
            json_path=json_path,
            proposals_path=proposals_path,
        )


def _file_timestamp(iso: str) -> str:
    """Convert ``2026-06-29T08:00:00Z`` → ``20260629T080000Z`` for filenames."""
    try:
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return iso.replace(":", "").replace("-", "")
    return dt.strftime("%Y%m%dT%H%M%SZ")


def copy_to_persistent(
    digest_path: Path,
    *,
    root: Path | None = None,
) -> Path | None:
    """Copy digest into ~/.clawcodex/reports/community-radar, return dest path or None."""
    try:
        target_root = root or (Path.home() / ".clawcodex" / "reports" / "community-radar")
        target_root.mkdir(parents=True, exist_ok=True)
        target = target_root / digest_path.name
        shutil.copy2(digest_path, target)
        return target
    except Exception as exc:  # noqa: BLE001
        _log.warning("copy_to_persistent failed for %s: %s", digest_path, exc)
        return None


def render_proposals(digest: CommunityDigest) -> dict[str, Any]:
    """Build the proposals.json payload from digest.trending with scores and candidate actions."""
    proposals: list[dict[str, Any]] = []
    for item in digest.trending:
        record = item.record
        score = item.score
        candidate_action = _candidate_action(record, score)
        proposals.append(
            {
                "id": record.id,
                "title": record.title,
                "description": record.description,
                "category": record.category.value,
                "feature_type": record.feature_type.value,
                "source_projects": [record.source, *record.related_projects],
                "score": {
                    "overall": score.overall,
                    "popularity": score.popularity,
                    "maturity": score.maturity,
                    "adaptation_cost": score.adaptation_cost,
                    "strategic_value": score.strategic_value,
                    "architecture_fit": score.architecture_fit,
                },
                "candidate_action": candidate_action,
                "tags": list(record.tags),
                "released_at": record.released_at,
                "url": record.url,
            }
        )
    return {
        "schema_version": "1.0",
        "generated_at": digest.generated_at,
        "period": digest.period,
        "proposals": proposals,
    }


def _candidate_action(record: FeatureRecord, score: FeatureScore) -> str:
    """Pick one of ``adopt`` / ``observe`` / ``skip``."""
    if record.feature_type == FeatureType.BREAKING and score.strategic_value < 40:
        return "skip"
    if score.strategic_value >= 60 and score.architecture_fit >= 60:
        return "adopt"
    if score.overall >= 60:
        return "observe"
    return "skip"


__all__ = [
    "CommunityReporter",
    "DigestWriteResult",
    "copy_to_persistent",
    "render_proposals",
]
