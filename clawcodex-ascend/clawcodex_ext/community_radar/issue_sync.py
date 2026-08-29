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
"""GitCode / GitHub / Gitee issue synchronisation for Community Radar.

Pipelines scan results into actionable issues on the user's code-hosting
platform.  Supports two paths:

*Path A — automatic*: after a scan, create the top-N MAJOR-feature issues.
*Path B — manual*: the user picks a single feature via CLI or interactive
               table and creates a one-off issue.

Deduplication is a three-layer defence:

L1  Local cache (``issue_sync_cache.json``) — fast hash lookup.
L2  Remote cross-check — fetch open ``community-radar`` issues, parse the
    invisible ``<!-- community-radar-id: ... -->`` marker inside each body.
L3  Manual-mode confirmation gate — when L2 finds a match while the user
    explicitly requested a specific feature, warn and ask before proceeding.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RadarConfig
from .issue_platforms import (
    IssueClient,
    ResolvedTarget,
    resolve_target,
)
from .models import CommunityDigest, ScoredFeature

_log = logging.getLogger(__name__)

# Regex to extract the hidden community-radar-id from an issue body
_ID_MARKER_RE = re.compile(r"<!--\s*community-radar-id:\s*(\S+)\s*-->")


# ---------------------------------------------------------------------------
# IssueSyncCache (L1)
# ---------------------------------------------------------------------------


@dataclass
class IssueSyncCache:
    """Persistent ``feature_id → issue metadata`` mapping, keyed by repo."""

    repos: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> IssueSyncCache:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _log.warning("issue sync cache corrupted; starting fresh")
            return cls()
        repos = data.get("repos", {})
        if not isinstance(repos, dict):
            return cls()
        return cls(repos=repos)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": "2", "repos": self.repos}
        # Atomic write via temp file
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _repo_key(self, target: ResolvedTarget) -> str:
        return f"{target.platform.name}/{target.owner}/{target.repo}"

    def get(self, target: ResolvedTarget, feature_id: str) -> dict[str, Any] | None:
        """Return cached metadata for *feature_id* in *target* repo, or None."""
        repo_key = self._repo_key(target)
        return self.repos.get(repo_key, {}).get(feature_id)

    def put(self, target: ResolvedTarget, feature_id: str, meta: dict[str, Any]) -> None:
        """Store metadata for *feature_id*."""
        repo_key = self._repo_key(target)
        self.repos.setdefault(repo_key, {})[feature_id] = meta

    def exists(self, target: ResolvedTarget, feature_id: str) -> bool:
        """Check if *feature_id* already has an open issue in *target*."""
        entry = self.get(target, feature_id)
        if entry is None:
            return False
        return entry.get("state") == "open"

    def sync_from_remote(self, target: ResolvedTarget, remote_map: dict[str, dict[str, Any]]) -> int:
        """Merge a fresh remote mapping into the cache.  Returns count of new entries."""
        repo_key = self._repo_key(target)
        repo = self.repos.setdefault(repo_key, {})
        added = 0
        for fid, meta in remote_map.items():
            if fid not in repo:
                repo[fid] = meta
                added += 1
            else:
                # Update state if changed
                repo[fid].update(meta)
        return added


# ---------------------------------------------------------------------------
# IssueSyncResult
# ---------------------------------------------------------------------------


@dataclass
class IssueSyncResult:
    """Outcome of an issue-sync operation."""

    created: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    warned: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def _select_candidates(
    digest: CommunityDigest,
    llm_importance: dict[str, dict[str, str]],
    max_n: int,
) -> list[ScoredFeature]:
    """Select the top-N MAJOR features from *digest* highlights.

    Filters to only LLM-classified MAJOR features, then sorts by score
    descending and returns at most *max_n*.
    """
    candidates: list[ScoredFeature] = []
    for sf in digest.highlights:
        info = llm_importance.get(sf.record.id, {})
        if info.get("level") == "MAJOR":
            candidates.append(sf)
    # Filter out features with None scores (defensive against malformed data)
    candidates = [c for c in candidates if c.score.overall is not None]
    # Sort by overall score descending
    candidates.sort(key=lambda s: s.score.overall, reverse=True)
    return candidates[:max_n] if max_n > 0 else candidates


# ---------------------------------------------------------------------------
# Issue body rendering
# ---------------------------------------------------------------------------


def _bar_chart(value: float, width: int = 20) -> str:
    """Render a single ASCII bar for a 0-100 score value."""
    if not math.isfinite(value) or not math.isfinite(width):
        return "░" * max(0, width)
    if width <= 0:
        return ""
    filled = max(0, min(width, round(value / 100.0 * width)))
    return "█" * filled + "░" * (width - filled)


def _build_issue_body(
    sf: ScoredFeature,
    digest: CommunityDigest,
    llm_info: dict[str, str],
    *,
    duplicate_warning: str = "",
) -> str:
    """Render the issue body Markdown from template v1."""
    record = sf.record
    score = sf.score
    highlight = llm_info.get("highlight", record.description or "")
    title_zh = llm_info.get("title_zh", record.title)

    source_url = record.url or ""
    repo_url = f"https://github.com/{record.source}"
    period_label = digest.period or "weekly"
    period_range = f"{digest.period_start or '?'} ~ {digest.generated_at or '?'}"

    bars = (
        f"流行度     {_bar_chart(score.popularity)}  {score.popularity:.0f}\n"
        f"成熟度     {_bar_chart(score.maturity)}  {score.maturity:.0f}\n"
        f"采纳成本   {_bar_chart(score.adaptation_cost)}  {score.adaptation_cost:.0f}\n"
        f"战略价值   {_bar_chart(score.strategic_value)}  {score.strategic_value:.0f}\n"
        f"架构匹配   {_bar_chart(score.architecture_fit)}  {score.architecture_fit:.0f}"
    )

    related = "\n".join(f"- [ ] {p}" for p in (record.related_projects or [])[:10]) or "(无关联项目)"

    dup_section = ""
    if duplicate_warning:
        dup_section = f"\n> ⚠️ 重复提醒：{duplicate_warning}\n"

    return (
        f"## 🎯 {title_zh or record.title}\n"
        f"\n"
        f"> **一句话总结**：{highlight}\n"
        f"\n"
        f"| 维度 | 详情 |\n"
        f"|:---|:---|\n"
        f"| **来源项目** | [{record.source}]({repo_url}) |\n"
        f"| **原始链接** | [{record.title}]({source_url}) |\n"
        f"| **分类** | `{record.category.value}` · `{record.feature_type.value}` |\n"
        f"| **综合评分** | **{score.overall:.0f}/100** |\n"
        f"| **LLM 判定** | 🔴 MAJOR |\n"
        f"| **扫描周期** | {period_label}（{period_range}）|\n"
        f"\n"
        f"---\n"
        f"\n"
        f"### 📊 评分明细\n"
        f"\n"
        f"| 维度 | 得分 | 权重 | 说明 |\n"
        f"|:---|:---|:---|:---|\n"
        f"| 流行度 Popularity | {score.popularity:.0f}/100 | 15% | stars / forks / watchers |\n"
        f"| 成熟度 Maturity | {score.maturity:.0f}/100 | 20% | 版本数 / 社区活跃度 |\n"
        f"| 采纳成本 Adaptation Cost | {score.adaptation_cost:.0f}/100 | 25% | 越低越容易引入 |\n"
        f"| 战略价值 Strategic Value | {score.strategic_value:.0f}/100 | 25% | 与 ClawCodex 路线匹配度 |\n"
        f"| 架构匹配 Architecture Fit | {score.architecture_fit:.0f}/100 | 15% | Layer 0-2 架构约束 |\n"
        f"\n"
        f"```\n"
        f"{bars}\n"
        f"```\n"
        f"\n"
        f"---\n"
        f"\n"
        f"### 🔍 详细分析\n"
        f"\n"
        f"{highlight}\n"
        f"\n"
        f"### 📝 特性描述\n"
        f"\n"
        f"{record.description or '(无描述)'}\n"
        f"\n"
        f"### 🔗 关联项目\n"
        f"\n"
        f"{related}\n"
        f"\n"
        f"---\n"
        f"\n"
        f"### ✅ 决策清单\n"
        f"\n"
        f"> 由 ClawCodex 团队在评估后勾选\n"
        f"\n"
        f"- [ ] **已理解**：团队已阅读并理解该特性\n"
        f"- [ ] **可行性评估**：确认可在 ClawCodex 架构中实现\n"
        f"- [ ] **优先级判定**：P0 立即跟进 / P1 下个迭代 / P2 观望 / P3 暂不采纳\n"
        f"- [ ] **指派负责人**：\n"
        f"- [ ] **关联 Roadmap**：对应 FEATURE_PLAN.md 中的哪个章节？\n"
        f"\n"
        f"---\n"
        f"\n"
        f"{dup_section}"
        f"> 🤖 由 Community Radar 自动生成 · Feature ID: `{record.id}`\n"
        f"> 如对该 issue 的创建规则有疑问，请查看 `~/.clawcodex/community-radar/config.yaml`\n"
        f"<!-- community-radar-id: {record.id} -->\n"
    )


# ---------------------------------------------------------------------------
# L2: Remote dedup
# ---------------------------------------------------------------------------


def _fetch_remote_feature_map(client: IssueClient) -> dict[str, dict[str, Any]]:
    """Scan the remote repo's issues and build a ``feature_id → issue_metadata``
    map from hidden ``<!-- community-radar-id: ... -->`` body markers.

    Does NOT filter by label — some issues may have been created before the
    label-format fix or via a different code path, so we scan **all** issues
    (open + closed) for the marker.  Returns an empty dict on any error
    (the caller logs a warning).
    """
    try:
        issues = client.list_issues(state="all")
    except Exception as exc:  # noqa: BLE001
        _log.warning("L2 remote dedup: failed to list issues: %s", exc)
        return {}

    mapping: dict[str, dict[str, Any]] = {}
    for issue in issues:
        body_text = issue.get("body") or ""
        m = _ID_MARKER_RE.search(body_text)
        if m:
            fid = m.group(1)
            state = issue.get("state", "open")
            # When multiple issues share the same feature_id, keep the
            # entry with "open" state if one exists — otherwise the dedup
            # loop would see "closed" (from the oldest issue) and in
            # "retry" mode re-create yet another duplicate.
            if fid not in mapping or (mapping[fid].get("state") != "open" and state == "open"):
                mapping[fid] = {
                    "feature_id": fid,
                    "issue_number": issue.get("number") or issue.get("iid"),
                    "issue_url": issue.get("html_url", ""),
                    "state": state,
                    "feature_title": issue.get("title", ""),
                    "created_at": issue.get("created_at", ""),
                }
    return mapping


# ---------------------------------------------------------------------------
# Auto-sync: sync top-N features after scan
# ---------------------------------------------------------------------------


def _prompt_closed_issue(
    feature_id: str,
    feature_title: str,
    issue_number: str | int,
    issue_url: str,
) -> bool:
    """Ask the user whether to re-create an issue for a feature whose
    previous issue was closed.

    Returns ``True`` if the user wants to re-create, ``False`` otherwise.
    When stdin is not a TTY (cron / pipe), logs a warning and returns
    ``False`` — the safe default that avoids unwanted dupes.
    """
    if not sys.stdin.isatty():
        _log.warning(
            "closed-issue mode=ask but stdin is not a TTY; falling back to skip for feature %s (was #%s)",
            feature_id,
            issue_number,
        )
        return False
    short_title = feature_title[:80] if len(feature_title) > 80 else feature_title
    print(
        f"\n⚠️  \"{short_title}\" was previously tracked in #{issue_number} (closed).\n   {issue_url}",
    )
    while True:
        try:
            answer = input("   Re-create issue? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("   Please answer y or n.")


def sync_features_to_issues(
    *,
    digest: CommunityDigest,
    llm_importance: dict[str, dict[str, str]],
    config: RadarConfig,
    max_n: int | None = None,
    target: ResolvedTarget | None = None,
    cli_repo: str | None = None,
    cli_platform: str | None = None,
    cache_dir: str = ".cache/community-radar",
    closed_issue_mode: str | None = None,
) -> IssueSyncResult:
    """Auto-sync top-N MAJOR features to platform issues (Path A).

    Called from the pipeline after a scan completes.  Respects the
    three-layer dedup defence.
    """
    result = IssueSyncResult()

    if target is None:
        target = resolve_target(
            config_target_repo=config.target_repo,
            config_api_token=config.api_token,
            cli_repo=cli_repo,
            cli_platform=cli_platform,
        )
    if target is None:
        result.errors.append("无法确定目标仓库 — 请通过 --repo 或配置文件或 git remote 指定")
        return result
    if not target.api_token:
        _platform_name = target.platform.name.upper()
        env_names = ", ".join(target.platform.token_env_vars)
        result.errors.append(f"未提供 {target.platform.name} API token — 请设置 {env_names} 环境变量")
        return result

    n = max_n if max_n is not None else config.sync_issues_max_per_scan
    mode = closed_issue_mode if closed_issue_mode is not None else config.sync_issues_closed_issue_mode
    # Select ALL MAJOR candidates (not capped at n) so we can backfill
    # when earlier candidates are skipped due to cache hits.
    all_candidates = _select_candidates(digest, llm_importance, 0)
    candidates = all_candidates  # full list, iteration handles the cap

    if not all_candidates:
        result.skipped.append({"reason": "no MAJOR candidates found in this scan"})
        return result

    # L1 — load cache
    cache_path = Path(cache_dir) / "issue_sync_cache.json"
    cache = IssueSyncCache.load(cache_path)

    # L2 — fetch remote map
    client = IssueClient(target)
    try:
        try:
            remote_map = _fetch_remote_feature_map(client)
            cache.sync_from_remote(target, remote_map)
        except Exception as exc:  # noqa: BLE001
            _log.warning("L2 remote dedup failed (non-fatal): %s", exc)

        base_labels = list(config.sync_issues_labels)
        created_count = 0

        for sf in candidates:
            if created_count >= n:
                break

            fid = sf.record.id
            feature_title = sf.record.title

            # L1 / L2 dedup — always skip features with an open issue.
            # For closed issues, behaviour depends on ``closed_issue_mode``.
            existing = cache.get(target, fid)
            if existing and existing.get("state") == "open":
                result.skipped.append(
                    {
                        "feature_id": fid,
                        "feature_title": feature_title,
                        "reason": "already open",
                        "existing_issue": existing.get("issue_url"),
                    }
                )
                _log.info("skip: feature %s already has open issue %s", fid, existing.get("issue_url"))
                continue
            if existing and existing.get("state") == "closed":
                if mode == "skip":
                    result.skipped.append(
                        {
                            "feature_id": fid,
                            "feature_title": feature_title,
                            "reason": "closed issue exists (mode=skip)",
                            "existing_issue": existing.get("issue_url"),
                        }
                    )
                    _log.info("skip: feature %s has closed issue %s (mode=skip)", fid, existing.get("issue_url"))
                    continue
                if mode == "ask":
                    should_create = _prompt_closed_issue(
                        fid,
                        feature_title,
                        existing.get("issue_number", "?"),
                        existing.get("issue_url", ""),
                    )
                    if not should_create:
                        result.skipped.append(
                            {
                                "feature_id": fid,
                                "feature_title": feature_title,
                                "reason": "user declined re-create for closed issue",
                                "existing_issue": existing.get("issue_url"),
                            }
                        )
                        continue
                # mode == "retry": fall through and create a fresh issue

            llm_info = llm_importance.get(fid, {})
            body = _build_issue_body(sf, digest, llm_info)

            # Create the issue with base labels + category label
            issue_labels = base_labels + [f"category:{sf.record.category.value}"]
            resp = client.create_issue(title=feature_title, body=body, labels=issue_labels)
            if resp is None:
                result.errors.append(f"failed to create issue for {fid}: {feature_title}")
                continue

            # Add labels in a follow-up call when the platform skips them in the
            # create request (e.g. GitCode WAF blocks labels in form-encoded POSTs).
            if target.platform.create_issue_skip_labels and not client.add_labels_to_issue(
                resp.get("number") or resp.get("iid"),
                issue_labels,
            ):
                _log.warning(
                    "failed to add labels to issue %s for feature %s",
                    resp.get("number", "?"),
                    fid,
                )

            created_count += 1

            issue_number = resp.get("number") or resp.get("iid", "?")
            issue_url = resp.get("html_url", f"{target.web_url}/issues/{issue_number}")
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            cache.put(
                target,
                fid,
                {
                    "feature_id": fid,
                    "issue_number": issue_number,
                    "issue_url": issue_url,
                    "state": "open",
                    "feature_title": feature_title,
                    "created_at": now,
                },
            )
            result.created.append(
                {
                    "feature_id": fid,
                    "issue_number": issue_number,
                    "issue_url": issue_url,
                    "feature_title": feature_title,
                }
            )

        # Persist cache
        try:
            cache.save(cache_path)
        except OSError as exc:
            _log.warning("failed to save issue sync cache: %s", exc)

        return result
    finally:
        client.close()


# ---------------------------------------------------------------------------
# L3: duplicate override confirmation (Path B)
# ---------------------------------------------------------------------------


def _confirm_duplicate_override(
    feature_id: str,
    feature_title: str,
    existing: dict[str, Any],
) -> bool:
    """Print a warning and ask the user whether to proceed.

    Returns ``True`` if the user types ``y``.
    """
    print()
    print("⚠️  该特性已在目标仓库中存在关联 issue：")
    print(f"    Feature: \"{feature_title}\" ({feature_id})")
    print(f"    已有 Issue: {existing.get('issue_url', '?')}")
    print(f"    状态: {existing.get('state', '?')}")
    print(f"    创建于: {existing.get('created_at', '?')}")
    print()
    print("是否仍要继续创建新 issue？这会导致同一特性有多个 issue。")
    try:
        answer = input("(y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# Path B — manual: sync a single feature
# ---------------------------------------------------------------------------


def sync_single_feature(
    *,
    feature_id: str,
    config: RadarConfig,
    digest: CommunityDigest | None = None,
    llm_importance: dict[str, dict[str, str]] | None = None,
    target: ResolvedTarget | None = None,
    cache_dir: str = ".cache/community-radar",
    closed_issue_mode: str | None = None,
) -> IssueSyncResult:
    """Sync a single feature to an issue (Path B — manual mode).

    Supports the L1 → L2 → L3 dedup flow, including interactive
    confirmation when a duplicate is found remotely.
    """
    result = IssueSyncResult()

    if target is None:
        target = resolve_target(
            config_target_repo=config.target_repo,
            config_api_token=config.api_token,
        )
    if target is None:
        result.errors.append("无法确定目标仓库 — 请通过 --repo 或配置文件或 git remote 指定")
        return result
    if not target.api_token:
        env_names = ", ".join(target.platform.token_env_vars)
        result.errors.append(f"未提供 {target.platform.name} API token — 请设置 {env_names} 环境变量")
        return result

    mode = closed_issue_mode if closed_issue_mode is not None else config.sync_issues_closed_issue_mode

    # Look up feature from digest (if provided)
    sf: ScoredFeature | None = None
    if digest is not None:
        for s in digest.highlights:
            if s.record.id == feature_id:
                sf = s
                break
        if sf is None:
            for s in digest.trending:
                if s.record.id == feature_id:
                    sf = s
                    break

    if sf is None:
        result.errors.append(f"未找到 feature_id={feature_id} 的特性记录")
        return result

    if llm_importance is None:
        llm_importance = {}

    # L1 — load cache
    cache_path = Path(cache_dir) / "issue_sync_cache.json"
    cache = IssueSyncCache.load(cache_path)

    # L2 — fetch remote map
    client = IssueClient(target)
    try:
        try:
            remote_map = _fetch_remote_feature_map(client)
            cache.sync_from_remote(target, remote_map)
        except Exception as exc:  # noqa: BLE001
            _log.warning("L2 remote dedup failed (non-fatal): %s", exc)

        # Check L1/L2 cache
        existing = cache.get(target, feature_id)

        if existing:
            feature_title = sf.record.title
            if existing.get("state") == "open":
                if not _confirm_duplicate_override(feature_id, feature_title, existing):
                    result.warned.append(
                        {
                            "feature_id": feature_id,
                            "feature_title": feature_title,
                            "action": "user cancelled after duplicate warning",
                            "existing_issue": existing.get("issue_url"),
                        }
                    )
                    return result
            elif existing.get("state") == "closed":
                if mode == "skip":
                    result.skipped.append(
                        {
                            "feature_id": feature_id,
                            "feature_title": feature_title,
                            "reason": "closed issue exists (mode=skip)",
                            "existing_issue": existing.get("issue_url"),
                        }
                    )
                    return result
                elif mode == "ask":
                    if not _confirm_duplicate_override(feature_id, feature_title, existing):
                        result.warned.append(
                            {
                                "feature_id": feature_id,
                                "feature_title": feature_title,
                                "action": "user declined re-create for closed issue",
                                "existing_issue": existing.get("issue_url"),
                            }
                        )
                        return result
                # mode == "retry": fall through

        llm_info = llm_importance.get(feature_id, {})
        labels = list(config.sync_issues_labels) + [f"category:{sf.record.category.value}"]
        body = _build_issue_body(
            sf,
            digest or CommunityDigest(period="manual", generated_at="", summary=""),
            llm_info,
            duplicate_warning=(f"此 issue 与 #{existing.get('issue_number')} 关联同一 feature" if existing else ""),
        )

        resp = client.create_issue(title=sf.record.title, body=body, labels=labels)
        if resp is None:
            result.errors.append(f"failed to create issue for {feature_id}: {sf.record.title}")
            return result

        # Add labels in a follow-up call when the platform skips them in the
        # create request (e.g. GitCode WAF blocks labels in form-encoded POSTs).
        if target.platform.create_issue_skip_labels and not client.add_labels_to_issue(
            resp.get("number") or resp.get("iid"),
            labels,
        ):
            _log.warning(
                "failed to add labels to issue %s for feature %s",
                resp.get("number", "?"),
                feature_id,
            )

        issue_number = resp.get("number") or resp.get("iid", "?")
        issue_url = resp.get("html_url", f"{target.web_url}/issues/{issue_number}")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        cache.put(
            target,
            feature_id,
            {
                "feature_id": feature_id,
                "issue_number": issue_number,
                "issue_url": issue_url,
                "state": "open",
                "feature_title": sf.record.title,
                "created_at": now,
            },
        )
        result.created.append(
            {
                "feature_id": feature_id,
                "issue_number": issue_number,
                "issue_url": issue_url,
                "feature_title": sf.record.title,
            }
        )

        try:
            cache.save(cache_path)
        except OSError as exc:
            _log.warning("failed to save issue sync cache: %s", exc)

        return result
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Interactive candidates table
# ---------------------------------------------------------------------------


def list_candidates_interactive(
    *,
    config: RadarConfig,
    cache_dir: str = ".cache/community-radar",
) -> list[dict[str, Any]] | None:
    """Load the most recent scan digest and present an interactive table.

    Returns a list of candidate dicts or ``None`` if no digest is found.
    The caller is responsible for creating issues from the selections.
    """
    output_dir = Path(config.output_dir)
    if not output_dir.exists():
        print("未找到报告输出目录。请先执行 scan。")
        return None

    json_files = sorted(
        [p for p in output_dir.glob("*.json") if ".proposals." not in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not json_files:
        print("未找到 scan 报告。请先执行 clawcodex-dev community-radar scan。")
        return None

    latest = json_files[0]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"无法解析报告文件 {latest}: {exc}")
        return None

    highlights = data.get("highlights", [])
    llm_raw = data.get("llm_importance", {})
    if isinstance(llm_raw, list):
        llm_imp: dict[str, dict[str, str]] = {}
        for item in llm_raw:
            rid = item.get("feature_id") or item.get("id", "")
            if rid:
                llm_imp[rid] = item
    elif isinstance(llm_raw, dict):
        llm_imp = llm_raw
    else:
        llm_imp = {}

    candidates: list[dict[str, Any]] = []
    for h in highlights:
        rid = h.get("id") or h.get("record", {}).get("id", "")
        info: dict[str, Any] = llm_imp.get(rid, {})
        if isinstance(info, dict) and info.get("level") == "MAJOR":
            score_data = h.get("score", {})
            overall = score_data.get("overall", 0) if isinstance(score_data, dict) else 0
            source = h.get("source", "") or h.get("record", {}).get("source", "")
            title = h.get("title", "") or h.get("record", {}).get("title", "?")
            candidates.append(
                {
                    "feature_id": rid,
                    "title": title[:60],
                    "score": overall,
                    "source": source[:12],
                }
            )

    if not candidates:
        print("最近一次 scan 没有 MAJOR 特性可供选择。")
        return None

    candidates.sort(key=lambda c: c["score"], reverse=True)

    period = data.get("period", "?")
    print()
    print(f"┌{'─' * 64}┐")
    print(f"│  Community Radar — 最近扫描 MAJOR 特性 ({period}){' ' * (38 - len(period))}│")
    print(f"├────┬{'─' * 45}┬───────┬───────┤")
    print(f"│  # │ {'特性':<43} │ {'评分':<5} │ {'来源':<5} │")
    print(f"├────┼{'─' * 45}┼───────┼───────┤")
    for i, c in enumerate(candidates, 1):
        title = c["title"][:43]
        print(f"│ {i:>2} │ {title:<43} │ {c['score']:>5.1f} │ {c['source']:<5} │")
    print(f"├────┴{'─' * 45}┴───────┴───────┤")
    print(f"│  输入序号提 issue，输入 q 退出{' ' * 27}│")
    print(f"└{'─' * 64}┘")

    try:
        choice = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if choice.lower() in ("q", "quit", "exit"):
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(candidates):
            return [candidates[idx]]
    except ValueError:
        pass  # The candidate value is invalid; continue with the existing fallback.

    print(f"无效输入: {choice}")
    return None
