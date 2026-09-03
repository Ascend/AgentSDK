#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""orchestrator rules — inspect and manage learned PR review conventions.

Usage (noun-verb):

  # Query
  clawcodex orchestrator rules list [--workflow PATH]
  clawcodex orchestrator rules review --id <id> [--workflow PATH]
  clawcodex orchestrator rules stats [--workflow PATH]

  # Management
  clawcodex orchestrator rules delete --id <id> [--workflow PATH]
  clawcodex orchestrator rules refresh [--workflow PATH]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from extensions.orchestrator.rules_learner import RuleEngine, RuleStore
from extensions.orchestrator.workflow import WorkflowLoader


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_rules_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``rules`` sub-subcommands."""
    rules_parser = subparsers.add_parser(
        "rules",
        help="Inspect and manage learned PR review conventions",
        description="List, review, delete, refresh, or show stats for rules "
        "automatically extracted from PR review feedback.",
    )
    rules_sub = rules_parser.add_subparsers(
        dest="rules_subcommand",
        required=True,
    )

    # --- rules list ---
    list_parser = rules_sub.add_parser(
        "list",
        help="List all rules with summary and metadata",
        description="Display all rules in the workflow rules file. Idempotent (pure read).",
    )
    list_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (optional auto-detection override)",
    )

    # --- rules review ---
    review_parser = rules_sub.add_parser(
        "review",
        help="Show full detail for a specific rule",
        description="Display one rule with its full body, source, and metadata.",
    )
    review_parser.add_argument(
        "--id",
        type=int,
        required=True,
        metavar="RULE_ID",
        help="Rule id to review",
    )
    review_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (optional auto-detection override)",
    )

    # --- rules delete ---
    delete_parser = rules_sub.add_parser(
        "delete",
        help="Delete a rule by id",
        description="Remove a single rule from the rules file. "
        "The orchestrator will not re-add it unless re-extracted.",
    )
    delete_parser.add_argument(
        "--id",
        type=int,
        required=True,
        metavar="RULE_ID",
        help="Rule id to delete",
    )
    delete_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (optional auto-detection override)",
    )

    # --- rules extract ---
    extract_parser = rules_sub.add_parser(
        "extract",
        help="Extract rules from PR review follow-up commits",
        description="Scan workspace git history for review-followup commits "
        "(containing review metadata in their message) and use LLM to extract "
        "coding conventions from each commit's diff + review comment. "
        "Idempotent: already-extracted commits are skipped.",
    )
    extract_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (optional auto-detection override)",
    )
    extract_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="Max number of records to process (default: 10)",
    )
    extract_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan, do not write rules or update tracker",
    )

    # --- rules stats ---
    stats_parser = rules_sub.add_parser(
        "stats",
        help="Show rule statistics",
        description="Display aggregate statistics: total count, category distribution, "
        "average confidence, recent additions.",
    )
    stats_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (optional auto-detection override)",
    )


# ---------------------------------------------------------------------------
# Run dispatch
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate rules subcommand."""
    cmd = args.rules_subcommand

    workflow_path = _resolve_workflow_path(getattr(args, "workflow", None))
    if workflow_path is None:
        print(
            "No workflow found. Specify --workflow PATH or run from a workflow directory.",
            file=sys.stderr,
        )
        return 1

    rules_path = _resolve_rules_path(workflow_path)
    if rules_path is None:
        print("Rules are not enabled in the workflow configuration.", file=sys.stderr)
        return 1

    if cmd == "list":
        return _run_list(rules_path)
    elif cmd == "review":
        return _run_review(rules_path, args.id)
    elif cmd == "delete":
        return _run_delete(rules_path, args.id)
    elif cmd == "extract":
        return _run_extract(workflow_path, rules_path, args)
    elif cmd == "stats":
        return _run_stats(rules_path)
    else:
        print(f"Unknown rules subcommand: {cmd}", file=sys.stderr)
        return 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_workflow_path(workflow_arg: str | None) -> str | None:
    """Resolve the WORKFLOW.md path from CLI arg, cwd, or daemon metadata."""
    if workflow_arg:
        p = Path(workflow_arg)
        if p.exists():
            return str(p.resolve())
        print(f"Workflow file not found: {workflow_arg}", file=sys.stderr)
        return None

    # 1. Try WORKFLOW.md in CWD
    default = WorkflowLoader.default_path()
    if default.exists():
        return str(default.resolve())

    # 2. Try orchestrator daemon metadata (auto-discover running daemon)
    try:
        import json

        from extensions.orchestrator.workspace_locator import _find_latest_metadata

        meta_file = _find_latest_metadata()
        if meta_file:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            wf = data.get("workflow_path")
            if wf and Path(wf).exists():
                return str(Path(wf).resolve())
    except Exception:  # nosec B110 - best-effort daemon metadata discovery
        pass

    return None


def _resolve_rules_path(workflow_path: str) -> str | None:
    """Load the workflow config and resolve the rules file path."""
    try:
        config, _ = WorkflowLoader.load(workflow_path)
    except Exception as exc:
        print(f"Failed to load workflow: {exc}", file=sys.stderr)
        return None

    if not getattr(config, "rules", None) or not config.rules.enabled:
        return None

    return RuleStore.resolve_path(workflow_path, config.rules.path or "")


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _run_list(rules_path: str) -> int:
    data = RuleStore.load(rules_path)
    rules = data.get("rules", [])
    if not rules:
        print(f"No rules in {rules_path}")
        return 0

    print(f"Rules file: {rules_path}")
    print(f"Total: {len(rules)} rule(s)\n")
    for r in rules:
        rid = r.get("id", "?")
        summary = r.get("summary", "")
        conf = r.get("confidence", "?")
        support = r.get("support_count", 0)
        cat = r.get("category", "?")
        print(f"  [{rid}] ({cat}) [{conf}] (x{support}) {summary}")
    return 0


def _run_review(rules_path: str, rule_id: int) -> int:
    data = RuleStore.load(rules_path)
    rules = data.get("rules", [])
    for r in rules:
        if r.get("id") == rule_id:
            _print_rule(r)
            return 0

    print(f"Rule #{rule_id} not found in {rules_path}", file=sys.stderr)
    return 1


def _run_delete(rules_path: str, rule_id: int) -> int:
    data = RuleStore.load(rules_path)
    rules = data.get("rules", [])
    before = len(rules)
    rules = [r for r in rules if r.get("id") != rule_id]
    removed = before - len(rules)
    if removed == 0:
        print(f"Rule #{rule_id} not found — nothing to delete", file=sys.stderr)
        return 1
    RuleStore.save(rules_path, rules, version=data.get("version", 1))
    print(f"Deleted rule #{rule_id} from {rules_path}")
    return 0


def _run_extract(workflow_path: str, rules_path: str, args: argparse.Namespace) -> int:
    """Extract rules from review-followup commits via LLM analysis."""
    import asyncio

    return asyncio.run(_run_extract_async(workflow_path, rules_path, args))


async def _run_extract_async(workflow_path: str, rules_path: str, args: argparse.Namespace) -> int:
    """Async body of ``extract`` command."""
    import shutil
    import subprocess

    from extensions.orchestrator.issue_registry import IssueRegistry
    from extensions.orchestrator.rules_learner import (
        BatchedLLMJudge,
        ExtractTracker,
        _infer_category,
    )

    dry_run = getattr(args, "dry_run", False)
    limit = getattr(args, "limit", 10)

    # 1. Locate registry
    config, _ = WorkflowLoader.load(workflow_path)
    ws_root = getattr(config.workspace, "root", None)
    if not ws_root:
        print("No workspace.root in workflow config.", file=sys.stderr)
        return 1
    registry_path = Path(ws_root) / ".clawcodex_issue_registry.json"
    if not registry_path.exists():
        print(f"Registry not found at {registry_path}", file=sys.stderr)
        return 1
    registry = IssueRegistry(registry_path)

    # 2. Find records with PRs
    records = registry.iter_records_with_pr()
    if not records:
        print("No records with PRs found in registry.")
        return 0

    # 3. Load existing rules + tracker
    tracker = ExtractTracker(rules_path)
    processed = tracker.load()
    existing_data = RuleStore.load(rules_path)
    existing = existing_data.get("rules", [])

    count = 0
    for record in records[:limit]:
        ws_path = record.workspace_path or ""
        if not ws_path or not Path(ws_path).exists():
            print(f"  ⏭️  {record.issue_identifier}: workspace not found, skip")
            continue

        repo = Path(ws_path)
        pr_num = record.pr_number or ""
        branch = record.branch_name or ""

        # 4. Scan commits for review metadata
        # Use ASCII record separator (\x1e) between commits and unit
        # separator (\x1f) between SHA and body, so commit messages
        # containing newlines or '---' don't break parsing.
        try:
            log_output = subprocess.run(
                [shutil.which("git") or "git", "log", branch, "--format=%H%x1f%B%x1e"],
                capture_output=True,
                text=True,
                cwd=repo,
                timeout=30,
                check=False,
            )
            if log_output.returncode != 0:
                print(f"  ⏭️  {record.issue_identifier}: git log failed, skip")
                continue
        except Exception as exc:
            print(f"  ⏭️  {record.issue_identifier}: {exc}, skip")
            continue

        commits = []
        for entry in log_output.stdout.split("\x1e"):
            if not entry.strip():
                continue
            parts = entry.split("\x1f", 1)
            if len(parts) < 2:
                continue
            sha = parts[0].strip()
            msg = parts[1].strip()
            if "review-pr:" in msg and sha not in processed:
                commits.append((sha, msg))

        if not commits:
            print(f"  ⏭️  {record.issue_identifier}: no new review commits")
            continue

        print(f"\n  📦 {record.issue_identifier} (PR #{pr_num}, branch {branch}):")
        for sha, msg in commits[:5]:
            print(f"    commit {sha[:8]}: {msg.splitlines()[0][:60]}")
            if dry_run:
                processed.add(sha)
                count += 1
                continue

            # 5. LLM analyze this commit
            diff = subprocess.run(
                [shutil.which("git") or "git", "diff", f"{sha}^..{sha}", "--", "*.py"],
                capture_output=True,
                text=True,
                cwd=repo,
                timeout=30,
                check=False,
            ).stdout[:4000]

            # Extract review metadata from commit body
            review_pr = ""
            review_body = ""
            for line in msg.splitlines():
                if line.startswith("review-pr:"):
                    review_pr = line.split(":", 1)[1].strip()
                elif line.startswith("review-body:"):
                    review_body = line.split(":", 1)[1].strip()

            # 6. Build prompt for LLM
            judge = BatchedLLMJudge()
            judge_prompt = (
                f"Analyze this PR review follow-up commit and extract a coding "
                f"convention that should be followed going forward.\n\n"
                f"Review (PR {review_pr}): {review_body}\n\n"
                f"Code diff:\n```diff\n{diff}\n```\n\n"
                f"Extract ONE coding convention from this review. "
                f"Output it in this EXACT format:\n\n"
                f"- [category] Short summary of the convention\n"
                f"  Body: Detailed explanation with rationale. You MUST include this line.\n\n"
                f"category MUST be one of:\n"
                f'  naming          — e.g. "[naming] Use snake_case for function names"\n'
                f'  error_handling  — e.g. "[error_handling] Catch specific exceptions, not bare except"\n'
                f'  testing         — e.g. "[testing] Use pytest fixtures for shared setup"\n'
                f'  import_style    — e.g. "[import_style] Group stdlib imports first"\n'
                f'  code_style      — e.g. "[code_style] Use double quotes for string literals"\n'
                f'  type_annotation — e.g. "[type_annotation] Add return type to public functions"\n'
                f'  architecture    — e.g. "[architecture] Keep business logic out of route handlers"\n'
                f'  boilerplate     — e.g. "[boilerplate] Every module starts with a license header"\n'
                f'  security        — e.g. "[security] Never log API keys or tokens"\n'
                f'  performance     — e.g. "[performance] Use generator expressions for large datasets"\n'
                f"  other           — only if no category above fits\n\n"
                f"The Body line is MANDATORY — explain WHY this convention matters "
                f"and give a brief example."
            )

            try:
                llm_reply = await judge._run_clawcodex(judge_prompt)
            except Exception as exc:
                print(f"    ⚠ LLM analysis failed: {exc}")
                processed.add(sha)
                count += 1
                continue

            if not llm_reply.strip():
                print("    ⚠ LLM returned empty, record as processed")
                processed.add(sha)
                count += 1
                continue

            # 7. Parse and persist via RuleEngine
            candidates = RuleEngine.extract(f"## Extracted Rules\n{llm_reply}")
            if not candidates:
                print("    ⚠ No rules extracted from LLM output, record as processed")
                processed.add(sha)
                count += 1
                continue

            source = f"PR #{pr_num} commit {sha[:8]}"
            for c in candidates:
                c["source"] = source
                if not c.get("category") or c["category"] == "other":
                    inferred = _infer_category(f"{c.get('summary', '')} {c.get('body', '')}")
                    if inferred != "other":
                        c["category"] = inferred

            # LLM judge for semantic dedup / merge / conflict detection.
            # Skipped when no existing rules (first extraction) — judge()
            # would return all-NEW anyway. On failure, falls back to all-new.
            judge_results = None
            if existing:
                try:
                    judge_results = await judge.judge(candidates, existing)
                except Exception as exc:
                    print(f"    ⚠ LLM dedup judge failed ({exc}), using all-new fallback")

            merged = RuleEngine._deduplicate_and_merge(
                candidates,
                existing,
                _judge_results=judge_results,
            )

            # Assign IDs
            for i, r in enumerate(merged, start=1):
                r["id"] = i

            # Backfill conflict references from index-based to ID-based
            for r in merged:
                conflict_idx = r.pop("_conflict_with_idx", None)
                if conflict_idx is not None:
                    if isinstance(conflict_idx, list):
                        r["conflict_with"] = [merged[idx]["id"] for idx in conflict_idx]
                    else:
                        r["conflict_with"] = [merged[conflict_idx]["id"]]

            merged = RuleEngine.prune(merged, max_rules=20)

            # Save
            RuleStore.save(rules_path, merged, version=existing_data.get("version", 1))
            existing = merged
            existing_data = {"version": existing_data.get("version", 1), "rules": merged}

            processed.add(sha)
            count += 1
            print(f"    ✅ Extracted {len(candidates)} rule(s)")

    # 8. Save tracker
    if not dry_run and count > 0:
        tracker.save(processed)

    print(f"\nDone. Processed {count} commit(s).")
    return 0


def _run_refresh(workflow_path: str, rules_path: str) -> int:
    """已废弃 — 使用 ``extract`` 替代。"""
    print(
        "`refresh` is deprecated. Use `clawcodex orchestrator rules extract` instead.",
        file=sys.stderr,
    )
    return 1


def _run_stats(rules_path: str) -> int:
    data = RuleStore.load(rules_path)
    rules = data.get("rules", [])
    if not rules:
        print(f"No rules in {rules_path}")
        return 0

    total = len(rules)
    cats: dict[str, int] = {}
    confs: dict[str, int] = {}
    recent_7d = 0
    now = datetime.now(timezone.utc)

    for r in rules:
        cat = r.get("category", "other")
        cats[cat] = cats.get(cat, 0) + 1

        conf = r.get("confidence", "low")
        confs[conf] = confs.get(conf, 0) + 1

        created_str = r.get("created_at", "")
        if created_str:
            try:
                created = datetime.fromisoformat(created_str)
                if (now - created).total_seconds() < 7 * 86400:
                    recent_7d += 1
            except (ValueError, TypeError):
                pass

    # Average score
    scores = [RuleEngine.score(r) for r in rules]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"Rules file: {rules_path}")
    print(f"  Total rules:    {total}")
    print(f"  Avg quality:    {avg_score:.3f}")
    print(f"  Added (7d):     {recent_7d}")
    print()
    print("  Category distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    {cat:25s} {count:3d} ({pct:5.1f}%)")
    print()
    print("  Confidence distribution:")
    for level in ("high", "medium", "low"):
        count = confs.get(level, 0)
        pct = count / total * 100
        print(f"    {level:10s} {count:3d} ({pct:5.1f}%)")

    return 0


def _print_rule(r: dict[str, Any]) -> None:
    """Pretty-print a single rule dict."""
    rid = r.get("id", "?")
    summary = r.get("summary", "")
    body = r.get("body", "")
    cat = r.get("category", "?")
    conf = r.get("confidence", "?")
    support = r.get("support_count", 0)
    source = r.get("source", "")
    created = r.get("created_at", "")
    updated = r.get("updated_at", "")
    applied = r.get("last_applied", "")
    conflict = r.get("conflict_with", [])

    print(f"Rule #{rid}")
    print(f"  Category:   {cat}")
    print(f"  Confidence: {conf}")
    print(f"  Support:    x{support}")
    print(f"  Summary:    {summary}")
    if body:
        print("  Body:")
        for line in body.splitlines():
            print(f"    {line}")
    if conflict:
        print(f"  Conflicts:  rule(s) #{', #'.join(str(c) for c in conflict)}")
    if source:
        print(f"  Source:     {source}")
    print(f"  Created:    {created}")
    print(f"  Updated:    {updated}")
    print(f"  Applied:    {applied}")
