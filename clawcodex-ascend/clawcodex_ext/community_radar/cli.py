# pylint: disable=relative-beyond-top-level
"""``clawcodex-dev community-radar`` subcommand.

Mirrors the existing fast-path CLI style (see ``model_cmd/commands.py``
and ``provider_cmd/commands.py``). The subcommand is registered via
``clawcodex_ext.cli.subcommand_registry.register`` so the parent
dispatcher routes ``clawcodex-dev community-radar ...`` here without
touching ``src/*``.

Subcommands:

* ``scan`` — fetch + extract + score + write a digest.
* ``source list|add|remove|show`` — manage WatchSource registry.
* ``config show|set|init`` — manage RadarConfig.
* ``status`` — summarise the last scan and current registry state.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .config import RadarConfig, apply_env_overrides, default_config_path
from .discover import (
    MAX_COUNT_PER_DISCOVER,
    MAX_SOURCES_TOTAL,
    DiscoverResult,
    _format_stars,
    discover_sources,
)
from .models import (
    CommunityDigest,
    FeatureRecord,
    FeatureScore,
    ScoredFeature,
    WatchSource,
)
from .pipeline import CommunityRadarPipeline
from .registry import SourceRegistry, default_registry_path

_log = logging.getLogger(__name__)


USAGE = (
    "usage: clawcodex-dev community-radar <subcommand> [options]\n\n"
    "Subcommands:\n"
    "  scan [--period weekly|monthly|full] [--output DIR] [--no-write]\n"
    "                         [--no-persistent] [--compare] [--incremental]\n"
    "                         [--language zh|en] [--sync-issues] [--sync-max N]\n"
    "                         [--repo REPO] [--platform gitcode|github|gitee]\n"
    "                         [--closed-issue-mode ask|skip|retry]\n"
    "                         Fetch, extract, score, and persist a digest.\n"
    "  issue-sync [--feature-id ID | --interactive] [--repo REPO]\n"
    "                         [--platform gitcode|github|gitee]\n"
    "                         Create issues for a selected MAJOR candidate.\n"
    "  source list            List configured WatchSources.\n"
    "  source add NAME --repo OWNER/NAME [--track-releases|--track-commits|\n"
    "                         --track-prs] [--tag-filter REGEX]\n"
    "                         [--changelog PATH] [--notes TEXT]\n"
    "                         [--roadmap-keyword KW]...\n"
    "  source discover [--domain DOMAIN] [--min-stars N] [--count N]\n"
    "                         [--lang LANG]\n"
    "                         Auto-discover GitHub repos and add them as sources.\n"
    "  source remove NAME     Remove a configured source.\n"
    "  source show NAME       Print a single source's full record.\n"
    "  config show            Print the active RadarConfig.\n"
    "  config init            Write the default config to the user dir.\n"
    "  status                 Print current registry + last cache state.\n"
    "  help, --help, -h       Print this help.\n"
)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _load_registry(path: Path | None = None) -> SourceRegistry:
    """Load the registry, in-memory seeding with defaults when the file is absent.

    Unlike :func:`_maybe_load_registry`, this does **not** persist defaults
    to disk. Read-only commands (``source list``, ``source show``) use
    this so they never write a file just because the user listed sources.
    """
    target = path or default_registry_path()
    registry = SourceRegistry(target)
    if not target.exists():
        return SourceRegistry.with_defaults(target)
    registry.load()
    return registry


def _maybe_load_registry(path: Path | None) -> SourceRegistry:
    """Load the registry, seeding it with defaults if the file is absent.

    Persists the defaults to disk so subsequent invocations find a real
    config file. Used by mutating commands (``scan``, ``source add``).
    """
    target = path or default_registry_path()
    registry = SourceRegistry(target)
    if not target.exists():
        defaults = SourceRegistry.with_defaults(target)
        defaults.save()
        return defaults
    registry.load()
    return registry


# ---------------------------------------------------------------------------
# scan subcommand
# ---------------------------------------------------------------------------


def _cmd_scan(args: argparse.Namespace) -> int:
    config = apply_env_overrides(RadarConfig.from_dict(_read_config_file()))
    if args.output:
        config.output_dir = str(args.output)
    if args.language:
        config.language = args.language

    # Issue sync CLI flags override config
    sync_enabled = getattr(args, "sync_issues", None)
    if sync_enabled is not None:
        config.sync_issues = bool(sync_enabled)
    sync_max = getattr(args, "sync_max", None)
    if sync_max is not None:
        config.sync_issues_max_per_scan = int(sync_max)
    cli_repo = getattr(args, "repo", None)
    cli_platform = getattr(args, "platform", None)
    closed_issue_mode = getattr(args, "closed_issue_mode", None)
    if closed_issue_mode is not None:
        config.sync_issues_closed_issue_mode = closed_issue_mode

    if config.sync_issues:
        # Validate target resolution early for better user feedback
        from .issue_platforms import resolve_target

        target = resolve_target(
            config_target_repo=config.target_repo,
            config_api_token=config.api_token,
            cli_repo=cli_repo,
            cli_platform=cli_platform,
        )
        if target is None:
            print(
                "⚠️  无法确定目标仓库。请通过以下方式之一指定：\n"
                "    1. clawcodex-dev community-radar scan --sync-issues --repo owner/repo\n"
                "    2. 在 ~/.clawcodex/community-radar/config.yaml 中设置 target_repo\n"
                "    3. 确保当前 git 项目绑定了 GitCode/GitHub/Gitee remote"
            )
            return 1
        if not target.api_token:
            env_names = ", ".join(target.platform.token_env_vars)
            print(f"⚠️  未提供 {target.platform.name} API token。请设置 {env_names} 环境变量。")
            return 1
        _resolved_target = target
        _resolved_cli_repo = cli_repo
        _resolved_cli_platform = cli_platform
    else:
        _resolved_target = None
        _resolved_cli_repo = None
        _resolved_cli_platform = None

    registry = _maybe_load_registry(args.registry)
    pipeline = CommunityRadarPipeline(config=config, registry=registry)
    result = pipeline.run_scan(
        period=args.period,
        write=not args.no_write,
        output_dir=config.output_dir,
        persistent_copy=not args.no_persistent,
        compare=args.compare,
        incremental=args.incremental,
        issue_sync_target=_resolved_target,
        issue_sync_cli_repo=_resolved_cli_repo,
        issue_sync_cli_platform=_resolved_cli_platform,
        issue_sync_closed_issue_mode=closed_issue_mode,
    )

    summary = result.digest.to_dict()["stats"]
    print(
        f"Scan complete: {summary['total_features']} features from "
        f"{len(result.digest.sources_used)} sources "
        f"({summary['total_versions']} versions)."
    )
    if result.write_result:
        print(f"  digest: {result.write_result.markdown_path}")
        print(f"  json:   {result.write_result.json_path}")
    if result.digest.errors:
        print(f"  warnings: {len(result.digest.errors)} (see digest)")

    # ── Issue sync output ──
    if result.issue_sync:
        isr = result.issue_sync
        if isr.created:
            repo_url = (
                _resolved_target.web_url
                if _resolved_target
                else isr.created[0].get("issue_url", "").rsplit("/issues/", 1)[0]
            )
            created_n = len(isr.created)
            print(f"\nIssue sync: {created_n} created → {repo_url}")
            for item in isr.created:
                title_short = item.get("feature_title", "")[:50]
                issue_no = item.get("issue_number", "?")
                print(f"  #{issue_no}  {title_short}  → {item.get('issue_url', '')}")
        if isr.errors:
            for err in isr.errors:
                print(f"  issue sync error: {err}")
    return 0


# ---------------------------------------------------------------------------
# issue-sync subcommand (Path B: manual single-feature issue creation)
# ---------------------------------------------------------------------------


def _build_scored_list(raw_list: list[dict[str, Any]]) -> list[ScoredFeature]:
    """Convert raw digest JSON rows into :class:`ScoredFeature` objects."""
    result: list[ScoredFeature] = []
    for item in raw_list:
        record = FeatureRecord.from_dict(item.get("record", item))
        sd = item.get("score", {})
        if isinstance(sd, dict):
            score = FeatureScore(
                record_id=record.id,
                overall=float(sd.get("overall", 0)),
                popularity=float(sd.get("popularity", 0)),
                maturity=float(sd.get("maturity", 0)),
                adaptation_cost=float(sd.get("adaptation_cost", 0)),
                strategic_value=float(sd.get("strategic_value", 0)),
                architecture_fit=float(sd.get("architecture_fit", 0)),
            )
        else:
            score = FeatureScore(
                record_id=record.id,
                overall=0,
                popularity=0,
                maturity=0,
                adaptation_cost=0,
                strategic_value=0,
                architecture_fit=0,
            )
        result.append(ScoredFeature(record=record, score=score))
    return result


def _cmd_issue_sync(args: argparse.Namespace) -> int:
    """Manual single-feature issue creation."""
    config = apply_env_overrides(RadarConfig.from_dict(_read_config_file()))
    lang = getattr(args, "language", None)
    if lang:
        config.language = lang

    from .issue_platforms import resolve_target
    from .issue_sync import list_candidates_interactive, sync_single_feature

    target = resolve_target(
        config_target_repo=config.target_repo,
        config_api_token=config.api_token,
        cli_repo=args.repo,
        cli_platform=args.platform,
    )
    if target is None:
        print(
            "⚠️  无法确定目标仓库。请通过以下方式之一指定：\n"
            "    1. clawcodex-dev community-radar issue-sync --repo owner/repo\n"
            "    2. 在 ~/.clawcodex/community-radar/config.yaml 中设置 target_repo\n"
            "    3. 确保当前 git 项目绑定了 GitCode/GitHub/Gitee remote"
        )
        return 1
    if not target.api_token:
        env_names = ", ".join(target.platform.token_env_vars)
        print(f"⚠️  未提供 {target.platform.name} API token。请设置 {env_names} 环境变量。")
        return 1

    # Determine feature_id
    feature_id: str | None = getattr(args, "feature_id", None)

    if args.interactive or not feature_id:
        candidates = list_candidates_interactive(config=config, cache_dir=config.cache_dir)
        if not candidates:
            return 1
        feature_id = candidates[0]["feature_id"]

    if not feature_id:
        print("请指定 --feature-id 或使用 --interactive 选择特性。")
        return 1

    # Resolve closed_issue_mode from CLI
    closed_mode = getattr(args, "closed_issue_mode", None)
    if closed_mode is not None:
        config.sync_issues_closed_issue_mode = closed_mode

    # Load the latest digest for feature lookup and issue body generation
    digest: CommunityDigest | None = None
    llm_importance: dict[str, dict[str, str]] | None = None
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        json_files = sorted(
            [p for p in output_dir.glob("*.json") if ".proposals." not in p.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if json_files:
            try:
                data = json.loads(json_files[0].read_text(encoding="utf-8"))
                llm_raw = data.get("llm_importance", {})
                if isinstance(llm_raw, dict):
                    llm_importance = {
                        str(fid): {str(k): str(v) for k, v in entry.items()}
                        for fid, entry in llm_raw.items()
                        if isinstance(entry, dict)
                    }

                highlights = _build_scored_list(data.get("highlights", []))
                trending = _build_scored_list(data.get("trending", []))
                digest = CommunityDigest(
                    period=str(data.get("period", "weekly")),
                    generated_at=str(data.get("generated_at", "")),
                    summary=str(data.get("summary", "")),
                    period_start=str(data.get("period_start", "")),
                    highlights=highlights,
                    trending=trending,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("failed to load digest for issue-sync: %s", exc)

    result = sync_single_feature(
        feature_id=feature_id,
        config=config,
        digest=digest,
        llm_importance=llm_importance,
        target=target,
        cache_dir=config.cache_dir,
        closed_issue_mode=closed_mode,
    )

    if result.created:
        repo_url = target.web_url
        created_n = len(result.created)
        print(f"\nIssue sync: {created_n} created → {repo_url}")
        for item in result.created:
            title_short = item.get("feature_title", "")[:50]
            issue_no = item.get("issue_number", "?")
            print(f"  #{issue_no}  {title_short}  → {item.get('issue_url', '')}")
    if result.errors:
        for err in result.errors:
            print(f"  issue sync error: {err}")
    if result.warned:
        for w in result.warned:
            print(f"  ⚠️  {w.get('action', 'cancelled')}: {w.get('feature_title', '')}")
    return 0


# ---------------------------------------------------------------------------
# source subcommands
# ---------------------------------------------------------------------------


def _cmd_source_list(args: argparse.Namespace) -> int:
    registry = _load_registry(args.registry)
    if not registry.list():
        print("(no sources configured)")
        return 0
    for source in registry.list():
        flags = _format_track_flags(source)
        print(f"{source.name}\t{source.repo}\t{flags}")
    return 0


def _cmd_source_add(args: argparse.Namespace) -> int:
    registry = _load_registry(args.registry)
    data: dict[str, Any] = {
        "name": args.name,
        "repo": args.repo,
        "track_releases": args.track_releases,
        "track_commits": args.track_commits,
        "track_prs": args.track_prs,
        "track_issues": args.track_issues,
    }
    if args.domain:
        data["domain"] = args.domain
    if args.tag_filter:
        data["release_tag_filter"] = args.tag_filter
    if args.changelog:
        data["changelog_path"] = args.changelog
    if args.notes:
        data["notes"] = args.notes
    if args.roadmap_keyword:
        data["roadmap_keywords"] = list(args.roadmap_keyword)
    try:
        source = WatchSource.from_dict(data)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    registry.add(source)
    registry.save()
    print(f"Added source '{source.name}' -> {source.repo}")
    return 0


def _cmd_source_remove(args: argparse.Namespace) -> int:
    registry = _load_registry(args.registry)
    if not registry.remove(args.name):
        print(f"error: no source named '{args.name}'", file=sys.stderr)
        return 1
    registry.save()
    print(f"Removed source '{args.name}'")
    return 0


def _cmd_source_show(args: argparse.Namespace) -> int:
    registry = _load_registry(args.registry)
    source = registry.get(args.name)
    if source is None:
        print(f"error: no source named '{args.name}'", file=sys.stderr)
        return 1
    print(json.dumps(source.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_source_discover(args: argparse.Namespace) -> int:
    """``source discover`` — search GitHub and add repos to the registry."""
    from .fetcher import make_fetcher

    registry = _maybe_load_registry(args.registry)

    result = discover_sources(
        make_fetcher(cache_dir=".cache/community-radar"),
        registry,
        domain=args.domain,
        min_stars=args.min_stars,
        count=args.count,
        lang=args.lang,
    )

    _print_discover_result(result, args)
    return 0 if result.added else 1


# ---------------------------------------------------------------------------
# discover output formatting
# ---------------------------------------------------------------------------


def _print_discover_result(result: DiscoverResult, args: argparse.Namespace) -> None:
    """Print a user-friendly summary of a :class:`DiscoverResult`."""

    # ── Warnings ──────────────────────────────────────────────────────
    if result.search_total == 0:
        print("GitHub Search returned no results — try relaxing --min-stars or --domain.")
        return

    if result.total_limit_warning:
        print(f"\u26a0  Config already has {MAX_SOURCES_TOTAL} sources (the maximum) — cannot add more.")

    if result.count_ceiling_warning:
        print(f"\u26a0  --count capped at {MAX_COUNT_PER_DISCOVER} (requested {result.requested_count}).")

    if result.not_enough_warning and not result.total_limit_warning:
        added_n = len(result.added)
        wanted = min(args.count, MAX_COUNT_PER_DISCOVER)
        if added_n < wanted:
            print(f"\u26a0  Only {added_n} qualifying repos found (requested {wanted}).")

    # ── Additions ─────────────────────────────────────────────────────
    if result.added:
        print(f"Added {len(result.added)} source(s):")
        for s in result.added:
            stars = result.added_stars.get(s.repo, 0)
            star_str = _format_stars(stars)
            print(f"  - {s.repo} ({s.domain}) \u2b50 {star_str}")
    elif not result.total_limit_warning:
        print("No new sources added.")


# ---------------------------------------------------------------------------
# config subcommands
# ---------------------------------------------------------------------------


def _cmd_config_show(args: argparse.Namespace) -> int:
    config = apply_env_overrides(RadarConfig.from_dict(_read_config_file()))
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_config_init(args: argparse.Namespace) -> int:
    path = args.config or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = RadarConfig()
    path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Default config written to {path}")
    return 0


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> int:
    registry_path = args.registry or default_registry_path()
    config_path = args.config or default_config_path()
    registry = _load_registry(registry_path)
    print(f"Registry path: {registry_path}")
    print(f"  sources:    {len(registry)}")
    if registry.list():
        print(f"  names:      {', '.join(registry.names())}")
    print(f"Config path:  {config_path}  (exists={config_path.exists()})")
    cache_dir = os.environ.get(
        "CLAWCODEX_RADAR_CACHE_DIR",
        str(Path.home() / ".cache" / "community-radar"),
    )
    cache = Path(cache_dir)
    print(f"Cache dir:    {cache}  (exists={cache.exists()})")
    return 0


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_track_flags(source: WatchSource) -> str:
    flags = []
    if source.track_releases:
        flags.append("releases")
    if source.track_commits:
        flags.append("commits")
    if source.track_prs:
        flags.append("prs")
    if source.track_issues:
        flags.append("issues")
    return ",".join(flags) or "(none)"


def _read_config_file() -> dict[str, Any]:
    """Load the YAML/JSON RadarConfig file from disk (or empty dict)."""
    path = default_config_path()
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning("failed to read config %s: %s", path, exc)
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError:
            return {}
        loader = yaml.safe_load
    else:
        loader = json.loads
    try:
        data = loader(text)
    except Exception:  # noqa: BLE001
        return {}
    if isinstance(data, dict):
        inner = data.get("community_radar") or data.get("radar") or data
        return inner if isinstance(inner, dict) else {}
    return {}


# ---------------------------------------------------------------------------
# Argument parsing + dispatcher
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clawcodex-dev community-radar",
        description="SR-5.1 开源社区新特性雷达",
        add_help=False,
    )
    parser.add_argument("--registry", type=Path, default=None)
    sub = parser.add_subparsers(dest="subcommand")

    sub.add_parser("help")

    # scan
    scan_p = sub.add_parser("scan")
    scan_p.add_argument("--period", choices=("weekly", "monthly", "full"), default="full")
    scan_p.add_argument("--output", type=Path, default=None)
    scan_p.add_argument("--no-write", action="store_true")
    scan_p.add_argument("--no-persistent", action="store_true")
    scan_p.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Compare against the previous digest in the output directory.",
    )
    scan_p.add_argument(
        "--incremental",
        action="store_true",
        default=False,
        help="Incremental fetch (use cursors/ETags for speed, suitable for cron).",
    )
    scan_p.add_argument(
        "--language",
        choices=("zh", "en"),
        default=None,
        help="Report language: zh (Chinese) or en (English). "
        "Overrides the config file and CLAWCODEX_RADAR_LANGUAGE env var.",
    )
    # Issue sync flags (Path A: auto-sync after scan)
    scan_p.add_argument(
        "--sync-issues",
        action="store_true",
        default=None,
        help="Auto-create issues for top MAJOR features after scan.",
    )
    scan_p.add_argument(
        "--sync-max",
        type=int,
        default=None,
        help="Maximum issues to create (default: 2, from config).",
    )
    scan_p.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Target repository (owner/repo or platform.com/owner/repo).",
    )
    scan_p.add_argument(
        "--platform",
        type=str,
        choices=("gitcode", "github", "gitee"),
        default=None,
        help="Platform type (auto-detected if not specified).",
    )
    scan_p.add_argument(
        "--closed-issue-mode",
        type=str,
        choices=("ask", "skip", "retry"),
        default=None,
        help="How to handle features whose previous issue was closed: "
        "ask (prompt), skip (don't re-create), retry (always re-create). "
        "Default: ask.",
    )
    # issue-sync (Path B: manual single-feature issue creation)
    issue_sync_p = sub.add_parser("issue-sync")
    issue_sync_p.add_argument(
        "--feature-id",
        type=str,
        default=None,
        help="Feature ID from a JSON report to create an issue for.",
    )
    issue_sync_p.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Interactive table to select a feature from the last scan.",
    )
    issue_sync_p.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Target repository (owner/repo or platform.com/owner/repo).",
    )
    issue_sync_p.add_argument(
        "--platform",
        type=str,
        choices=("gitcode", "github", "gitee"),
        default=None,
        help="Platform type (auto-detected if not specified).",
    )
    issue_sync_p.add_argument(
        "--closed-issue-mode",
        type=str,
        choices=("ask", "skip", "retry"),
        default=None,
        help="How to handle features whose previous issue was closed.",
    )
    # source
    source_p = sub.add_parser("source")
    source_sub = source_p.add_subparsers(dest="source_cmd", required=True)

    source_sub.add_parser("list")

    add_p = source_sub.add_parser("add")
    add_p.add_argument("name")
    add_p.add_argument("--repo", required=True)
    add_p.add_argument(
        "--domain",
        choices=("code_agent", "embodied_ai", "spatial_intelligence", "general"),
        default=None,
        help="Source domain for cross-domain classification guard.",
    )
    add_p.add_argument("--track-releases", dest="track_releases", action="store_true", default=True)
    add_p.add_argument("--no-track-releases", dest="track_releases", action="store_false")
    add_p.add_argument("--track-commits", dest="track_commits", action="store_true", default=False)
    add_p.add_argument("--track-prs", dest="track_prs", action="store_true", default=False)
    add_p.add_argument("--track-issues", dest="track_issues", action="store_true", default=False)
    add_p.add_argument("--tag-filter", dest="tag_filter", default=None)
    add_p.add_argument("--changelog", dest="changelog", default=None)
    add_p.add_argument("--notes", dest="notes", default=None)
    add_p.add_argument("--roadmap-keyword", dest="roadmap_keyword", action="append", default=[])

    rm_p = source_sub.add_parser("remove")
    rm_p.add_argument("name")

    show_p = source_sub.add_parser("show")
    show_p.add_argument("name")

    discover_p = source_sub.add_parser("discover")
    discover_p.add_argument(
        "--domain",
        choices=("code_agent", "embodied_ai", "spatial_intelligence"),
        default=None,
        help="Only search for repos in this domain.",
    )
    discover_p.add_argument(
        "--min-stars",
        type=int,
        default=100,
        help="Minimum stars filter (default: 100, set to 0 to disable).",
    )
    discover_p.add_argument(
        "--count",
        type=int,
        default=5,
        help=f"Number of sources to add (max: {MAX_COUNT_PER_DISCOVER}).",
    )
    discover_p.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Filter by programming language (e.g. python, typescript).",
    )

    # config
    cfg_p = sub.add_parser("config")
    cfg_sub = cfg_p.add_subparsers(dest="config_cmd", required=True)
    cfg_sub.add_parser("show")
    init_p = cfg_sub.add_parser("init")
    init_p.add_argument("--config", type=Path, default=None)

    # status
    status_p = sub.add_parser("status")
    status_p.add_argument("--config", type=Path, default=None)

    return parser


_DISPATCH: dict[str, Callable[[argparse.Namespace], int]] = {
    "scan": _cmd_scan,
    "source_list": _cmd_source_list,
    "source_add": _cmd_source_add,
    "source_remove": _cmd_source_remove,
    "source_show": _cmd_source_show,
    "source_discover": _cmd_source_discover,
    "config_show": _cmd_config_show,
    "config_init": _cmd_config_init,
    "status": _cmd_status,
    "issue-sync": _cmd_issue_sync,
}


def _route(args: argparse.Namespace) -> int:
    sub = args.subcommand
    if sub is None or sub == "help":
        print(USAGE)
        return 0
    if sub == "source":
        key = f"source_{args.source_cmd}"
        handler = _DISPATCH.get(key)
        if handler is None:
            print(USAGE)
            return 2
        return handler(args)
    if sub == "config":
        key = f"config_{args.config_cmd}"
        handler = _DISPATCH.get(key)
        if handler is None:
            print(USAGE)
            return 2
        return handler(args)
    handler = _DISPATCH.get(sub)
    if handler is None:
        print(USAGE)
        return 2
    return handler(args)


def run(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (defaulting to ``sys.argv[1:]``) and dispatch."""
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in {"help", "--help", "-h"}:
        print(USAGE)
        return 0
    parser = _build_parser()
    args = parser.parse_args(list(argv))
    return _route(args)


# ---------------------------------------------------------------------------
# Subcommand registration
# ---------------------------------------------------------------------------


def register_community_radar_subcommand() -> None:
    """Register ``community-radar`` with the downstream CLI dispatcher.

    Mirrors the registration style in
    ``clawcodex_ext/cli/session_migrate_cmd.py`` — we import the
    registry lazily to avoid a circular import.
    """
    from clawcodex_ext.cli.subcommand_registry import register

    @register("community-radar")
    def _handler(args: list[str]) -> int:
        try:
            return run(args)
        except Exception as exc:  # noqa: BLE001
            print(f"community-radar: {exc}", file=sys.stderr)
            return 1
