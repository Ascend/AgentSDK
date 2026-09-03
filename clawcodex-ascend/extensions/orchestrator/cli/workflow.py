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

"""orchestrator workflow — scaffold a workflow.md from the packaged template.

Usage (noun-verb):
  clawcodex orchestrator workflow init [--kind <tracker>] [--owner <o>] [--repo <r>]

Design:
  - Copies the workflow template from the package to the current directory.
  - Replaces ``{{PLACEHOLDER}}`` tokens with user-provided values.
  - The template path is resolved from the installed package, so users
    who installed via ``pip`` (no source tree) can still generate a
    workflow.md without manually hunting for template files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ── Template resolution ──────────────────────────────────────────────


def _template_path(variant: str = "workflow") -> Path:
    """Return the path to ``{variant}.template.md`` inside the package.

    Works for both editable installs (source tree) and non-editable
    installs (site-packages zip / wheel).

    Raises ``FileNotFoundError`` if the variant does not exist.
    """
    import extensions.orchestrator.templates as tpl_mod

    # Try named variant first, fall back to "workflow" generic name.
    # Supports both {variant}.template.md and {variant}.template (e.g. workflow.yaml.template)
    candidates: list[str] = []
    if "." in variant:
        # e.g. "workflow.yaml" → "workflow.yaml.template"
        candidates.append(f"{variant}.template")
    candidates.append(f"{variant}.template.md")
    candidates.append("workflow.template.md")

    for p in tpl_mod.__path__:  # type: ignore[attr-defined]
        for name in candidates:
            candidate = Path(p) / name
            if candidate.exists():
                return candidate

    # Python 3.9+ importlib.resources API as second resort
    try:
        from importlib.resources import files

        for name in candidates:
            ref = files("extensions.orchestrator.templates") / name  # type: ignore[arg-type]
            if hasattr(ref, "__fspath__"):
                p = Path(os.fspath(ref))
                if p.exists():
                    return p
            # MultiplexedPath
            if str(ref) and Path(str(ref)).exists():
                return Path(str(ref))
    except Exception:  # nosec B110 - best-effort template lookup fallback
        pass

    # Last resort: walk sys.modules for the orchestrator package
    import extensions.orchestrator as orch_mod

    base = Path(orch_mod.__file__).parent  # type: ignore[arg-type]
    for name in candidates:
        candidate = base / "templates" / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot locate template '{variant}' — your install may be corrupt. "
        f"Run 'clawcodex orchestrator workflow list-templates' to see available variants."
    )


def _available_templates() -> dict[str, str]:
    """Return {variant_name: template_path} for all packaged templates."""
    import extensions.orchestrator.templates as tpl_mod

    templates: dict[str, str] = {}
    for p in tpl_mod.__path__:  # type: ignore[attr-defined]
        for f in Path(p).glob("*.template.md"):
            name = f.stem  # e.g. "workflow" or "workflow-local"
            templates[name] = str(f)
        for f in Path(p).glob("*.yaml.template"):
            name = f.stem  # e.g. "workflow.yaml"
            templates[name] = str(f)
    return templates


# ── Placeholder substitution ─────────────────────────────────────────


def _prompt(label: str, default: str = "", secret: bool = False) -> str:
    """Prompt interactively, or return *default* when stdin is not a TTY."""
    if not sys.stdin.isatty():
        return default
    try:
        if secret:
            import getpass

            raw = getpass.getpass(f"  {label} [{default}]: ")
        else:
            raw = input(f"  {label} [{default}]: ")
        return raw.strip() or default
    except (EOFError, KeyboardInterrupt):
        return default


def _fill_placeholders(content: str, values: dict[str, str]) -> str:
    """Replace ``{{KEY}}`` and ``<KEY>`` placeholders with corresponding *values*.

    Handles two placeholder styles:
    - ``{{KEY}}`` — used by the remote-tracker workflow template
    - ``<KEY>``   — used by the local-tracker workflow template and issue cards
    """
    for key, val in values.items():
        content = content.replace("{{" + key + "}}", val)
        # Also replace ${{KEY}}_ENV style
        content = content.replace("${{" + key + "_ENV}}", val)
        # Also replace <KEY> style (used by workflow-local.template.md)
        content = content.replace("<" + key + ">", val)
    return content


# ── Parser ───────────────────────────────────────────────────────────


def add_workflow_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``workflow`` sub-subcommands (init | list-templates)."""
    parser = subparsers.add_parser(
        "workflow",
        help="Scaffold and manage orchestrator workflow files",
        description="Generate ``workflow.md`` from the packaged template, or list available template variants.",
    )
    wf_sub = parser.add_subparsers(
        dest="workflow_subcommand",
        required=True,
    )

    # --- workflow init ---
    init_parser = wf_sub.add_parser(
        "init",
        help="Generate workflow.md from template",
        description="Copy the packaged workflow template to the current directory "
        "and replace placeholders with your values. "
        "Use --workflow-yaml to also generate a declarative workflow.yaml "
        "for multi-stage DAG execution with quality gates and decision branches.",
    )
    init_parser.add_argument(
        "--template",
        "-t",
        default="workflow",
        metavar="VARIANT",
        help="Template variant: workflow (default, remote tracker), "
        "workflow-local (local file-based tracker). "
        "Run 'list-templates' to see all available variants.",
    )
    init_parser.add_argument(
        "--kind",
        "-k",
        default="github",
        metavar="TRACKER",
        help="Tracker kind: github, gitcode, gitee, linear, local (default: github)",
    )
    init_parser.add_argument(
        "--owner",
        "-o",
        default="",
        metavar="OWNER",
        help="Upstream repository owner (e.g. my-org)",
    )
    init_parser.add_argument(
        "--repo",
        "-r",
        default="",
        metavar="REPO",
        help="Repository name (e.g. my-project)",
    )
    init_parser.add_argument(
        "--endpoint",
        default="",
        metavar="URL",
        help="API endpoint for self-hosted instances",
    )
    init_parser.add_argument(
        "--assignee",
        default="",
        metavar="USER",
        help="Only process issues assigned to this user",
    )
    init_parser.add_argument(
        "--branch-prefix",
        default="clawcodex",
        metavar="PREFIX",
        help="Branch prefix for issue branches (default: clawcodex)",
    )
    init_parser.add_argument(
        "--workspace-root",
        default="/tmp/symphony_workspaces/myproject",  # nosec B108 - documented default workspace path
        metavar="PATH",
        help="Local workspace root path",
    )
    init_parser.add_argument(
        "--fork-owner",
        default="",
        metavar="OWNER",
        help="Fork owner for fork workflow (empty or same as owner = single-repo mode)",
    )
    init_parser.add_argument(
        "--output",
        "--out",
        default="workflow.md",
        metavar="FILE",
        help="Output file path (default: ./workflow.md)",
    )
    init_parser.add_argument(
        "--workflow-yaml",
        action="store_true",
        help="Also generate a declarative workflow.yaml for DeclarativeWorkflowEngine",
    )
    init_parser.add_argument(
        "--workflow-name",
        default="",
        metavar="NAME",
        help="Workflow name for workflow.yaml (default: {repo}-code-review)",
    )
    init_parser.add_argument(
        "--workflow-yaml-output",
        default="workflow.yaml",
        metavar="FILE",
        help="Output path for workflow.yaml (default: ./workflow.yaml)",
    )
    init_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts; use defaults for missing values",
    )

    # --- workflow list-templates ---
    wf_sub.add_parser(
        "list-templates",
        help="List available workflow template variants",
        description="Show all packaged template files and their locations.",
    )


# ── Dispatch ─────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate workflow subcommand."""
    cmd = args.workflow_subcommand
    if cmd == "init":
        return _run_init(args)
    elif cmd == "list-templates":
        return _run_list_templates(args)
    print(f"error: unknown workflow subcommand '{cmd}'", file=sys.stderr)
    return 2


# ── Implementations ──────────────────────────────────────────────────


def _run_init(args: argparse.Namespace) -> int:
    """Copy and fill the workflow template."""
    variant = args.template or "workflow"

    # Locate template
    try:
        tpl = _template_path(variant)
    except FileNotFoundError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    # Determine output path
    out = Path(args.output).expanduser().resolve()
    if out.exists():
        print(f"✗ {out} already exists — remove it first or use --output", file=sys.stderr)
        return 1

    interactive = sys.stdin.isatty() and not args.non_interactive

    # Gather values (flag → prompt → default)
    def val(flag_val: str, label: str, default: str = "") -> str:
        if flag_val:
            return flag_val
        if interactive:
            return _prompt(label, default)
        return default

    # Auto-infer --kind from --template when not explicitly provided
    _TEMPLATE_KIND_HINTS = {
        "workflow-local": "local",
        "workflow": "github",
    }
    kind_default = _TEMPLATE_KIND_HINTS.get(variant, "github")
    kind = val(args.kind, "Tracker kind (github/gitcode/gitee/linear/local)", kind_default)

    # Cross-validate --template and --kind
    _TEMPLATE_KIND_COMPAT = {
        "workflow-local": {"local"},
        "workflow": {"github", "gitcode", "gitee", "linear"},
    }
    compat = _TEMPLATE_KIND_COMPAT.get(variant)
    if compat and kind not in compat:
        print(
            f"✗ Template '{variant}' is not compatible with --kind '{kind}'.\n"
            f"  Expected one of: {', '.join(sorted(compat))}\n"
            f"  Hint: use --template workflow-local for local tracker, "
            f"or --template workflow for remote trackers.",
            file=sys.stderr,
        )
        return 1

    owner = val(args.owner, "Upstream repository owner")
    repo = val(args.repo, "Repository name")
    endpoint = val(args.endpoint, "API endpoint (leave blank for default)")
    assignee = val(args.assignee, "Issue assignee (leave blank for all)")
    branch_prefix = val(args.branch_prefix, "Branch prefix", "clawcodex")
    ws_root = val(args.workspace_root, "Workspace root", "/tmp/symphony_workspaces/myproject")  # nosec B108 - fallback default

    # Build clone_url
    clone_url = ""
    push_user = ""
    if kind in ("github", "gitcode", "gitee"):
        domains = {"github": "github.com", "gitcode": "gitcode.com", "gitee": "gitee.com"}
        domain = domains.get(kind, "github.com")
        if owner and repo:
            clone_url = f"https://{domain}/{owner}/{repo}.git"
            push_user = owner

    # Fork 工作流：--owner/--repo 是上游，--fork-owner 是 fork 方
    #   repo_clone_url    = fork 仓库 (clone/push)  → https://domain/fork_owner/repo.git
    #   upstream_clone_url = 上游仓库 (PR 目标)      → https://domain/owner/repo.git
    #   --fork-owner 为空或与 --owner 相同 → 两者相同，退化为单仓库模式
    upstream_clone_url = clone_url  # 上游 URL，始终由 --owner/--repo 拼接
    if kind in ("github", "gitcode", "gitee") and repo:
        domain = domains.get(kind, "github.com")
        fork_owner = args.fork_owner
        if not fork_owner and interactive:
            fork_owner = _prompt(
                "Fork owner (leave blank for single-repo mode)",
                "",
            ).strip()
        if fork_owner and fork_owner != owner:
            clone_url = f"https://{domain}/{fork_owner}/{repo}.git"
        else:
            clone_url = upstream_clone_url  # 单仓库模式：两者相同

    # Determine token env var
    token_env_map = {
        "github": "GITHUB_TOKEN",
        "gitcode": "GITCODE_TOKEN",
        "gitee": "GITEE_TOKEN",
        "linear": "LINEAR_API_KEY",
    }
    token_env = token_env_map.get(kind, "TRACKER_API_KEY")

    # Determine tracker endpoint
    endpoint_defaults = {
        "gitcode": "https://api.gitcode.com/api/v5",
        "gitee": "https://gitee.com/api/v5",
    }
    if not endpoint:
        endpoint = endpoint_defaults.get(kind, "")

    # Build substitution map (covers both {{KEY}} and <KEY> placeholder styles)
    values = {
        "TRACKER_KIND": kind,
        "TRACKER_ENDPOINT": endpoint,
        "REPO_OWNER": owner,
        "REPO_NAME": repo,
        "REPO_CLONE_URL": clone_url,
        "UPSTREAM_CLONE_URL": upstream_clone_url,
        "TRACKER_API_KEY_ENV": token_env,
        "REPO_ASSIGNEE": assignee,
        "BRANCH_PREFIX": branch_prefix,
        "WORKSPACE_ROOT": ws_root,
        "GIT_PUSH_USER": push_user,
        "GIT_PUSH_TOKEN_ENV": token_env,
        "REPO_URL": clone_url.removesuffix(".git") if clone_url else "",
        # Keys for <KEY> style placeholders in workflow-local.template.md
        "OWNER": owner,
        "REPO": repo,
        "ISSUES_PATH": val("", "Issues path (local tracker)", ".issues"),
        "REVIEW_REMOTE": val("", "Review remote name", "origin"),
        "REVIEW_PREFIX": val("", "Review branch prefix", "review"),
        "TEST_COMMAND": val("", "Test command (empty to skip)", ""),
    }

    # Read and fill template
    raw = tpl.read_text(encoding="utf-8")
    filled = _fill_placeholders(raw, values)

    # Write
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(filled, encoding="utf-8")

    print(f"✓ Generated {out}")

    # ── workflow.yaml generation ───────────────────────────────────
    yaml_out = None
    if args.workflow_yaml:
        yaml_out = Path(args.workflow_yaml_output).expanduser().resolve()
        if yaml_out.exists():
            print(
                f"✗ {yaml_out} already exists — remove it first or use --workflow-yaml-output",
                file=sys.stderr,
            )
            return 1

        yaml_tpl_path = _template_path("workflow.yaml")
        yaml_raw = yaml_tpl_path.read_text(encoding="utf-8")

        # Add workflow YAML specific values
        yaml_values = values.copy()
        yaml_values["WORKFLOW_NAME"] = val(
            args.workflow_name if hasattr(args, "workflow_name") else "",
            "Workflow name",
            f"{repo or 'project'}-code-review",
        )
        yaml_values["PROJECT_NAME"] = repo or owner or "project"
        yaml_values["PROVIDER"] = "anthropic"
        yaml_values["MODEL"] = "claude-sonnet-4-20250514"

        yaml_filled = _fill_placeholders(yaml_raw, yaml_values)
        yaml_out.parent.mkdir(parents=True, exist_ok=True)
        yaml_out.write_text(yaml_filled, encoding="utf-8")
        print(f"✓ Generated {yaml_out}")

    print()
    print("  Next steps:")
    print(f"    1. Edit {out.name} — check every placeholder was replaced")
    if yaml_out:
        print(f"    2. Edit {yaml_out.name} — customize workflow stages for your project")
        print(f"    3. Set the required env var: export {token_env}=<your-token>")
        print(
            f"    4. Start: clawcodex orchestrator server start --workflow {out.name} --workflow-yaml {yaml_out.name}"
        )
    else:
        print(f"    2. Set the required env var: export {token_env}=<your-token>")
        print(f"    3. Start: clawcodex orchestrator server start --workflow {out.name}")
    print()
    if interactive:
        print("  Hint: re-run with --non-interactive and CLI flags to skip prompts.")
    return 0


def _run_list_templates(args: argparse.Namespace) -> int:
    """List all available template files."""
    try:
        templates = _available_templates()
    except Exception as exc:
        print(f"✗ Cannot list templates: {exc}", file=sys.stderr)
        return 1

    print("Available workflow templates:")
    print()
    for name, path in sorted(templates.items()):
        print(f"  {name:30s}  {path}")
    print()
    print("Usage:  clawcodex orchestrator workflow init")
    return 0
