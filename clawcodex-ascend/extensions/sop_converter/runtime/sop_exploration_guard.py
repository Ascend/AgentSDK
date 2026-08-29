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

"""Runtime guards blocking SDK *tool-discovery* exploration in SOP bundle mode.

Workspace config lookup (``spec.yaml``, ``*.yaml``), SDK runtime-data home
(``SOP_SDK_RUNTIME_HOME`` / default ``~/.openjiuwen`` profile), and reads
under the bundle manifest ``sdk_source_dir`` remain allowed.  Only searches
that substitute Skill → ToolSearch → call are blocked.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from extensions.sop_converter import bundle_context as _bundle_context
except ImportError:
    _bundle_context = None

logger = logging.getLogger(__name__)

_EXPLORATION_TOOLS = frozenset({"Grep", "Glob", "Read", "Bash"})

# ToolSearch hint syntax — searching for these means skipping ToolSearch.
# SDK tool names are matched SDK-agnostically against the active bundle's
# registered tool/skill names (see _registered_name_markers).
_TOOL_DISCOVERY_RE = re.compile(
    r"\bselect:[a-z0-9-]+\b",
    re.IGNORECASE,
)

# Workspace / runtime config the user may read before delegating.
_WORKSPACE_CONFIG_RE = re.compile(
    r"spec\.ya?ml|[^/\\]+\.ya?ml$|[^/\\]+\.json$|[^/\\]+\.toml$|"
    r"config\.|settings\.|\.clawcodex[/\\]",
    re.IGNORECASE,
)

# Bash text-search stages in a pipeline (find alone is not tool-hunting).
_BASH_TEXT_SEARCH_RE = re.compile(
    r"\b(grep|rg|ripgrep)\b|xargs\s+(grep|rg)|-exec\s+(grep|rg)",
    re.IGNORECASE,
)

_PATH_CANDIDATE_RE = re.compile(
    r"(/(?:[^\s'\";|&]+)|"
    r"[A-Za-z]:[/\\][^\s'\";|&]+)",
)

# Common test/fixture directory segments under a source tree (not SDK-specific).
_SDK_TEST_TREE_SEGMENT_RE = re.compile(
    r"(?:^|[/\\])(?:tests|test|testing|fixtures|__tests__)(?:[/\\]|$)",
    re.IGNORECASE,
)

# Config/fixture discovery signals when combined with test-tree paths.
_FIXTURE_CONFIG_HUNT_RE = re.compile(
    r"\.ya?ml\b|\.json\b|\.toml\b|\bspec\.|config\.|fixture|"
    r"\*\.*\.(?:ya?ml|yml|json)|"
    r"find\b.*\.(?:ya?ml|yml|json)",
    re.IGNORECASE,
)

_DIAGNOSTIC_PATH_MARKERS = (
    "agent-tools/",
    "agent-tools\\",
    "/agent-tools/",
    "\\agent-tools\\",
)


def _current_agent_type(context: Any) -> str | None:
    agent_type = getattr(context, "agent_type", None)
    if isinstance(agent_type, str) and agent_type:
        return agent_type
    startup = getattr(context, "startup_agent", None)
    if startup is not None:
        st = getattr(startup, "agent_type", None)
        if isinstance(st, str) and st:
            return st
    return None


def _is_overview_agent(agent_type: str | None) -> bool:
    if not agent_type:
        return False
    return agent_type == "clawcodex-overview" or agent_type.endswith("-overview")


def _is_domain_agent(agent_type: str | None) -> bool:
    return bool(agent_type and agent_type.endswith("-agent") and not _is_overview_agent(agent_type))


def _block_name(block: Any) -> str | None:
    if isinstance(block, dict):
        name = block.get("name")
        return name if isinstance(name, str) else None
    name = getattr(block, "name", None)
    return name if isinstance(name, str) else None


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        btype = block.get("type")
        return btype if isinstance(btype, str) else None
    btype = getattr(block, "type", None)
    return btype if isinstance(btype, str) else None


def _block_id(block: Any) -> str | None:
    if isinstance(block, dict):
        tid = block.get("id")
        return tid if isinstance(tid, str) else None
    tid = getattr(block, "id", None)
    return tid if isinstance(tid, str) else None


def _result_tool_use_id(block: Any) -> str | None:
    if isinstance(block, dict):
        tid = block.get("tool_use_id")
        return tid if isinstance(tid, str) else None
    tid = getattr(block, "tool_use_id", None)
    return tid if isinstance(tid, str) else None


def _result_is_error(block: Any) -> bool:
    if isinstance(block, dict):
        return bool(block.get("is_error"))
    return bool(getattr(block, "is_error", False))


def _skill_invoked(messages: list[Any] | None) -> bool:
    for msg in messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            if _block_type(block) == "tool_use" and _block_name(block) == "Skill":
                return True
    return False


def _sdk_tool_call_failed(messages: list[Any] | None) -> bool:
    registered = {n.lower() for n in _registered_bundle_names()}
    sdk_use_ids: set[str] = set()
    for msg in messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            btype = _block_type(block)
            if btype == "tool_use":
                name = _block_name(block) or ""
                tid = _block_id(block)
                if tid and name.lower() in registered:
                    sdk_use_ids.add(tid)
            elif btype == "tool_result":
                tid = _result_tool_use_id(block)
                if tid and tid in sdk_use_ids and _result_is_error(block):
                    return True
    return False


def _path_text(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name == "Read":
        return str(tool_input.get("file_path") or "")
    if tool_name == "Glob":
        parts = [str(tool_input.get("path") or ""), str(tool_input.get("pattern") or "")]
        return " ".join(parts)
    if tool_name == "Grep":
        parts = [
            str(tool_input.get("path") or ""),
            str(tool_input.get("glob") or ""),
            str(tool_input.get("pattern") or ""),
        ]
        return " ".join(parts)
    if tool_name == "Bash":
        return str(tool_input.get("command") or "")
    return ""


def _normalized_path_text(text: str) -> str:
    return text.replace("\\", "/")


def _wsl_to_windows_path(text: str) -> str | None:
    """Map ``/mnt/d/projects/...`` → ``D:/projects/...`` on Windows."""
    if sys.platform != "win32":
        return None
    norm = _normalized_path_text(text).lower()
    match = re.match(r"^/mnt/([a-z])/(.+)$", norm)
    if not match:
        return None
    return f"{match.group(1).upper()}:/{match.group(2)}"


def _fix_windows_mnt_resolution(text: str) -> str:
    """Undo ``Path('/mnt/d/...').resolve()`` → ``D:/mnt/d/...`` on Windows."""
    if sys.platform != "win32":
        return _normalized_path_text(text)
    norm = _normalized_path_text(text)
    match = re.match(r"^([A-Za-z]):/mnt/[a-z]/(.+)$", norm, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()}:/{match.group(2)}"
    return norm


def _normalize_sdk_source_dir(sdk: Path) -> Path:
    raw = _normalized_path_text(str(sdk))
    wsl = _wsl_to_windows_path(raw)
    if wsl is not None:
        return Path(wsl)
    fixed = _fix_windows_mnt_resolution(raw)
    if fixed != raw:
        return Path(fixed)
    try:
        return sdk.expanduser().resolve()
    except OSError:
        return sdk


def _sdk_root_match_prefixes(sdk_root: Path) -> tuple[str, ...]:
    """Normalized lowercase path prefixes that identify the authorized SDK root."""
    prefixes: set[str] = set()
    for candidate in (sdk_root, _normalize_sdk_source_dir(sdk_root)):
        norm = _fix_windows_mnt_resolution(_normalized_path_text(str(candidate))).lower().rstrip("/")
        if norm:
            prefixes.add(norm)
        wsl = _wsl_to_windows_path(norm)
        if wsl:
            prefixes.add(wsl.lower().rstrip("/"))
    return tuple(sorted(prefixes, key=len, reverse=True))


def _sdk_root_dir_names(sdk_root: Path) -> tuple[str, ...]:
    """Case-folded directory names that mark a raw SDK checkout path.

    Derived from the manifest ``sdk_source_dir`` so the guard stays
    SDK-agnostic: the SDK root's own directory name (in any of its
    normalized forms) is the name under which a raw checkout of that SDK
    would appear inside the workspace.
    """
    names: set[str] = set()
    for candidate in (sdk_root, _normalize_sdk_source_dir(sdk_root)):
        name = candidate.name.strip("/\\")
        if name:
            names.add(name.lower())
    return tuple(sorted(names, key=len, reverse=True))


def _normalize_candidate_path(raw: str) -> Path:
    norm = _normalized_path_text(raw)
    wsl = _wsl_to_windows_path(norm)
    if wsl is not None:
        return Path(wsl)
    fixed = _fix_windows_mnt_resolution(norm)
    if fixed != norm:
        return Path(fixed)
    return Path(raw)


def _active_bundle(context: Any = None) -> Any | None:
    """Active ``BundleContext`` from the tool context or the module registry."""
    if context is not None:
        bundle = getattr(context, "bundle_context", None)
        if bundle is not None:
            return bundle
    if _bundle_context is not None:
        return _bundle_context.get_active_bundle()
    return None


def _resolve_sdk_source_dir(context: Any) -> Path | None:
    bundle = _active_bundle(context)
    if bundle is None:
        return None
    sdk = getattr(bundle, "sdk_source_dir", None)
    if sdk is None:
        return None
    try:
        resolved = _normalize_sdk_source_dir(Path(sdk))
    except OSError:
        return None
    return resolved


def _registered_bundle_names(context: Any = None) -> frozenset[str]:
    """Tool/skill names registered for the active bundle (SDK-agnostic).

    Mirrors the ``BundleContext.tool_names`` / ``skill_names`` contract used
    by ``sop_routing._prompt_names_bundle_registered``; empty when no bundle
    or no registered names (callers fail open).
    """
    bundle = _active_bundle(context)
    if bundle is None:
        return frozenset()
    names: set[str] = set()
    for attr in ("tool_names", "skill_names"):
        for name in getattr(bundle, attr, None) or ():
            if isinstance(name, str) and name:
                names.add(name)
    return frozenset(names)


def _registered_name_markers(names: frozenset[str]) -> re.Pattern:
    """Word-bounded markers for registered tool/skill names.

    Markers are the full registered names plus their ``>=2``-segment kebab
    tails of ``>=6`` chars, so greps for a memorable tail (e.g.
    ``team-memory-dir``) are caught without hardcoding any SDK name prefix.
    """
    markers: list[str] = []
    for name in names:
        segs = [s for s in name.lower().split("-") if s]
        for i in range(len(segs) - 1):
            tail = "-".join(segs[i:])
            if len(tail) >= 6:
                markers.append(tail)
    if not markers:
        return re.compile(r"(?!x)x")
    return re.compile(
        "|".join(rf"\b{re.escape(m)}\b" for m in dict.fromkeys(markers)),
        re.IGNORECASE,
    )


def _runtime_data_roots() -> tuple[Path, ...]:
    """Configured SDK runtime-data home dirs.

    Env-driven via ``SOP_SDK_RUNTIME_HOME``; the Jiuwen profile
    (``~/.openjiuwen``) remains the default for existing deployments.
    """
    roots: list[Path] = []
    raw = os.environ.get("SOP_SDK_RUNTIME_HOME")
    if raw:
        roots.append(Path(raw).expanduser())
    roots.append(Path.home() / ".openjiuwen")
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            out.append(root)
    return tuple(out)


def _runtime_root_prefixes(root: Path) -> tuple[str, ...]:
    """Normalized lowercase path prefixes that identify a runtime-data root."""
    prefixes: set[str] = set()
    for candidate in (root, _normalize_sdk_source_dir(root)):
        norm = _fix_windows_mnt_resolution(_normalized_path_text(str(candidate))).lower().rstrip("/")
        if norm:
            prefixes.add(norm)
        wsl = _wsl_to_windows_path(norm)
        if wsl:
            prefixes.add(wsl.lower().rstrip("/"))
    return tuple(sorted(prefixes, key=len, reverse=True))


def _looks_like_runtime_data_path(text: str) -> bool:
    """True when *text* references SDK runtime-data dirs (allowed, not hunting).

    Matches any configured runtime-data root, or the default Jiuwen profile
    directory ``.openjiuwen`` under any parent (e.g. ``~/.openjiuwen``,
    ``/root/.openjiuwen``).  Sub-directory literals of a specific SDK are not
    hardcoded — anything under a runtime root counts as runtime data.
    """
    norm = _fix_windows_mnt_resolution(_normalized_path_text(text)).lower()
    for root in _runtime_data_roots():
        for prefix in _runtime_root_prefixes(root):
            if prefix and prefix in norm:
                return True
    if re.search(r"(?:^|[/\\])\.openjiuwen(?:[/\\]|$)", norm):
        return True
    return False


def _path_is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _candidate_paths_from_text(text: str) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for match in _PATH_CANDIDATE_RE.finditer(text):
        raw = match.group(1).strip().strip("'\"")
        raw = raw.rstrip("/\\")
        if not raw or raw in seen:
            continue
        seen.add(raw)
        try:
            candidates.append(_normalize_candidate_path(raw))
        except OSError:
            continue
    return candidates


def _path_is_under_sdk_root(path: Path, sdk_root: Path) -> bool:
    roots = (_normalize_sdk_source_dir(sdk_root), sdk_root)
    for root in roots:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def _text_targets_sdk_source(text: str, sdk_root: Path) -> bool:
    for candidate in _candidate_paths_from_text(text):
        if _path_is_under_sdk_root(candidate, sdk_root):
            return True
    text_norm = _fix_windows_mnt_resolution(_normalized_path_text(text)).lower()
    for prefix in _sdk_root_match_prefixes(sdk_root):
        if prefix and text_norm.startswith(prefix):
            return True
        if prefix and prefix in text_norm:
            return True
    return False


def _looks_like_sdk_test_tree_fixture_hunt(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    sdk_root: Path | None,
) -> bool:
    """Block config/fixture hunting under generic ``tests/`` / ``fixtures/`` in SDK source."""
    if sdk_root is None:
        return False
    text = _path_text(tool_name, tool_input)
    if not text.strip() or not _text_targets_sdk_source(text, sdk_root):
        return False
    norm = _normalized_path_text(text)
    if not _SDK_TEST_TREE_SEGMENT_RE.search(norm):
        return False
    if _FIXTURE_CONFIG_HUNT_RE.search(norm):
        return True
    if tool_name == "Bash" and re.search(r"\bfind\b", text, re.IGNORECASE):
        return True
    if tool_name == "Glob" and re.search(r"\*\.(?:ya?ml|yml|json)", text, re.IGNORECASE):
        return True
    return False


def _sdk_test_tree_block_message() -> str:
    return (
        "SOP bundle mode: do not search SDK source-tree tests/fixtures directories for "
        "user config or launcher scripts. Use workspace user config (spec.yaml, etc.) and "
        "follow 「交互式终端停损」— give the user a real-terminal command from the task guide / "
        "ToolSearch tool / wrapper _SOURCE_DIR public API."
    )


def _is_diagnostic_path(path_text: str) -> bool:
    lowered = _normalized_path_text(path_text).lower()
    return any(marker.replace("\\", "/") in lowered for marker in _DIAGNOSTIC_PATH_MARKERS)


def _looks_like_workspace_config_access(tool_name: str, tool_input: dict[str, Any]) -> bool:
    text = _path_text(tool_name, tool_input)
    if not text.strip():
        return False
    if tool_name == "Bash" and _bash_looks_like_tool_hunt(text):
        return False
    norm = _normalized_path_text(text)
    if _WORKSPACE_CONFIG_RE.search(norm):
        return True
    if tool_name == "Bash" and _looks_like_runtime_data_bash(text):
        return True
    if tool_name in {"Glob", "Read", "Grep"} and _looks_like_runtime_data_path(text):
        return True
    return False


def _looks_like_runtime_data_bash(cmd: str) -> bool:
    """``ls`` / ``find -type f`` under a SDK runtime-data home — data, not hunting."""
    if not _looks_like_runtime_data_path(cmd):
        return False
    if re.search(r"\bls\b", cmd, re.IGNORECASE):
        return True
    if re.search(r"\bfind\b", cmd, re.IGNORECASE) and not _BASH_TEXT_SEARCH_RE.search(cmd):
        return True
    return False


def _bash_looks_like_tool_hunt(cmd: str) -> bool:
    if _TOOL_DISCOVERY_RE.search(cmd):
        return True
    if "agent-tools" in cmd.lower():
        return True
    if _registered_name_markers(_registered_bundle_names()).search(cmd):
        return True
    if not _BASH_TEXT_SEARCH_RE.search(cmd):
        return False
    return False


def _looks_like_wrong_workspace_sdk_path(
    text: str,
    context: Any,
    sdk_root: Path | None,
) -> bool:
    """Block raw SDK-checkout paths under the workspace that are not the
    manifest ``sdk_source_dir`` root.

    SDK-agnostic: the name under which a raw checkout of the SDK would sit
    inside the workspace (``<workspace>/<sdk_root_dir_name>/...``) is derived
    from the bundle manifest's ``sdk_source_dir`` instead of being hardcoded
    to a specific SDK.  Reads under the authorized SDK root remain allowed
    (checked first).  Without a manifest SDK root nothing can be derived, so
    the heuristic fails open.
    """
    if sdk_root is None:
        return False
    root_names = _sdk_root_dir_names(sdk_root)
    if not root_names:
        return False
    if _text_targets_sdk_source(text, sdk_root):
        return False

    # A raw checkout path: "<sep><sdk_root_dir_name>/<something>".
    segment_re = re.compile(
        r"(?:^|[/\\])(" + "|".join(re.escape(n) for n in root_names) + r")[/\\][^/\\]+",
        re.IGNORECASE,
    )

    workspace = getattr(context, "workspace_root", None) or getattr(context, "cwd", None)
    if workspace is not None:
        try:
            ws = Path(str(workspace)).expanduser().resolve()
            for candidate in _candidate_paths_from_text(text):
                if _path_is_under_root(candidate, ws) and segment_re.search(_normalized_path_text(str(candidate))):
                    return True
        except OSError:
            pass

    if segment_re.search(_normalized_path_text(text)):
        return True
    return False


def _looks_like_authorized_sdk_source_access(
    tool_name: str,
    tool_input: dict[str, Any],
    sdk_root: Path | None,
) -> bool:
    """Allow Read/Glob/Grep/ls under bundle ``sdk_source_dir`` for source understanding."""
    if sdk_root is None:
        return False
    text = _path_text(tool_name, tool_input)
    if not text.strip() or not _text_targets_sdk_source(text, sdk_root):
        return False

    if _looks_like_sdk_test_tree_fixture_hunt(tool_name, tool_input, sdk_root=sdk_root):
        return False

    if tool_name == "Grep":
        pattern = str(tool_input.get("pattern") or "")
        names = _registered_bundle_names()
        return not (_TOOL_DISCOVERY_RE.search(pattern) or _registered_name_markers(names).search(pattern))

    if tool_name == "Bash":
        return not _bash_looks_like_tool_hunt(text)

    return tool_name in {"Read", "Glob"}


def _looks_like_sdk_tool_discovery(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    context: Any,
    sdk_root: Path | None,
) -> bool:
    """True when exploration is trying to locate a deferred SDK tool/API by name."""
    text = _path_text(tool_name, tool_input)
    if not text.strip():
        return False
    if tool_name == "Bash":
        if _looks_like_runtime_data_bash(text):
            return False
        return _bash_looks_like_tool_hunt(text)
    if _TOOL_DISCOVERY_RE.search(text):
        return True
    if _registered_name_markers(_registered_bundle_names(context)).search(text):
        return True
    if _looks_like_wrong_workspace_sdk_path(text, context, sdk_root):
        return True
    return False


def _overview_block_message(tool_name: str, agent_definitions: list[Any]) -> str:
    domain_agents = [
        getattr(a, "agent_type", "")
        for a in agent_definitions
        if isinstance(getattr(a, "agent_type", ""), str)
        and str(getattr(a, "agent_type", "")).endswith("-agent")
        and getattr(a, "agent_type", "") != "clawcodex-overview"
    ]
    domain_agents = sorted(set(domain_agents))
    examples = ", ".join(f'Agent(subagent_type="{n}", prompt="...")' for n in domain_agents[:3])
    if len(domain_agents) > 3:
        examples += ", ..."
    return (
        f"SOP bundle mode: do not use {tool_name} to hunt SDK tool names/schemas — "
        f"use Skill → ToolSearch → SDK tool, or delegate: {examples}. "
        f"Workspace config (spec.yaml), runtime data under the SDK runtime home "
        f"(e.g. ~/.openjiuwen), and reads under the bundle SDK source root are "
        f"still allowed."
    )


def _domain_block_message(tool_name: str, *, need_skill: bool) -> str:
    if need_skill:
        return (
            f"SOP bundle mode: for SDK API calls, call Skill(...) first, then ToolSearch. "
            f"Do not use {tool_name} to search for kebab tool names. "
            f"Reading workspace config (spec.yaml) or SDK source under sdk_source_dir is allowed."
        )
    return (
        f"SOP bundle mode: ToolSearch already identifies the SDK tool — call it directly. "
        f"Do not use {tool_name} to look up tool definitions. "
        f"If the SDK tool failed, follow the limited diagnostic steps in your system prompt."
    )


def check_bundle_source_exploration(
    tool_name: str,
    tool_input: dict[str, Any] | None,
    context: Any,
    *,
    agent_definitions: list[Any] | None = None,
) -> str | None:
    """Return an error message when exploration should be blocked."""
    if _bundle_context is None or _bundle_context.get_active_bundle() is None:
        return None

    if tool_name not in _EXPLORATION_TOOLS:
        return None

    tool_input = tool_input or {}
    sdk_root = _resolve_sdk_source_dir(context)

    if _looks_like_sdk_test_tree_fixture_hunt(tool_name, tool_input, sdk_root=sdk_root):
        return _sdk_test_tree_block_message()

    if _looks_like_workspace_config_access(tool_name, tool_input):
        return None

    if _looks_like_authorized_sdk_source_access(tool_name, tool_input, sdk_root):
        return None

    agent_type = _current_agent_type(context)
    messages = getattr(context, "messages", None) or []

    if _is_diagnostic_path(_path_text(tool_name, tool_input)):
        if _sdk_tool_call_failed(messages):
            return None
        if not _is_overview_agent(agent_type):
            return (
                "SOP bundle mode: read agent-tools specs only after an SDK tool call fails. "
                "Complete Skill → ToolSearch → SDK tool first."
            )

    discovery = _looks_like_sdk_tool_discovery(
        tool_name,
        tool_input,
        context=context,
        sdk_root=sdk_root,
    )
    if not discovery:
        return None

    if _is_overview_agent(agent_type):
        defs = agent_definitions
        if defs is None:
            defs = _load_agent_definitions(context)
        return _overview_block_message(tool_name, defs or [])

    if _is_domain_agent(agent_type):
        if not _skill_invoked(messages):
            return _domain_block_message(tool_name, need_skill=True)
        if not _sdk_tool_call_failed(messages):
            return _domain_block_message(tool_name, need_skill=False)

    return None


def _load_agent_definitions(context: Any) -> list[Any]:
    try:
        from ..adapters import DEFAULTS

        agents = list(DEFAULTS.agent_loader())
        ad_override = getattr(context, "_agent_dir_override", None)
        if ad_override is not None:
            from clawcodex_ext.agent.load_agents_dir import (
                get_agent_definitions_with_overrides,
            )

            extra = get_agent_definitions_with_overrides(str(ad_override))
            extra_types = {a.agent_type for a in extra}
            agents = [a for a in agents if a.agent_type not in extra_types]
            agents.extend(extra)
        return agents
    except ImportError:
        # Migration-period missing module — acceptable; fall back to an empty
        # agent list and keep the guard functional.
        logger.debug("agent_loader unavailable (ImportError); using empty agent list", exc_info=True)
        return []
    except Exception as exc:
        # Do not silently swallow real failures.  Re-raising is not an option
        # here: permission-hook callers (Read/Grep/Glob/Bash) only guard
        # ImportError, so a raise would break every exploration tool in SOP
        # bundle mode.  Log at error level with traceback and degrade.
        logger.error("agent_loader failed: %s", exc, exc_info=True)
        return []


def sop_exploration_permission_check(
    tool_name: str,
    tool_input: dict[str, Any] | None,
    context: Any,
):
    """Permission hook helper — deny when SOP exploration guard fires."""
    from clawcodex_ext.permissions.types import (
        PermissionDenyDecision,
        PermissionPassthroughResult,
    )

    message = check_bundle_source_exploration(tool_name, tool_input, context)
    if message:
        return PermissionDenyDecision(message=message)
    return PermissionPassthroughResult()
