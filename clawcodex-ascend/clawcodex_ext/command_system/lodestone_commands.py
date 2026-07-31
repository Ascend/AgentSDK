"""F-97 LODESTONE — ``/link`` slash commands.

Surface area:

*   ``/link parse <text>``          — show structured anchors
*   ``/link resolve <text>``        — show resolved targets + URLs
*   ``/link open <text>``           — invoke the OS default handler
*   ``/link config <key>=<value>``  — mutates ``LodestoneConfig`` (persistable)
*   ``/link status``                — current config + probes
*   ``/link targets list``          — registered targets
*   ``/link targets test <id> <path:line>``     — verify template
*   ``/link targets register …``    — install a custom target
*   ``/link targets unregister <id>``           — remove a target

The command is registered lazily via :func:`register_lodestone_commands`
so that an unrelated failing extension never breaks ``/help``.
"""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

from .engine import CommandContext
from .registry import CommandRegistry, get_command_registry
from .types import LocalCommand, LocalCommandResult

from clawcodex_ext.services.lodestone import (
    AnchorContext,
    AnchorTarget,
    AnchorTargetRegistry,
    LodestoneService,
    get_lodestone_service,
)
from clawcodex_ext.services.lodestone.config import save_config
from clawcodex_ext.services.lodestone.fingerprint import detect_workspace_fingerprint
from clawcodex_ext.services.lodestone.parser import AnchorParser
from clawcodex_ext.services.lodestone.renderer import OpenLaunchError, open_uri
from clawcodex_ext.services.lodestone.resolver import probe_editor_from_env

log = logging.getLogger(__name__)

__all__ = [
    "LODESTONE_COMMAND",
    "register_lodestone_commands",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service(context: CommandContext) -> LodestoneService:
    """Pick a service instance — tests may attach one on the context."""
    return getattr(context, "lodestone_service", None) or get_lodestone_service()


def _ctx(context: CommandContext) -> AnchorContext:
    """Build an :class:`AnchorContext` from the active command context."""
    cfg = _service(context).config
    workspace_root = getattr(context, "workspace_root", None)
    if workspace_root is None:
        workspace_root = getattr(context, "cwd", None)
    fingerprint = None
    branch = None
    remote_url = None
    if workspace_root:
        with suppress(Exception):
            fingerprint = detect_workspace_fingerprint(workspace_root, use_cache=False)
            branch = fingerprint.default_branch
            remote_url = fingerprint.primary_remote_url
    return AnchorContext(
        workspace_root=Path(workspace_root).resolve() if workspace_root else None,
        session_id=getattr(context, "session_id", None),
        config=cfg,
        remote_url=remote_url,
        branch=branch,
        env=None,
    )


def _workspace_root(context: CommandContext) -> Optional[Path]:
    root = getattr(context, "workspace_root", None) or getattr(context, "cwd", None)
    return Path(root).resolve() if root else None


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def _cmd_parse(args: str, context: CommandContext) -> LocalCommandResult:
    parser = AnchorParser()
    anchors = parser.parse(args.strip())
    if not anchors:
        return LocalCommandResult(type="text", value="(no anchors detected)")
    payload = [
        {
            "kind": a.kind,
            "raw": a.raw,
            "span": list(a.span) if a.span else None,
            "file_path": a.file_path,
            "line": a.line,
            "column": a.column,
            "end_line": a.end_line,
            "end_column": a.end_column,
            "symbol": a.symbol,
            "git_sha": a.git_sha,
            "tracker_key": list(a.tracker_key) if a.tracker_key else None,
            "url": a.url,
        }
        for a in anchors
    ]
    return LocalCommandResult(type="text", value=json.dumps(payload, ensure_ascii=False, indent=2))


def _cmd_resolve(args: str, context: CommandContext) -> LocalCommandResult:
    service = _service(context)
    out = service.resolve_text(args, sink="markdown", workspace_root=_workspace_root(context)) if False else None  # noqa  # pylint: disable=using-constant-test
    parser = AnchorParser()
    anchors = parser.parse(args.strip())
    if not anchors:
        return LocalCommandResult(type="text", value="(no anchors detected)")
    lines: list[str] = []
    for a in anchors:
        rendered = service.resolve_one(a, sink="markdown", workspace_root=_workspace_root(context))
        lines.append(
            f"{a.kind:>14}  {rendered.rendered}  target={rendered.target.target_id if rendered.target else '-'} fallback={rendered.fallback_reason or '-'}"
        )
    return LocalCommandResult(type="text", value="\n".join(lines))


def _cmd_open(args: str, context: CommandContext) -> LocalCommandResult:
    service = _service(context)
    parser = AnchorParser()
    args = args.strip()
    explicit_target: Optional[str] = None
    if "--target" in args:
        head, _, rest = args.partition("--target")
        rest = rest.strip()
        explicit_target = rest.split()[0] if rest else None
        args = head.strip()
    anchors = parser.parse(args)
    if not anchors:
        return LocalCommandResult(type="text", value="(no anchors detected in input)")
    errors: list[str] = []
    opened = 0
    for a in anchors:
        rendered = service.resolve_one(
            a,
            sink="text",
            workspace_root=_workspace_root(context),
            target_override=explicit_target,
        )
        url = rendered.rendered if rendered.is_anchor else None
        if not url:
            errors.append(f"{a.raw}: {rendered.fallback_reason}")
            continue
        try:
            open_uri(url)
            opened += 1
        except OpenLaunchError as exc:
            errors.append(f"{a.raw}: {exc}")
    msg = f"Opened {opened} anchor(s)."
    if errors:
        msg += "\nErrors:\n - " + "\n - ".join(errors)
    return LocalCommandResult(type="text", value=msg)


def _cmd_config(args: str, context: CommandContext) -> LocalCommandResult:
    svc = _service(context)
    args = args.strip()
    if not args:
        return _cmd_status(args, context)
    changes: dict[str, Any] = {}
    for piece in args.split():
        if "=" not in piece:
            continue
        k, _, v = piece.partition("=")
        k = k.strip()
        v = v.strip().strip("'\"")
        if k == "editor":
            changes["default_editor"] = v
            continue
        if k == "fallback":
            changes["fallback_editor"] = v
            continue
        if k == "renderer":
            if v not in {"text", "markdown", "osc8", "auto"}:
                return LocalCommandResult(type="text", value=f"invalid renderer: {v}")
            changes["renderer"] = v
            continue
        if k in ("enabled", "auto_remote"):
            changes[k] = v.lower() in {"1", "true", "on", "yes"}
            continue
        if k == "default_editor":
            changes["default_editor"] = v
            continue
        if k == "fallback_editor":
            changes["fallback_editor"] = v
            continue
        if k == "default_tracker_host":
            changes["default_tracker_host"] = v
            continue
        if k == "default_tracker_owner":
            cur = svc.config.default_tracker_repo or ("", "")
            changes["default_tracker_repo"] = (v, cur[1])
            continue
        if k == "default_tracker_repo":
            cur = svc.config.default_tracker_repo or ("", "")
            changes["default_tracker_repo"] = (cur[0], v)
            continue
        return LocalCommandResult(type="text", value=f"unknown config key: {k}")
    if not changes:
        return _cmd_status("", context)
    new_cfg = svc.update_config(**changes)
    try:
        save_config(new_cfg)
    except OSError as exc:
        return LocalCommandResult(type="text", value=f"updated in-memory; failed to persist: {exc}")
    return LocalCommandResult(type="text", value=f"config updated: {changes}")


def _cmd_status(_args: str, context: CommandContext) -> LocalCommandResult:
    svc = _service(context)
    cfg = svc.config
    editor = probe_editor_from_env()
    lines = [
        f"enabled: {cfg.enabled}",
        f"renderer: {cfg.renderer}",
        f"default_editor: {cfg.default_editor}",
        f"fallback_editor: {cfg.fallback_editor}",
        f"auto_remote: {cfg.auto_remote}",
        f"disabled_kinds: {list(cfg.disabled_kinds)}",
        f"default_tracker_host: {cfg.default_tracker_host}",
        f"default_tracker_repo: {cfg.default_tracker_repo}",
        f"env-editor-hint: {editor or '-'}",
    ]
    return LocalCommandResult(type="text", value="\n".join(lines))


def _cmd_targets_list(_args: str, context: CommandContext) -> LocalCommandResult:
    registry: AnchorTargetRegistry = _service(context).registry
    lines = []
    for t in registry.list():
        lines.append(f"{t.target_id:>22}  kind={t.kind:>14}  remote={t.is_remote}  template={t.template}")
    return LocalCommandResult(type="text", value="\n".join(lines) or "(no targets registered)")


def _cmd_targets_test(args: str, context: CommandContext) -> LocalCommandResult:
    registry: AnchorTargetRegistry = _service(context).registry
    parts = args.strip().split(maxsplit=1)
    if len(parts) < 2:
        return LocalCommandResult(type="text", value="usage: /link targets test <target-id> <anchor>")
    target_id, anchor_text = parts
    target = registry.get(target_id)
    if target is None:
        return LocalCommandResult(type="text", value=f"unknown target: {target_id}")
    parser = AnchorParser()
    anchors = parser.parse(anchor_text)
    if not anchors:
        return LocalCommandResult(type="text", value="(no anchor detected)")
    a = anchors[0]
    if a.kind != target.kind:
        return LocalCommandResult(
            type="text",
            value=f"target {target_id} expects kind={target.kind}; input kind={a.kind}",
        )
    rendered = _service(context).resolve_one(
        a,
        sink="markdown",
        workspace_root=_workspace_root(context),
        target_override=target_id,
    )
    return LocalCommandResult(type="text", value=f"would render: {rendered.rendered}")


def _cmd_targets_register(args: str, context: CommandContext) -> LocalCommandResult:
    svc = _service(context)
    registry = svc.registry
    parts = args.split()
    if len(parts) < 3:
        return LocalCommandResult(
            type="text",
            value="usage: /link targets register <id> <kind> <template> [--force]",
        )
    target_id, kind, *rest = parts
    template_parts: list[str] = []
    force = False
    for p in rest:
        if p == "--force":
            force = True
            break
        template_parts.append(p)
    template = " ".join(template_parts)
    if kind not in ("file_path", "function_ref", "git_blob", "git_commit", "tracker_issue", "url"):
        return LocalCommandResult(type="text", value=f"invalid kind: {kind}")
    try:
        target = AnchorTarget(
            kind=kind,  # type: ignore[arg-type]
            target_id=target_id,
            template=template,
        )
        registry.register(target, overwrite=force)
    except ValueError as exc:
        return LocalCommandResult(type="text", value=f"register failed: {exc}")
    # Persist via config
    persisted: list[AnchorTarget] = list(svc.config.custom_targets) + [target]
    svc.update_config(custom_targets=tuple(persisted))
    save_config(svc.config)
    return LocalCommandResult(type="text", value=f"registered {target_id}")


def _cmd_targets_unregister(args: str, context: CommandContext) -> LocalCommandResult:
    target_id = args.strip().split()[0] if args.strip() else ""
    if not target_id:
        return LocalCommandResult(type="text", value="usage: /link targets unregister <id>")
    svc = _service(context)
    removed = svc.registry.unregister(target_id)
    if removed:
        remaining = tuple(t for t in svc.config.custom_targets if t.target_id != target_id)
        svc.update_config(custom_targets=remaining)
        try:
            save_config(svc.config)
        except OSError:  # nosec B110
            pass
        return LocalCommandResult(type="text", value=f"unregistered {target_id}")
    return LocalCommandResult(type="text", value=f"target {target_id} not found")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _call(args: str, context: CommandContext) -> LocalCommandResult:
    body = (args or "").strip()
    if not body:
        return _cmd_status("", context)
    parts = body.split(maxsplit=1)
    head = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if head == "parse":
        return _cmd_parse(rest, context)
    if head == "resolve":
        return _cmd_resolve(rest, context)
    if head == "open":
        return _cmd_open(rest, context)
    if head == "config":
        return _cmd_config(rest, context)
    if head == "status":
        return _cmd_status(rest, context)
    if head == "targets":
        subparts = rest.split(maxsplit=1)
        sub = subparts[0] if subparts else "list"
        subargs = subparts[1] if len(subparts) > 1 else ""
        if sub == "list":
            return _cmd_targets_list(subargs, context)
        if sub == "test":
            return _cmd_targets_test(subargs, context)
        if sub == "register":
            return _cmd_targets_register(subargs, context)
        if sub == "unregister":
            return _cmd_targets_unregister(subargs, context)
        return LocalCommandResult(
            type="text",
            value=f"unknown targets sub-command: {sub}; expected list | test | register | unregister",
        )
    if head in ("help", "-h", "--help"):
        return LocalCommandResult(
            type="text",
            value=(
                "/link parse <text>\n"
                "/link resolve <text>\n"
                "/link open <text> [--target <id>]\n"
                "/link config [key=value ...]\n"
                "/link status\n"
                "/link targets list|test|register|unregister ..."
            ),
        )
    # Bare anchor — treat as ``resolve``.
    return _cmd_resolve(body, context)


LODESTONE_COMMAND = LocalCommand(
    name="link",
    description="Resolve and open code / git / tracker anchors as deep links.",
    argument_hint="[parse|resolve|open|config|status|targets] ...",
    aliases=("lodestone",),
    supports_non_interactive=True,
    run_in_thread=False,
)
LODESTONE_COMMAND.set_call(_call)


def register_lodestone_commands(registry: Optional[CommandRegistry] = None) -> None:
    """Register ``/link`` on the given registry (or the global one)."""
    target = registry if registry is not None else get_command_registry()
    if target.has(LODESTONE_COMMAND.name):
        return
    target.register(LODESTONE_COMMAND)


# Silence unused-import lint for re-exports
_ = (LocalCommand, LocalCommandResult)
