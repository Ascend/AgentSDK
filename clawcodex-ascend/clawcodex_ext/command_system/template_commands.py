"""F-95 /template command family."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from clawcodex_ext.command_system.engine import CommandContext
from clawcodex_ext.command_system.registry import CommandRegistry
from clawcodex_ext.command_system.types import LocalCommand, LocalCommandResult
from clawcodex_ext.services.templates import (
    TemplateCatalogue,
    TemplateGenerator,
    TemplateKind,
    TemplateRegistry,
    TemplateRenderer,
    bootstrap_default_templates,
    get_default_template_registry,
    parse_template_file,
)
from clawcodex_ext.services.templates.exceptions import TemplatesError
from clawcodex_ext.services.templates.models import get_manifest


def _usage() -> str:
    return "\n".join(
        [
            "Usage:",
            "  /template list [--kind agent|skill|workflow|prompt|issue|generic] [--source SOURCE]",
            "  /template show <template-id>",
            "  /template search <query>",
            "  /template preview <template-id> --var key=value ...",
            "  /template render <template-id> --var key=value ... [--output path] [--overwrite]",
            "  /template create skill --name <name> [--description text] [--overwrite]",
            "  /template install <path>",
            "  /template validate <file>",
        ]
    )


def _parts(args: str) -> list[str]:
    try:
        return shlex.split(args or "")
    except ValueError as exc:
        raise ValueError(f"invalid arguments: {exc}") from exc


def _take_option(parts: list[str], name: str) -> str | None:
    if name not in parts:
        return None
    index = parts.index(name)
    if index + 1 >= len(parts):
        raise ValueError(f"{name} requires a value")
    value = parts[index + 1]
    del parts[index : index + 2]
    return value


def _flag(parts: list[str], name: str) -> bool:
    if name not in parts:
        return False
    parts.remove(name)
    return True


def _vars(parts: list[str]) -> dict[str, object]:
    values: dict[str, object] = {}
    index = 0
    while index < len(parts):
        if parts[index] != "--var":
            index += 1
            continue
        if index + 1 >= len(parts) or "=" not in parts[index + 1]:
            raise ValueError("--var requires key=value")
        key, value = parts[index + 1].split("=", 1)
        if not key:
            raise ValueError("--var requires a non-empty key")
        values[key] = value
        del parts[index : index + 2]
    return values


def _registry_for(context: CommandContext) -> TemplateRegistry:
    bootstrap_default_templates(cwd=context.cwd or context.workspace_root)
    return get_default_template_registry()


def _format_template(template) -> str:
    manifest = get_manifest(template)
    tags = ",".join(manifest.tags) if manifest.tags else "-"
    return f"{template.id:28} {manifest.kind:8} {template.source:9} {tags:24} {template.title}"


def _list(parts: list[str], context: CommandContext) -> str:
    kind_raw = _take_option(parts, "--kind")
    source = _take_option(parts, "--source")
    if parts:
        raise ValueError(f"unexpected arguments: {' '.join(parts)}")
    kind: TemplateKind | None = None
    if kind_raw:
        if kind_raw not in {"agent", "skill", "workflow", "prompt", "issue", "generic"}:
            raise ValueError(f"unknown template kind: {kind_raw}")
        kind = kind_raw  # type: ignore[assignment]
    catalogue = TemplateCatalogue(_registry_for(context))
    templates = catalogue.list(kind=kind, source=source)
    if not templates:
        return "No templates found."
    lines = [
        "Templates:",
        "",
        "id                           kind     source    tags                     title",
    ]
    lines.extend(_format_template(t) for t in templates)
    return "\n".join(lines)


def _show(parts: list[str], context: CommandContext) -> str:
    if len(parts) != 1:
        raise ValueError("show requires exactly one template id")
    catalogue = TemplateCatalogue(_registry_for(context))
    manifest = catalogue.describe(parts[0])
    data = manifest.to_dict()
    lines = [f"{data['id']} - {data['title']}"]
    for key in (
        "kind",
        "source",
        "category",
        "tags",
        "schema_version",
        "min_clawcodex_version",
        "output_path_template",
        "description",
    ):
        if key in data:
            lines.append(f"{key}: {data[key]}")
    if manifest.variables:
        lines.append("variables:")
        for var in manifest.variables:
            req = "required" if var.required else "optional"
            secret = ", secret" if var.secret else ""
            default = f", default={var.default!r}" if var.default is not None else ""
            lines.append(f"  - {var.name} ({req}{secret}{default}) {var.description}")
    return "\n".join(lines)


def _search(parts: list[str], context: CommandContext) -> str:
    query = " ".join(parts).strip()
    if not query:
        raise ValueError("search requires a query")
    catalogue = TemplateCatalogue(_registry_for(context))
    results = catalogue.search(query)
    if not results:
        return "No matching templates."
    lines = [f"Templates matching {query!r}:", ""]
    lines.extend(_format_template(t) for t in results)
    return "\n".join(lines)


def _preview(parts: list[str], context: CommandContext) -> str:
    values = _vars(parts)
    if len(parts) != 1:
        raise ValueError("preview requires exactly one template id")
    catalogue = TemplateCatalogue(_registry_for(context))
    rendered = catalogue.preview(
        parts[0],
        values,
        workspace_root=context.workspace_root,
    )
    lines = [f"Template: {rendered.template_id}", f"Kind: {rendered.kind}"]
    if rendered.output_path is not None:
        lines.append(f"Output: {rendered.output_path}")
    if rendered.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in rendered.warnings)
    lines.append("")
    lines.append(rendered.content)
    return "\n".join(lines)


def _render(parts: list[str], context: CommandContext) -> str:
    values = _vars(parts)
    output = _take_option(parts, "--output")
    overwrite = _flag(parts, "--overwrite")
    if len(parts) != 1:
        raise ValueError("render requires exactly one template id")
    registry = _registry_for(context)
    template = registry.get(parts[0])
    if output:
        metadata = dict(template.metadata)
        metadata["output_path_template"] = output
        from clawcodex_ext.services.templates.models import Template

        template = Template(
            id=template.id,
            title=template.title,
            description=template.description,
            fields=dict(template.fields),
            metadata=metadata,
            source=template.source,
        )
    rendered = TemplateRenderer().render(template, values, workspace_root=context.workspace_root)
    path = TemplateGenerator(workspace_root=context.workspace_root).generate(
        rendered,
        overwrite=overwrite,
    )
    return f"Rendered template {template.id!r} to {path}"


def _create(parts: list[str], context: CommandContext) -> str:
    if not parts:
        raise ValueError("create requires a kind")
    kind = parts.pop(0)
    if kind != "skill":
        raise ValueError("create currently supports: skill")
    name = _take_option(parts, "--name")
    description = _take_option(parts, "--description") or "Generated ClawCodex skill."
    overwrite = _flag(parts, "--overwrite")
    if parts:
        raise ValueError(f"unexpected arguments: {' '.join(parts)}")
    if not name:
        raise ValueError("create skill requires --name")
    from clawcodex_ext.services.templates.models import Template

    template = Template(
        id="skill-inline",
        title="Inline Skill",
        fields={
            "content_template": "# {{ skill_name }}\n\n{{ description }}\n",
        },
        metadata={
            "kind": "skill",
            "schema_version": "1",
            "output_path_template": ".claude/skills/{{ skill_name }}/SKILL.md",
            "variables": [
                {
                    "name": "skill_name",
                    "description": "Skill directory name.",
                    "pattern": r"^[A-Za-z0-9_-]+$",
                },
                {"name": "description", "description": "Short skill description."},
            ],
        },
    )
    rendered = TemplateRenderer().render(
        template,
        {"skill_name": name, "description": description},
        workspace_root=context.workspace_root,
    )
    path = TemplateGenerator(workspace_root=context.workspace_root).generate(
        rendered,
        overwrite=overwrite,
    )
    return f"Created skill template at {path}"


def _install(parts: list[str], context: CommandContext) -> str:
    if len(parts) != 1:
        raise ValueError("install requires exactly one local file path")
    path = Path(parts[0]).expanduser()
    if not path.is_absolute():
        path = (context.cwd or context.workspace_root) / path
    templates = parse_template_file(path)
    registry = _registry_for(context)
    count = registry.register_many(templates, overwrite=True)
    return f"Installed {count} template(s) from {path}"


def _validate(parts: list[str], context: CommandContext) -> str:
    if len(parts) != 1:
        raise ValueError("validate requires exactly one file path")
    path = Path(parts[0]).expanduser()
    if not path.is_absolute():
        path = (context.cwd or context.workspace_root) / path
    templates = parse_template_file(path)
    from clawcodex_ext.services.templates import check_compatibility

    for template in templates:
        check_compatibility(template)
        get_manifest(template)
    return f"Valid template file: {path} ({len(templates)} template(s))"


def template_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    try:
        parts = _parts(args)
        if not parts:
            return LocalCommandResult(type="text", value=_usage())
        subcommand = parts.pop(0)
        handlers: dict[str, Any] = {
            "list": _list,
            "show": _show,
            "search": _search,
            "preview": _preview,
            "render": _render,
            "create": _create,
            "install": _install,
            "validate": _validate,
        }
        handler = handlers.get(subcommand)
        if handler is None:
            raise ValueError(f"unknown template subcommand: {subcommand}")
        return LocalCommandResult(type="text", value=handler(parts, context))
    except (TemplatesError, ValueError) as exc:
        return LocalCommandResult(type="text", value=f"{exc}\n\n{_usage()}")


TEMPLATE_COMMAND = LocalCommand(
    name="template",
    description="Browse, preview, render, create, install, and validate templates",
    argument_hint="list|show|search|preview|render|create|install|validate",
    supports_non_interactive=True,
)
TEMPLATE_COMMAND.set_call(template_command_call)


def register_template_commands(registry: CommandRegistry) -> None:
    registry.register(TEMPLATE_COMMAND)


__all__ = ["TEMPLATE_COMMAND", "register_template_commands", "template_command_call"]
