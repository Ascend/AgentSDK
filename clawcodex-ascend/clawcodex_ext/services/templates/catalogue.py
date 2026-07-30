"""Catalogue browsing, search, describe, and preview for F-95 templates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from difflib import get_close_matches

from .exceptions import TemplateNotFoundError
from .models import Template, TemplateKind, TemplateManifest, get_manifest, tags_of
from .registry import TemplateRegistry
from .renderer import TemplateRenderer


def _haystack(template: Template) -> str:
    manifest = get_manifest(template)
    return " ".join(
        part
        for part in (
            template.id,
            template.title,
            template.description or "",
            manifest.kind,
            manifest.category or "",
            " ".join(manifest.tags),
            template.source,
        )
        if part
    ).lower()


class TemplateCatalogue:
    """Filter and search a :class:`TemplateRegistry`."""

    def __init__(
        self,
        registry: TemplateRegistry,
        *,
        renderer: TemplateRenderer | None = None,
    ) -> None:
        self.registry = registry
        self.renderer = renderer or TemplateRenderer()

    def list(
        self,
        *,
        kind: TemplateKind | None = None,
        source: str | None = None,
        tags: Iterable[str] = (),
    ) -> list[Template]:
        required_tags = set(tags)
        out: list[Template] = []
        for template in self.registry.list_templates(source=source):
            manifest = get_manifest(template)
            if kind is not None and manifest.kind != kind:
                continue
            if required_tags and not required_tags.issubset(set(tags_of(template))):
                continue
            out.append(template)
        return out

    def search(self, query: str, *, top_k: int = 20) -> list[Template]:
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return self.list()[:top_k]
        scored: list[tuple[int, str, Template]] = []
        for template in self.registry.list_templates():
            haystack = _haystack(template)
            score = sum(3 if term in template.id.lower() else 1 for term in terms if term in haystack)
            if score:
                scored.append((score, template.id, template))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[:top_k]]

    def describe(self, template_id: str) -> TemplateManifest:
        try:
            return get_manifest(self.registry.get(template_id))
        except TemplateNotFoundError as exc:
            suggestions = self.suggest(template_id)
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise TemplateNotFoundError(f"template not registered: {template_id}.{hint}") from exc

    def preview(
        self,
        template_id: str,
        variables: Mapping[str, object],
        *,
        workspace_root=None,
    ):
        try:
            template = self.registry.get(template_id)
        except TemplateNotFoundError as exc:
            suggestions = self.suggest(template_id)
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise TemplateNotFoundError(f"template not registered: {template_id}.{hint}") from exc
        return self.renderer.preview(template, variables, workspace_root=workspace_root)

    def suggest(self, template_id: str, *, limit: int = 5) -> list[str]:
        ids = self.registry.list_ids()
        return get_close_matches(template_id, ids, n=limit, cutoff=0.35)


__all__ = ["TemplateCatalogue"]
