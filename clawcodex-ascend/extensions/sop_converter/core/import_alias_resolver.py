"""Resolve type hints to module-qualified identity keys via import aliases.

Disambiguates same-named types (e.g. ``ReActAgentConfig`` in two modules) without
SDK-specific hardcoding.
"""

from __future__ import annotations

import ast
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from extensions.sop_converter.core.source_parser import SourceComponent

logger = logging.getLogger(__name__)


def _has_top_level_union(hint: str) -> bool:
    """True when ``|`` separates union members outside of brackets."""
    depth = 0
    for ch in hint:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(depth - 1, 0)
        elif ch == "|" and depth == 0:
            return True
    return False


def _split_union(type_hint: str) -> list[str]:
    cleaned = type_hint.strip()
    if not cleaned:
        return []

    if cleaned.startswith("Union[") and cleaned.endswith("]"):
        inner = cleaned[len("Union[") : -1]
        # depth-tracked split so commas inside nested generics
        # (e.g. Union[Dict[str, int], List[str]]) are not treated as separators.
        parts = []
        current: list[str] = []
        depth = 0
        for ch in inner:
            if ch in "[({":
                depth += 1
            elif ch in "])}":
                depth = max(depth - 1, 0)
            if ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(ch)
        parts.append("".join(current).strip())
    elif _has_top_level_union(cleaned):
        parts = []
        current: list[str] = []
        depth = 0
        for ch in cleaned:
            if ch in "[({":
                depth += 1
            elif ch in "])}":
                depth = max(depth - 1, 0)
            if ch == "|" and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(ch)
        parts.append("".join(current).strip())
    else:
        parts = [cleaned]

    out: list[str] = []
    for part in parts:
        if part in ("None", "NoneType"):
            continue
        if part.startswith("Optional[") and part.endswith("]"):
            out.extend(_split_union(part[len("Optional[") : -1]))
        else:
            out.append(part)
    return out


def resolve_module_path(
    component: SourceComponent,
    source_dir: str,
    file_stem: str,
) -> str:
    """Infer dotted import path for a source file within a component."""
    source_dir_path = Path(source_dir).resolve()
    comp_rel = Path(component.file_path)

    try:
        # source_parser sets file_path relative to source_dir.parent, so anchor
        # there rather than a bare dir name or the process cwd.
        module_dir = (source_dir_path.parent / comp_rel).resolve().relative_to(source_dir_path)
    except ValueError:
        module_dir = comp_rel

    parts = list(module_dir.parts) if module_dir.parts != (".",) else []
    parts.append(file_stem)
    return ".".join(parts)


def type_identity_key(qualified: str) -> str:
    """Normalize ``module.Class`` to a comparable dependency token."""
    cleaned = qualified.strip().strip("'\"")
    cleaned = re.sub(r"[|\\/:*?\"<>']", "", cleaned.replace(".", "_"))
    return cleaned.lower()


@lru_cache(maxsize=64)
def _module_file_index(source_dir: str) -> dict[str, Path]:
    root = Path(source_dir).resolve()
    index: dict[str, Path] = {}
    if not root.is_dir():
        return index

    skip_dirs = frozenset({"__pycache__", ".git", "tests", "test", "examples", "example", ".clawcodex"})
    for path in root.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        rel = path.relative_to(root)
        if rel.name == "__init__.py":
            module = ".".join(rel.parent.parts)
        else:
            module = ".".join(rel.with_suffix("").parts)
        index[module] = path
    return index


_MODULE_IMPORT_INDEX_CACHE: dict[str, "ModuleImportIndex"] = {}


class ModuleImportIndex:
    """Per-module local classes, import aliases, and simple re-export assignments."""

    def __init__(self, source_dir: str) -> None:
        self._source_dir = str(Path(source_dir).resolve())

        if self._source_dir in _MODULE_IMPORT_INDEX_CACHE:
            cached = _MODULE_IMPORT_INDEX_CACHE[self._source_dir]
            self._file_index = cached._file_index
            self._local_classes = cached._local_classes
            self._import_aliases = cached._import_aliases
            self._assign_aliases = cached._assign_aliases
            return

        self._file_index = _module_file_index(self._source_dir)
        self._local_classes: dict[str, set[str]] = {}
        self._import_aliases: dict[str, dict[str, tuple[str, str]]] = {}
        self._assign_aliases: dict[str, dict[str, str]] = {}
        for module_path, path in self._file_index.items():
            self._parse_module(module_path, path)

        _MODULE_IMPORT_INDEX_CACHE[self._source_dir] = self

    def resolve_type_identity(self, module_path: str, type_hint: str | None) -> str | None:
        if not type_hint:
            return None
        if type_hint.startswith(("Optional[", "Union[")) or _has_top_level_union(type_hint):
            for part in _split_union(type_hint):
                identity = self._resolve_one(module_path, part.strip())
                if identity:
                    return identity
            return None
        return self._resolve_one(module_path, type_hint)

    def resolve_import_path(self, module_path: str, type_hint: str | None) -> tuple[str, str] | None:
        """Resolve a type hint to the actual ``(module_path, class_name)`` in *module_path*.

        Unlike :meth:`resolve_type_identity`, which returns a normalised identity
        key, this method returns the real dotted module path and class name so
        callers can generate ``from module import Class`` statements.  It follows
        local class definitions, import aliases and simple re-export assignments.
        """
        if not type_hint:
            return None
        if type_hint.startswith(("Optional[", "Union[")) or _has_top_level_union(type_hint):
            for part in _split_union(type_hint):
                resolved = self._resolve_import_one(module_path, part.strip())
                if resolved:
                    return resolved
            return None
        return self._resolve_import_one(module_path, type_hint)

    def _resolve_relative(
        self,
        current_module: str,
        module: str | None,
        level: int,
    ) -> str:
        if level == 0:
            return module or ""
        parts = current_module.split(".") if current_module else []
        current_path = self._file_index.get(current_module)
        package_parts = parts if current_path is not None and current_path.name == "__init__.py" else parts[:-1]
        ascend = max(level - 1, 0)
        if ascend > len(package_parts):
            return module or ""
        base = package_parts[: len(package_parts) - ascend] if ascend else list(package_parts)
        if module:
            base.extend(module.split("."))
        return ".".join(base)

    def _parse_module(self, module_path: str, path: Path) -> None:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            logger.debug("Failed to parse module %s: %s", module_path, exc)
            return

        local: set[str] = set()
        aliases: dict[str, tuple[str, str]] = {}
        assigns: dict[str, str] = {}

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                local.add(node.name)
            elif isinstance(node, ast.ImportFrom):
                target_mod = self._resolve_relative(module_path, node.module, node.level)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    aliases[bound] = (target_mod, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name
                    aliases[bound] = (alias.name, alias.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    src = self._assignment_source_name(node.value)
                    if src:
                        assigns[target.id] = src

        self._local_classes[module_path] = local
        self._import_aliases[module_path] = aliases
        self._assign_aliases[module_path] = assigns

    @staticmethod
    def _assignment_source_name(value: ast.AST) -> str | None:
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Call) and value.args:
            first = value.args[0]
            if isinstance(first, ast.Name):
                return first.id
        return None

    def _resolve_one(self, module_path: str, hint: str) -> str | None:
        hint = hint.strip().strip("'\"")
        if not hint:
            return None

        if hint.startswith(("Optional[", "Union[")) or _has_top_level_union(hint):
            parts = _split_union(hint)
            for part in parts:
                resolved = self._resolve_one(module_path, part.strip())
                if resolved:
                    return resolved
            return None

        if "." in hint and not hint.endswith("]"):
            return type_identity_key(hint)

        root = re.sub(r"\[.*", "", hint).strip()
        if not root:
            return None

        resolved = self._follow_name(module_path, root)
        if resolved:
            return resolved

        from extensions.sop_converter.core.tool_dependencies import sanitize_type_name

        return sanitize_type_name(hint)

    def _follow_name(
        self,
        module_path: str,
        name: str,
        *,
        depth: int = 0,
        formatter: Callable[[str, str], str | tuple[str, str]] | None = None,
    ) -> str | tuple[str, str] | None:
        """Follow ``name`` through local / assignment / import aliases to its origin.

        ``formatter(module_path, name)`` builds the return value; the default
        produces a string identity key, and callers needing the raw
        ``(module_path, name)`` pair pass a tuple-building formatter.
        """
        fmt = formatter or (lambda mod, nm: type_identity_key(f"{mod}.{nm}"))
        if depth > 6:
            return None

        if name in self._local_classes.get(module_path, ()):
            return fmt(module_path, name)

        assign_src = self._assign_aliases.get(module_path, {}).get(name)
        if assign_src:
            followed = self._follow_name(module_path, assign_src, depth=depth + 1, formatter=formatter)
            if followed:
                return followed

        imported = self._import_aliases.get(module_path, {}).get(name)
        if imported:
            target_mod, target_name = imported
            if not target_mod:
                return fmt(module_path, target_name)
            followed = self._follow_name(target_mod, target_name, depth=depth + 1, formatter=formatter)
            if followed:
                return followed
            return fmt(target_mod, target_name)

        return None

    def _resolve_import_one(self, module_path: str, hint: str) -> tuple[str, str] | None:
        hint = hint.strip().strip("'\"")
        if not hint:
            return None

        if hint.startswith(("Optional[", "Union[")) or _has_top_level_union(hint):
            parts = _split_union(hint)
            for part in parts:
                resolved = self._resolve_import_one(module_path, part.strip())
                if resolved:
                    return resolved
            return None

        if "." in hint and not hint.endswith("]"):
            # Already module-qualified, e.g. ``module.Class``.
            mod, _, cls = hint.rpartition(".")
            return mod, cls

        root = re.sub(r"\[.*", "", hint).strip()
        if not root:
            return None

        return self._follow_name(
            module_path,
            root,
            formatter=lambda mod, nm: (mod, nm),
        )
