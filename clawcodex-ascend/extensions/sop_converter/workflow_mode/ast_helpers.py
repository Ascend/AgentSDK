#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""AST helpers shared by the discriminator and extractors."""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "tests",
    "test",
    # clawcodex's own output/config dir — generated bundle artifacts
    # (agent-tools/scripts wrappers) are not source and must not be
    # walked by the workflow discriminator / extractors.  A stale
    # generated wrapper with an invalid signature would otherwise emit
    # "Failed to parse ..." noise (and valid ones could yield false
    # enum/gate detections).
    ".clawcodex",
}
_STAGE_NAME_RE = re.compile(r"STAGE|PHASE|STEP|PIPELINE", re.IGNORECASE)
_HTTP_STATUS_RE = re.compile(r"\b(?:OK|ERROR|HTTP|STATUS)\b", re.IGNORECASE)


def walk_py_files(source_dir: Path, *, exclude: set[str] | None = None) -> Iterator[Path]:
    """Recursively yield ``.py`` files, skipping excluded directory names."""
    skip = exclude if exclude is not None else _EXCLUDE_DIRS
    root = source_dir.resolve()
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.py")):
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in skip for part in rel_parts[:-1]):
            continue
        yield path


def parse_ast(py_file: Path) -> ast.Module | None:
    try:
        source = py_file.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(py_file))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        logger.warning("Failed to parse %s: %s", py_file, exc)
        return None


def get_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def find_enum_classes(tree: ast.Module) -> list[ast.ClassDef]:
    result: list[ast.ClassDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = get_name(base)
            if base_name in ("IntEnum", "Enum"):
                result.append(node)
                break
    return result


def get_enum_members(cls: ast.ClassDef) -> dict[str, int]:
    return dict(get_enum_members_ordered(cls))


def get_enum_members_ordered(cls: ast.ClassDef) -> list[tuple[str, int]]:
    """Enum members in source definition order."""
    members: list[tuple[str, int]] = []
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    value = _const_int(node.value)
                    if value is not None:
                        members.append((target.id, value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _const_int(node.value) if node.value is not None else None
            if value is not None:
                members.append((node.target.id, value))
    return members


def _const_int(node: ast.expr | None) -> int | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def is_stage_like_enum(cls: ast.ClassDef, members: dict[str, int]) -> tuple[bool, float]:
    """Return (is_stage_enum, weight_factor). weight_factor 1.0 = full, 0.6 = partial."""
    if _STAGE_NAME_RE.search(cls.name):
        return True, 1.0
    if any(_STAGE_NAME_RE.search(name) for name in members):
        return True, 1.0
    if _looks_like_http_status_enum(members):
        return False, 0.0
    if len(members) >= 2:
        return True, 0.6
    return False, 0.0


def _looks_like_http_status_enum(members: dict[str, int]) -> bool:
    if len(members) < 2:
        return False
    has_status_name = any(_HTTP_STATUS_RE.search(name) for name in members)
    has_high_value = any(value >= 100 for value in members.values())
    return has_status_name and has_high_value


def pick_primary_stage_enum(trees: dict[Path, ast.Module]) -> str | None:
    """Pick the most likely stage enum class name across all trees."""
    best_name: str | None = None
    best_score = -1.0
    for tree in trees.values():
        for cls in find_enum_classes(tree):
            members = get_enum_members(cls)
            if not members:
                continue
            is_stage, factor = is_stage_like_enum(cls, members)
            if not is_stage:
                continue
            score = len(members) * factor
            if _STAGE_NAME_RE.search(cls.name):
                score += 10
            if score > best_score:
                best_score = score
                best_name = cls.name
    return best_name


def collect_enum_member_names(
    trees: dict[Path, ast.Module],
    *,
    primary_class: str | None = None,
) -> set[str]:
    names: set[str] = set()
    for tree in trees.values():
        for cls in find_enum_classes(tree):
            if primary_class and cls.name != primary_class:
                continue
            names.update(get_enum_members(cls).keys())
    return names


def _iter_module_named_assignments(tree: ast.Module):
    """Yield ``(name, value)`` for module-level ``Assign`` and ``AnnAssign``."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    yield target.id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                yield node.target.id, node.value


def find_dict_assignments(tree: ast.Module) -> list[tuple[str, ast.Dict]]:
    """Literal ``{...}`` dict assignments only (backward compatible)."""
    return [(name, expr) for name, expr in find_dict_mapping_assignments(tree) if isinstance(expr, ast.Dict)]


def find_dict_mapping_assignments(tree: ast.Module) -> list[tuple[str, ast.expr]]:
    """Module-level dict or dict-comprehension assignments."""
    return [
        (name, expr)
        for name, expr in _iter_module_named_assignments(tree)
        if isinstance(expr, (ast.Dict, ast.DictComp))
    ]


def find_named_assign_expr(tree: ast.Module, *names: str) -> dict[str, ast.expr]:
    wanted = set(names)
    found: dict[str, ast.expr] = {}
    for name, expr in _iter_module_named_assignments(tree):
        if name in wanted:
            found[name] = expr
    return found


def parse_stage_sequence_from_expr(
    expr: ast.expr,
    *,
    enum_class_names: set[str],
    enum_members_ordered: list[tuple[str, int]],
    module_assigns: dict[str, ast.expr] | None = None,
) -> list[int]:
    """Resolve ``tuple(Stage)``, ``STAGE_SEQUENCE``, or ordered enum members."""
    module_assigns = module_assigns or {}
    if isinstance(expr, ast.Name):
        if expr.id in module_assigns:
            return parse_stage_sequence_from_expr(
                module_assigns[expr.id],
                enum_class_names=enum_class_names,
                enum_members_ordered=enum_members_ordered,
                module_assigns=module_assigns,
            )
        if expr.id in enum_class_names:
            return [value for _, value in enum_members_ordered]
    if isinstance(expr, ast.Call) and get_name(expr.func) == "tuple":
        if expr.args:
            return parse_stage_sequence_from_expr(
                expr.args[0],
                enum_class_names=enum_class_names,
                enum_members_ordered=enum_members_ordered,
                module_assigns=module_assigns,
            )
    if isinstance(expr, (ast.Tuple, ast.List)):
        ids: list[int] = []
        for el in expr.elts:
            ref = resolve_enum_member(el, enum_class_names, dict(enum_members_ordered))
            if ref:
                ids.append(ref[1])
        if ids:
            return ids
    return [value for _, value in enum_members_ordered]


def _dictcomp_has_linear_stage_loop(gen: ast.comprehension) -> bool:
    """True when the dictcomp loop looks like a stage-sequence walk."""
    target = gen.target
    if isinstance(target, ast.Tuple):
        return len(target.elts) == 2
    if isinstance(target, ast.Name):
        return True
    return False


def parse_linear_next_stage_dictcomp(
    dictcomp: ast.DictComp,
    *,
    stage_sequence: list[int],
) -> list[tuple[int, int]]:
    """``{s: seq[i+1] for i, s in enumerate(seq)}`` → consecutive transitions."""
    if not stage_sequence:
        return []
    generators = dictcomp.generators
    if len(generators) != 1:
        return []
    gen = generators[0]
    if not _dictcomp_has_linear_stage_loop(gen):
        return []
    pairs: list[tuple[int, int]] = []
    for idx, sid in enumerate(stage_sequence):
        if idx + 1 < len(stage_sequence):
            pairs.append((sid, stage_sequence[idx + 1]))
    return pairs


def parse_enum_dict_mapping_from_expr(
    expr: ast.expr,
    enum_class_names: set[str],
    member_to_value: dict[str, int],
    *,
    stage_sequence: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Parse enum→enum mappings from literal dict or NEXT_STAGE-style dictcomp."""
    if isinstance(expr, ast.Dict):
        return parse_enum_dict_mapping(expr, enum_class_names, member_to_value)
    if isinstance(expr, ast.DictComp):
        seq = stage_sequence or sorted(member_to_value.values())
        return parse_linear_next_stage_dictcomp(expr, stage_sequence=seq)
    return []


def _const_str_tuple(node: ast.expr | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, (ast.Tuple, ast.List)):
        result: list[str] = []
        for el in node.elts:
            if isinstance(el, ast.Constant) and isinstance(el.value, str):
                result.append(el.value)
        return result
    return []


def parse_contracts_dict(
    dict_node: ast.Dict,
    enum_class_names: set[str],
    member_to_value: dict[str, int],
    *,
    contract_call_names: tuple[str, ...] = ("StageContract",),
) -> dict[int, tuple[list[str], list[str], str | None, str]]:
    """Parse ``CONTRACTS = {Stage.X: StageContract(...), ...}``."""
    contracts: dict[int, tuple[list[str], list[str], str | None, str]] = {}
    for key, value in zip(dict_node.keys, dict_node.values):
        if key is None or value is None:
            continue
        key_ref = resolve_enum_member(key, enum_class_names, member_to_value)
        if not key_ref:
            continue
        stage_id = key_ref[1]
        if not isinstance(value, ast.Call):
            continue
        call_name = get_name(value.func)
        if call_name not in contract_call_names:
            continue
        input_files: list[str] = []
        output_files: list[str] = []
        dod = ""
        for kw in value.keywords:
            if kw.arg == "input_files":
                input_files = _const_str_tuple(kw.value)
            elif kw.arg == "output_files":
                output_files = _const_str_tuple(kw.value)
            elif kw.arg == "dod":
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    dod = kw.value.value
        contracts[stage_id] = (input_files, output_files, call_name, dod)
    return contracts


def parse_enum_to_name_dict(
    dict_node: ast.Dict,
    enum_class_names: set[str],
    member_to_value: dict[str, int],
) -> dict[int, str]:
    """Parse ``{Stage.X: handler_name, ...}`` where values are function Name refs."""
    mapping: dict[int, str] = {}
    for key, value in zip(dict_node.keys, dict_node.values):
        if key is None or value is None:
            continue
        key_ref = resolve_enum_member(key, enum_class_names, member_to_value)
        if not key_ref:
            continue
        func_name = get_name(value)
        if func_name:
            mapping[key_ref[1]] = func_name
    return mapping


def parse_string_to_stage_dict(
    dict_node: ast.Dict,
    enum_class_names: set[str],
    member_to_value: dict[str, int],
) -> dict[str, int]:
    """Parse ``{"pivot": Stage.X, ...}`` → outcome → stage id."""
    mapping: dict[str, int] = {}
    for key, value in zip(dict_node.keys, dict_node.values):
        if key is None or value is None:
            continue
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            outcome = key.value
        else:
            continue
        ref = resolve_enum_member(value, enum_class_names, member_to_value)
        if ref:
            mapping[outcome] = ref[1]
    return mapping


def find_dataclass_defs(tree: ast.Module) -> list[ast.ClassDef]:
    result: list[ast.ClassDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if _has_decorator(node, "dataclass"):
            result.append(node)
    return result


def _has_decorator(cls: ast.ClassDef, name: str) -> bool:
    for dec in cls.decorator_list:
        dec_name = get_name(dec)
        if dec_name == name:
            return True
    return False


def class_annotation_names(cls: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def find_gate_assigns(tree: ast.Module) -> list[tuple[str, ast.expr]]:
    """Find module-level assignments whose variable name contains ``GATE``.

    The name-filtered lookup is used by gate-detection heuristics; the
    assigned value's type is validated by the callers (e.g. checking for a
    ``frozenset``/``set`` literal), so this helper only performs the
    variable-name match.
    """
    results: list[tuple[str, ast.expr]] = []
    for var_name, value in _iter_module_named_assignments(tree):
        if "GATE" in var_name.upper():
            results.append((var_name, value))
    return results


def find_func_by_prefix(
    tree: ast.Module,
    prefixes: tuple[str, ...] = ("decide_", "should_", "check_gate", "resolve_stage", "resolve_"),
    keywords: tuple[str, ...] = ("pivot", "refine", "proceed", "gate"),
) -> list[ast.FunctionDef]:
    result: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if any(name.startswith(p) or name == p.rstrip("_") for p in prefixes):
                result.append(node)  # type: ignore[arg-type]
            elif keywords and any(k in name for k in keywords):
                result.append(node)  # type: ignore[arg-type]
    return result


def extract_docstring_first_para(node: ast.AST) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    return doc.strip().split("\n\n")[0].strip().split("\n")[0].strip()


def resolve_enum_member(
    node: ast.expr,
    enum_class_names: set[str],
    member_to_value: dict[str, int] | None = None,
) -> tuple[str, int] | None:
    """Resolve AST node to (member_name, int_value)."""
    member_to_value = member_to_value or {}
    if isinstance(node, ast.Name):
        name = node.id
        if name in member_to_value:
            return name, member_to_value[name]
        return None
    if isinstance(node, ast.Attribute):
        base = node.value.id if isinstance(node.value, ast.Name) else None
        if base in enum_class_names:
            attr = node.attr
            if attr in member_to_value:
                return attr, member_to_value[attr]
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return str(node.value), node.value
    return None


def parse_enum_dict_mapping(
    dict_node: ast.Dict,
    enum_class_names: set[str],
    member_to_value: dict[str, int],
) -> list[tuple[int, int]]:
    """Parse ``{Stage.A: Stage.B, ...}`` → [(from_id, to_id), ...]."""
    pairs: list[tuple[int, int]] = []
    for key, value in zip(dict_node.keys, dict_node.values):
        if key is None or value is None:
            continue
        from_ref = resolve_enum_member(key, enum_class_names, member_to_value)
        to_ref = resolve_enum_member(value, enum_class_names, member_to_value)
        if from_ref and to_ref:
            pairs.append((from_ref[1], to_ref[1]))
    return pairs


def parse_frozenset_members(
    expr: ast.expr,
    enum_class_names: set[str],
    member_to_value: dict[str, int],
) -> list[int]:
    ids: list[int] = []
    elements: list[ast.expr] = []
    if isinstance(expr, ast.Set):
        elements = list(expr.elts)
    elif isinstance(expr, ast.Call):
        call_name = get_name(expr.func)
        if call_name in ("frozenset", "set"):
            if expr.args and isinstance(expr.args[0], (ast.Set, ast.List, ast.Tuple)):
                container = expr.args[0]
                if isinstance(container, ast.Set):
                    elements = list(container.elts)
                elif isinstance(container, (ast.List, ast.Tuple)):
                    elements = list(container.elts)
    for el in elements:
        ref = resolve_enum_member(el, enum_class_names, member_to_value)
        if ref:
            ids.append(ref[1])
    return ids


def to_kebab(name: str) -> str:
    """Convert ``camelCase``/``snake_case`` identifiers to ``kebab-case``.

    Shared by the workflow extractors and generator modules for
    cross-referencing stage/skill names with registered tool names.
    """
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name)
    return s.replace("_", "-").lower()


def extract_callee_names(file_path: Path) -> set[str]:
    """Extract function call names from a Python source file.

    Walks the AST collecting names from ``func()`` and ``obj.method()`` call
    sites.  Returns snake_case callee names only — the caller converts to
    kebab-case for cross-referencing with registered tool names.
    """
    tree = parse_ast(file_path)
    if tree is None:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = get_name(node.func)
        if fn and not fn.startswith("_"):
            # Collect both the simple name and any attribute chain leaf
            names.add(fn)
            if isinstance(node.func, ast.Attribute):
                # Also collect full dotted path: obj.method → "obj_method"
                parts: list[str] = []
                current: ast.expr = node.func
                while isinstance(current, ast.Attribute):
                    parts.insert(0, current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.insert(0, current.id)
                dotted = "_".join(parts)
                if not dotted.startswith("_"):
                    names.add(dotted)
    return names
