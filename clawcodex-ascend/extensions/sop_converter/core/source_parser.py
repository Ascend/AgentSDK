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

"""SourceCodeParser — Python source AST parser that extracts ``SourceComponent[]``.

Recursively scans ``.py`` files under a Python source directory to extract
class definitions, method signatures, docstrings, parameter type annotations,
and import dependencies, producing a structured ``list[SourceComponent]``.

Design decisions:
- No semantic analysis of the source — structural extraction only
  (classes / methods / parameters / docstrings).
- ``SourceComponent`` / ``SourceOperation`` / ``ParamSpec`` are pure data containers.
- Docstrings in Google / NumPy / reST / Chinese ``参数:`` styles are supported
  (parameter names, descriptions, and explicit type fields where present),
  with a unified fallback to the first paragraph.
"""

# pylint: disable=too-many-nested-blocks
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ASYNC_ITER_RETURN_RE = re.compile(r"\bAsync(?:Iterator|Generator)\b")


class _InteractiveInputDetector(ast.NodeVisitor):
    """AST visitor that detects interactive input calls.

    Detects:
    - input() calls
    - getpass.getpass() calls
    - sys.stdin.readline() calls
    - sys.stdin.read() calls
    """

    def __init__(self) -> None:
        self.has_interactive_input = False
        self.prompts: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)

        if isinstance(node.func, ast.Name) and node.func.id == "input":
            self.has_interactive_input = True
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                self.prompts.append(node.args[0].value)
            else:
                self.prompts.append("")
            return

        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            value = node.func.value

            if attr == "getpass" and isinstance(value, ast.Name) and value.id == "getpass":
                self.has_interactive_input = True
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    self.prompts.append(node.args[0].value)
                else:
                    self.prompts.append("")
                return

            if attr == "readline" and isinstance(value, ast.Attribute):
                if isinstance(value.value, ast.Name) and value.value.id == "sys" and value.attr == "stdin":
                    self.has_interactive_input = True
                    self.prompts.append("")
                    return

            if attr == "read" and isinstance(value, ast.Attribute):
                if isinstance(value.value, ast.Name) and value.value.id == "sys" and value.attr == "stdin":
                    self.has_interactive_input = True
                    self.prompts.append("")
                    return


def detect_interactive_input(node: ast.AST) -> tuple[bool, list[str]]:
    """Detect if an AST node contains interactive input calls.

    Returns:
        (has_interactive_input, prompts)
    """
    detector = _InteractiveInputDetector()
    detector.visit(node)
    return detector.has_interactive_input, detector.prompts


def is_async_generator_operation(
    node: ast.AST,
    return_type: str | None,
) -> bool:
    """True when *node* is an async function that yields or annotates AsyncIterator."""
    if not isinstance(node, ast.AsyncFunctionDef):
        return False
    if return_type and _ASYNC_ITER_RETURN_RE.search(return_type):
        return True
    for child in ast.walk(node):
        if isinstance(child, (ast.Yield, ast.YieldFrom)):
            return True
    return False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ParamSpec:
    """Specification of a single operation parameter."""

    name: str
    type_hint: str | None = None
    default: Any | None = None
    required: bool = True
    description: str = ""


_FACTORY_PREFIXES = frozenset({"create_", "build_", "make_", "new_"})


_SIMPLE_RETURN_TYPES = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "None",
        "NoneType",
        "Path",
        "str | None",
        "int | None",
    }
)

# Docstring descriptions that imply a mapping parameter (no explicit type annotation).
_DICT_DESC_RE = re.compile(
    r"(?:字典|映射|键值|键-值|\bdict\b|\bmapping\b|\bjson\s+object\b)",
    re.IGNORECASE,
)

_DOC_PARAM_LINE_RE = re.compile(r"^(\w+)\s*(?:\(([^)]*)\))?\s*[：:]\s*(.*)")

_DOC_SECTION_STOP_MARKERS = (
    "returns:",
    "raises:",
    "yields:",
    "返回:",
    "返回：",
    "抛出:",
    "抛出：",
    "异常:",
    "异常：",
    "说明:",
    "说明：",
)


def infer_type_hint_from_description(description: str) -> str | None:
    """Infer a Python type hint from a parameter description line.

    Used when docstrings omit explicit types (common in Chinese ``参数:`` blocks).
    Currently recognises mapping/dict semantics only — keeps false positives low.
    """
    if not description:
        return None
    if _DICT_DESC_RE.search(description.strip()):
        return "dict"
    return None


def _resolve_doc_param_type_hint(type_str: str | None, description: str) -> str | None:
    """Prefer explicit doc type; otherwise infer from description keywords."""
    if type_str and type_str.strip():
        return type_str.strip()
    return infer_type_hint_from_description(description)


@dataclass
class SourceOperation:
    """An operation inside a component that an agent can invoke."""

    name: str
    description: str  # first paragraph of the method docstring
    parameters: list[ParamSpec] = field(default_factory=list)
    return_type: str | None = None
    source_code: str = ""  # full source snippet, embedded in the skill reference
    class_name: str | None = None  # owning class name (used for ClassName.methodName naming in IO_RELATION)
    file_stem: str = ""  # source file name without .py (used to disambiguate top-level functions)
    has_docstring: bool = False  # whether the original docstring is non-empty
    is_async: bool = False  # whether defined with ``async def``
    is_async_generator: bool = False  # async def that returns/yields an async iterator
    is_property: bool = False  # read-only attribute decorated with @property (no params, not callable)
    is_factory: bool = False  # factory function (create_xxx, build_xxx, make_xxx)
    requires_interactive_input: bool = False  # needs interactive input (input()/getpass.getpass()/sys.stdin.readline())
    interactive_prompts: list[str] = field(default_factory=list)  # detected interactive prompt texts


@dataclass
class SourceComponent:
    """A "component" extracted from Python source — a module directory or a class."""

    name: str  # e.g. "VideoOperations"
    file_path: str  # e.g. "components/video_ops/video_operations.py"
    description: str  # first paragraph of the docstring
    operations: list[SourceOperation] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # import list (deduplicated local files)
    input_schema: dict = field(default_factory=dict)  # {name: type_hint}
    output_schema: dict = field(default_factory=dict)  # {name: type_hint}
    # ``ClassName`` → ``__init__`` parameters (for pos-converter wrapper injection).
    class_init_params: dict[str, list[ParamSpec]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SourceCodeParser
# ---------------------------------------------------------------------------


class SourceCodeParser:
    """Python source parser.

    Input: a directory path (recursively scans ``.py`` files).
    Output: ``list[SourceComponent]``.

    Parameters
    ----------
    source_dir : str | Path
        The source root directory.
    exclude_patterns : list[str] | None
        File/directory name patterns to exclude (e.g. ``["__pycache__", "*.pyc"]``).
    max_depth : int | None
        Maximum recursion depth (None means unlimited).
    """

    _EXCLUDE_DIRS = frozenset(
        {
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            ".egg-info",
            "dist",
            "build",
            "test",
            "tests",
            "example",
            "examples",
            # clawcodex's own output/config dir — never treat generated
            # bundle artifacts (agent-tools/scripts wrappers, etc.) as
            # source to be re-parsed.
            ".clawcodex",
        }
    )

    # Patterns appended to user exclude_patterns for test/example filtering.
    # target  exact-name dirs, test_*/example_* prefix dirs, *_test/*_tests/
    # *_example/*_examples suffix dirs — without matching unrelated names
    # like "latest" or "attest" that a bare "*test*" glob would catch.
    _DEFAULT_EXCLUDE_PATTERNS = [
        "test_*",
        "*_test",
        "*_tests",
        "example_*",
        "*_example",
        "*_examples",
    ]

    def __init__(
        self,
        source_dir: str | Path,
        *,
        exclude_patterns: list[str] | None = None,
        max_depth: int | None = None,
        extern_only: bool = True,
    ) -> None:
        self._source_dir = Path(source_dir).resolve()
        self._exclude_patterns = (exclude_patterns or []) + self._DEFAULT_EXCLUDE_PATTERNS
        self._max_depth = max_depth
        self._extern_only = extern_only
        self._parsed: list[SourceComponent] | None = None

    # ---- public API -------------------------------------------------------

    def parse(self) -> list[SourceComponent]:
        """Parse the source directory and return the component list."""
        if self._parsed is not None:
            return self._parsed

        if not self._source_dir.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {self._source_dir}")

        components: list[SourceComponent] = []
        self._walk_module(self._source_dir, depth=0, components=components)

        self._parsed = components
        return self._parsed

    def parse_file(self, file_path: str | Path) -> list[SourceOperation]:
        """Parse a single Python file and return the operation list."""
        path = Path(file_path).resolve()
        if not path.is_file() or path.suffix != ".py":
            return []

        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            logger.warning("Syntax error in %s: %s", path, exc)
            return []

        lines = source.splitlines()
        operations: list[SourceOperation] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_ops, _ = self._extract_class(path, node, lines)
                operations.extend(cls_ops)

        operations.extend(self._extract_top_functions(path, tree, lines))

        return operations

    # ---- directory traversal ----------------------------------------------

    def _walk_module(
        self,
        dir_path: Path,
        depth: int,
        components: list[SourceComponent],
    ) -> None:
        """Recursively scan a directory and collect SourceComponents."""
        if self._max_depth is not None and depth > self._max_depth:
            return

        # Find __init__.py for package-level docstring
        init_file = dir_path / "__init__.py"
        package_desc = ""
        if init_file.is_file():
            try:
                init_source = init_file.read_text(encoding="utf-8")
                init_tree = ast.parse(init_source, filename=str(init_file))
                package_desc = self._extract_module_docstring(init_tree)
            except (SyntaxError, UnicodeDecodeError) as exc:
                logger.warning("Skipping package docstring for %s: %s", init_file, exc)

        # Gather all operations from .py files in this directory
        all_ops: list[SourceOperation] = []
        all_class_inits: dict[str, list[ParamSpec]] = {}
        all_deps: set[str] = set()

        py_files = sorted(dir_path.glob("*.py"))
        for py_file in py_files:
            if py_file.name in ("__init__.py",) or self._should_exclude(py_file.name):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
                lines = source.splitlines()
            except (SyntaxError, UnicodeDecodeError) as exc:
                logger.warning("Skipping %s: %s", py_file, exc)
                continue

            # Extract imports
            file_deps = self._extract_imports(tree)
            all_deps.update(file_deps)

            # Extract class definitions
            file_ops: list[SourceOperation] = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    ops, init_params = self._extract_class(py_file, node, lines)
                    if init_params:
                        all_class_inits[node.name] = init_params
                    file_ops.extend(ops)

            # Extract top-level functions
            top_ops = self._extract_top_functions(py_file, tree, lines)
            file_ops.extend(top_ops)

            # Apply extern_only filtering per-file
            if self._extern_only:
                exported = self._parse_all_export(tree)
                if exported is not None:
                    # __all__ defined: filter by name + docstring
                    # Exclude dunder methods (__init__, __call__, etc.) —
                    # Python protocol methods, not standalone external interfaces.
                    file_ops = [
                        op
                        for op in file_ops
                        if op.name != "main"
                        and not op.name.startswith("__")
                        and not op.name.startswith("test")
                        and op.has_docstring
                        and (
                            (op.class_name is not None and op.class_name in exported)
                            or (op.class_name is None and op.name in exported)
                        )
                    ]
                else:
                    # No __all__: filter by docstring only
                    # Exclude dunder methods (__init__, __call__, etc.) —
                    # Python protocol methods, not standalone external interfaces.
                    # Exclude test* — test methods are never external API.
                    # Exclude main — CLI entry points, not library API.
                    file_ops = [
                        op
                        for op in file_ops
                        if op.name != "main"
                        and not op.name.startswith("__")
                        and not op.name.startswith("test")
                        and op.has_docstring
                    ]

            all_ops.extend(file_ops)

        # Build a component for this directory.
        # Derive the component name from the relative path under source_dir
        # so it is globally unique — two directories with the same basename
        # at different levels (e.g. package/harness vs tools/harness)
        # must not produce the same component name.
        try:
            rel_path = dir_path.relative_to(self._source_dir)
        except ValueError:
            rel_path = dir_path
        if str(rel_path) == ".":
            # The source_dir itself contains .py files — use its basename.
            component_name = self._source_dir.name.replace("-", "_").replace(" ", "_")
        else:
            component_name = str(rel_path).replace("\\", "/").replace("/", ".").replace("-", "_").replace(" ", "_")
        if all_ops:
            input_schema, output_schema = self._build_io_schema(all_ops)

            components.append(
                SourceComponent(
                    name=component_name,
                    file_path=str(dir_path.relative_to(self._source_dir.parent)),
                    description=package_desc or f"Module: {dir_path.name}",
                    operations=all_ops,
                    dependencies=sorted(all_deps),
                    input_schema=input_schema,
                    output_schema=output_schema,
                    class_init_params=all_class_inits,
                )
            )

        # Recurse into subdirectories
        for child in sorted(dir_path.iterdir()):
            if child.is_dir() and not self._should_exclude(child.name):
                self._walk_module(child, depth + 1, components)

    def _should_exclude(self, name: str) -> bool:
        """Check whether this file/directory name should be excluded."""
        if name in self._EXCLUDE_DIRS:
            return True
        for pattern in self._exclude_patterns:
            import fnmatch

            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    # ---- class / function extraction --------------------------------------

    def _extract_class(
        self,
        file_path: Path,
        cls_node: ast.ClassDef,
        lines: list[str],
    ) -> tuple[list[SourceOperation], list[ParamSpec]]:
        """Extract public methods and ``__init__`` parameters from an AST class definition."""
        operations: list[SourceOperation] = []
        init_params: list[ParamSpec] = []

        for node in ast.iter_child_nodes(cls_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "__init__":
                    init_params = self._infer_params(node.args)
                    continue
                # Skip private / dunder methods
                if node.name.startswith("_") and node.name not in (
                    "__call__",
                    "__enter__",
                    "__exit__",
                ):
                    continue
                op = self._extract_operation(node, lines, file_path=file_path)
                if op:
                    op.class_name = cls_node.name
                    operations.append(op)

        return operations, init_params

    def _extract_top_functions(
        self,
        file_path: Path,
        module_node: ast.Module,
        lines: list[str],
    ) -> list[SourceOperation]:
        """Extract top-level functions from the module."""
        operations: list[SourceOperation] = []

        for node in ast.iter_child_nodes(module_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                op = self._extract_operation(node, lines, file_path=file_path)
                if op:
                    operations.append(op)

        return operations

    def _extract_operation(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        file_path: Path | None = None,
    ) -> SourceOperation | None:
        """Extract a SourceOperation from an AST function/method node.

        Args:
            node: The AST function or async function definition node.
            lines: The source file split into lines.
            file_path: The source file path, used to populate the ``file_stem`` field.
        """
        # docstring
        docstring = ast.get_docstring(node) or ""
        description, params_from_doc = self._parse_docstring(docstring)

        # Parameters from AST
        ast_params = self._infer_params(node.args)

        # Merge: docstring-derived descriptions enrich AST parameters
        doc_param_map = {p.name: p for p in params_from_doc}
        for ap in ast_params:
            if ap.name in doc_param_map:
                dp = doc_param_map[ap.name]
                if dp.description:
                    ap.description = dp.description
                if dp.type_hint and not ap.type_hint:
                    ap.type_hint = dp.type_hint

        # Return type
        return_type = self._resolve_type_hint(node.returns)

        # Source code snippet
        source_code = self._get_source_code(lines, node)

        has_doc = bool(docstring and docstring.strip())

        is_property = any(
            (isinstance(d, ast.Name) and d.id == "property") or (isinstance(d, ast.Attribute) and d.attr == "property")
            for d in node.decorator_list
        )
        if is_property:
            ast_params = []

        has_factory_prefix = node.name.startswith(tuple(_FACTORY_PREFIXES))
        has_complex_return_type = (
            return_type is not None
            and return_type.strip() not in _SIMPLE_RETURN_TYPES
            and not return_type.strip().startswith(("list[", "dict[", "tuple[", "set["))
        )
        is_factory = has_factory_prefix and has_complex_return_type

        requires_interactive_input, interactive_prompts = detect_interactive_input(node)

        return SourceOperation(
            name=node.name,
            description=description or node.name,
            parameters=ast_params,
            return_type=return_type,
            source_code=source_code,
            file_stem=file_path.stem if file_path else "",
            has_docstring=has_doc,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_async_generator=is_async_generator_operation(node, return_type),
            is_property=is_property,
            is_factory=is_factory,
            requires_interactive_input=requires_interactive_input,
            interactive_prompts=interactive_prompts,
        )

    # ---- docstring parsing ------------------------------------------------

    def _parse_docstring(self, docstring: str | None) -> tuple[str, list[ParamSpec]]:
        """Parse a docstring, extracting the first-paragraph description and the parameter list.

        Supports Google / NumPy / reST / Chinese ``参数:`` styles, falling back to
        the first plain-text paragraph.
        """
        if not docstring:
            return "", []

        description = self._get_first_paragraph(docstring)
        params: list[ParamSpec] = []

        # Try Google style: Args: / Returns:
        params = self._parse_google_style(docstring)
        if params:
            return description, params

        # Try NumPy style: Parameters\\n---\\n
        params = self._parse_numpy_style(docstring)
        if params:
            return description, params

        # Try reST: :param name: desc / :type name: type
        params = self._parse_rest_style(docstring)
        if params:
            return description, params

        # Try Chinese style: 参数: / 参数：
        params = self._parse_chinese_style(docstring)
        if params:
            return description, params

        return description, []

    @staticmethod
    def _parse_all_export(tree: ast.Module) -> set[str] | None:
        """Parse ``__all__`` from the module AST and return the set of exported names, or None.

        Returns None if ``__all__`` is dynamically evaluated (not a list/tuple literal),
        or if the module does not define ``__all__``.
        """
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value = node.value
                    if isinstance(value, (ast.List, ast.Tuple)):
                        names: set[str] = set()
                        for elt in value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                names.add(elt.value)
                        return names
                    return None  # dynamic __all__, can't parse
        return None

    def _get_first_paragraph(self, text: str) -> str:
        """Extract the first paragraph of the text (truncated at a blank line)."""
        lines = text.strip().split("\n")
        para: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped and para:
                break
            if stripped:
                para.append(stripped)
        return " ".join(para).strip()

    def _parse_google_style(self, docstring: str) -> list[ParamSpec]:
        """Parse a Google-style ``Args:`` section."""
        params: list[ParamSpec] = []
        lines = docstring.split("\n")

        in_args = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("args:"):
                in_args = True
                continue
            if in_args:
                if not stripped or stripped.startswith(("returns:", "raises:", "yields:")):
                    break
                # Match "name (type): description" or "name: description"
                match = _DOC_PARAM_LINE_RE.match(stripped)
                if match:
                    name, type_str, desc = match.groups()
                    desc_stripped = desc.strip()
                    params.append(
                        ParamSpec(
                            name=name,
                            type_hint=_resolve_doc_param_type_hint(type_str, desc_stripped),
                            description=desc_stripped,
                        )
                    )
        return params

    def _parse_chinese_style(self, docstring: str) -> list[ParamSpec]:
        """Parse Chinese-style ``参数:`` / ``参数：`` sections."""
        params: list[ParamSpec] = []
        in_params = False

        for line in docstring.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("参数:", "参数：")):
                in_params = True
                continue
            if not in_params:
                continue
            if not stripped:
                continue
            lowered = stripped.lower()
            if any(lowered.startswith(marker) for marker in _DOC_SECTION_STOP_MARKERS):
                break
            match = _DOC_PARAM_LINE_RE.match(stripped)
            if match:
                name, type_str, desc = match.groups()
                desc_stripped = desc.strip()
                params.append(
                    ParamSpec(
                        name=name,
                        type_hint=_resolve_doc_param_type_hint(type_str, desc_stripped),
                        description=desc_stripped,
                    )
                )
        return params

    def _parse_numpy_style(self, docstring: str) -> list[ParamSpec]:
        """Parse a NumPy-style ``Parameters`` section."""
        params: list[ParamSpec] = []
        lines = docstring.split("\n")

        in_params = False
        for i, line in enumerate(lines):
            if line.strip().lower() == "parameters":
                # Next line should be a separator line of dashes
                if i + 1 < len(lines) and re.match(r"^[-]+\s*$", lines[i + 1]):
                    in_params = True
                    continue
            if in_params:
                stripped = line.strip()
                if not stripped:
                    if i + 1 < len(lines) and (
                        lines[i + 1].strip().lower() in ("returns", "raises", "yields", "note", "see also")
                        or re.match(r"^[-]+\s*$", lines[i + 1])
                    ):
                        break
                    continue
                # Match "name : type" or "name"
                match = re.match(r"^(\w+)\s*:\s*(.*)", stripped)
                if match:
                    name, rest = match.groups()
                    desc = rest.strip()
                    # Check if next lines are continuation of description
                    params.append(ParamSpec(name=name, type_hint=desc if desc else None))
                elif params:
                    # Continuation of previous param's description
                    pass
        return params

    def _parse_rest_style(self, docstring: str) -> list[ParamSpec]:
        """Parse an reST-style ``:param name: desc`` / ``:type name: type`` section."""
        # Collect explicit types from :type fields first (they usually follow :param).
        doc_types: dict[str, str] = {}
        for line in docstring.split("\n"):
            stripped = line.strip()
            type_match = re.match(r":type\s+(\w+):\s*(.+)", stripped)
            if type_match:
                doc_types[type_match.group(1)] = type_match.group(2).strip()

        params: list[ParamSpec] = []
        for line in docstring.split("\n"):
            stripped = line.strip()
            match = re.match(r":param\s+(\w+):\s*(.*)", stripped)
            if match:
                name, desc = match.groups()
                desc_stripped = desc.strip()
                params.append(
                    ParamSpec(
                        name=name,
                        type_hint=doc_types.get(name) or infer_type_hint_from_description(desc_stripped),
                        description=desc_stripped,
                    )
                )
        return params

    def _extract_module_docstring(self, tree: ast.Module) -> str:
        """Extract the module-level docstring."""
        docstring = ast.get_docstring(tree)
        if docstring:
            return self._get_first_paragraph(docstring)
        return ""

    # ---- parameter inference ----------------------------------------------

    def _infer_params(self, args: ast.arguments) -> list[ParamSpec]:
        """Extract the parameter list from an AST ``arguments`` node."""
        params: list[ParamSpec] = []
        defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)

        for i, arg in enumerate(args.args):
            if arg.arg in ("self", "cls"):
                continue
            name = arg.arg
            type_hint = self._resolve_type_hint(arg.annotation)
            default = defaults[i] if i < len(defaults) else None
            required = default is None
            params.append(
                ParamSpec(
                    name=name,
                    type_hint=type_hint,
                    default=ast.unparse(default) if default is not None else None,
                    required=required,
                )
            )

        # Handle keyword-only arguments (after * or *args)
        kw_defaults = list(args.kw_defaults) if args.kw_defaults else []
        # Pad kw_defaults to match kwonlyargs length (defaults align to the right)
        kw_defaults = [None] * (len(args.kwonlyargs) - len(kw_defaults)) + kw_defaults
        for i, arg in enumerate(args.kwonlyargs):
            name = arg.arg
            type_hint = self._resolve_type_hint(arg.annotation)
            default = kw_defaults[i] if i < len(kw_defaults) else None
            required = default is None
            params.append(
                ParamSpec(
                    name=name,
                    type_hint=type_hint,
                    default=ast.unparse(default) if default is not None else None,
                    required=required,
                )
            )

        # Handle *args, **kwargs
        if args.vararg:
            params.append(
                ParamSpec(
                    name=f"*{args.vararg.arg}",
                    type_hint=self._resolve_type_hint(args.vararg.annotation),
                    required=False,
                )
            )
        if args.kwarg:
            params.append(
                ParamSpec(
                    name=f"**{args.kwarg.arg}",
                    type_hint=self._resolve_type_hint(args.kwarg.annotation),
                    required=False,
                )
            )

        return params

    @staticmethod
    def _resolve_type_hint(annotation: ast.AST | None) -> str | None:
        """Convert an AST type annotation to a string."""
        if annotation is None:
            return None
        try:
            return ast.unparse(annotation)
        except Exception:
            return None

    # ---- import analysis --------------------------------------------------

    def _extract_imports(self, tree: ast.Module) -> list[str]:
        """Extract local module references from import statements."""
        deps: list[str] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if self._is_local_import(alias.name):
                        deps.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and self._is_local_import(node.module):
                    for alias in node.names:
                        deps.append(f"{node.module}.{alias.name}")

        return deps

    def _is_local_import(self, module_name: str) -> bool:
        """Check whether this is a local module import (relative to source_dir)."""
        # Relative imports
        if module_name.startswith("."):
            return True
        # Check if module path exists relative to source_dir
        module_path = module_name.replace(".", "/")
        potential_path = self._source_dir / f"{module_path}.py"
        potential_pkg = self._source_dir / module_path / "__init__.py"
        return potential_path.exists() or potential_pkg.exists()

    # ---- source code extraction -------------------------------------------

    @staticmethod
    def _get_source_code(lines: list[str], node: ast.AST) -> str:
        """Extract the source snippet for the given AST node from the file lines."""
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            start = node.lineno - 1
            end = node.end_lineno
            return "\n".join(lines[start:end])
        return ""

    # ---- IO schema --------------------------------------------------------

    @staticmethod
    def _build_io_schema(
        operations: list[SourceOperation],
    ) -> tuple[dict, dict]:
        """Build the input/output schemas from the operation list.

        Returns:
            (input_schema, output_schema) tuple:
            input_schema  → {parameter name: type hint}  e.g. ``{"path": "str"}``
            output_schema → {method name: return type}   e.g. ``{"encode": "bool"}``
        """
        input_schema: dict[str, str] = {}
        output_schema: dict[str, str] = {}

        for op in operations:
            for param in op.parameters:
                if param.type_hint:
                    input_schema[param.name] = param.type_hint
            if op.return_type:
                output_schema[op.name] = op.return_type

        return input_schema, output_schema
