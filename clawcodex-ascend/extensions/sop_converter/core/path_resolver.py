"""Path resolution utilities for SOP converter.

Provides shared functions for resolving Python module paths and inferring
additional sys.path entries needed for correct import resolution.
"""

# pylint: disable=too-many-nested-blocks
from __future__ import annotations

import re
from pathlib import Path


def resolve_source_file(source_dir: str, module_name: str) -> Path:
    """Resolve a module name to its source file path."""
    return Path(source_dir) / Path(*module_name.split(".")).with_suffix(".py")


_BACKEND_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+backend(?:\.|\s)|import\s+backend(?:\.|\s|$))",
    re.MULTILINE,
)


def infer_extra_sys_path_entries(source_dir: str, module_name: str) -> list[str]:
    """Return subproject roots required for external package imports.

    Handles three scenarios:
    1. ``from backend.*`` imports: Some SDK apps use a top-level ``backend``
       package relative to their own project directory.
    2. Self-referential absolute imports: SDKs that use ``from <package_name>.*``
       where <package_name> matches the source_dir basename (e.g. ``demo_sdk``).
    3. Sibling ``src/`` layout: a script next to ``src/`` does bare imports
       (``from run_data_pipeline import ...``) while the module lives under
       ``src/`` (common demo layout).

    In all cases, inject the needed directory into generated wrapper scripts
    / schema import paths so the imports can resolve correctly.
    """
    source_file = resolve_source_file(source_dir, module_name)
    if not source_file.is_file():
        return []

    try:
        text = source_file.read_text(encoding="utf-8")
    except OSError:
        return []

    extra_paths: list[str] = []
    root = Path(source_dir).resolve()

    if _BACKEND_IMPORT_RE.search(text):
        current = source_file.parent.resolve()
        while True:
            backend_dir = current / "backend"
            if backend_dir.is_dir() and any(backend_dir.rglob("*.py")):
                extra_paths.append(str(current))
                break
            if current == root:
                break
            parent = current.parent
            if parent == current:
                break
            current = parent

    package_name = root.name
    if package_name:
        self_import_re = re.compile(
            rf"^\s*(?:from\s+{re.escape(package_name)}(?:\.|\s)|import\s+{re.escape(package_name)}(?:\.|\s|$))",
            re.MULTILINE,
        )
        if self_import_re.search(text):
            parent_dir = root.parent
            if parent_dir != root and str(parent_dir) not in extra_paths:
                extra_paths.append(str(parent_dir))
        else:
            init_file = source_file.parent / "__init__.py"
            if init_file.is_file():
                try:
                    init_text = init_file.read_text(encoding="utf-8")
                    if self_import_re.search(init_text):
                        parent_dir = root.parent
                        if parent_dir != root and str(parent_dir) not in extra_paths:
                            extra_paths.append(str(parent_dir))
                except OSError:
                    pass

    # ponytail: only the module's own sibling src/, never invent paths
    src_dir = source_file.parent / "src"
    if src_dir.is_dir() and any(src_dir.rglob("*.py")):
        src_str = str(src_dir.resolve())
        if src_str not in extra_paths:
            extra_paths.append(src_str)

    return extra_paths


def format_extra_sys_path_inserts(
    extra_sys_path_entries: list[str],
    *,
    normalizer: str | None = None,
) -> str:
    """Render ``sys.path.insert`` lines for subproject roots."""
    if not extra_sys_path_entries:
        return ""
    if normalizer:
        lines = [f"sys.path.insert(0, {normalizer}({entry!r}))" for entry in extra_sys_path_entries]
    else:
        lines = [f"sys.path.insert(0, {entry!r})" for entry in extra_sys_path_entries]
    return "\n".join(lines)
