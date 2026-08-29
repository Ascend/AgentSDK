# ruff: noqa
# pylint: skip-file
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

"""Unit tests for :mod:`extensions.sop_converter.tool_registry_bridge`.

Covers the bridge that converts parsed source operations into
executable agent tools with bash-callable wrapper scripts.

* :func:`_type_hint_to_json_type` — Python type-hint → JSON Schema.
* :func:`_strip_optional_union` — Optional/Union/X | None reduction.
* :func:`_to_kebab_case` — dot/snake → kebab-case conversion.
* :func:`_resolve_module_path` — dotted module path from file path.
* :func:`_script_name_for_class` / :func:`_script_name_for_functions`
  — deterministic script filenames.
* :func:`_generate_wrapper_script` — wrapper script creation.
* :func:`_enrich_bridge_params` — json_args injection.
* :func:`operation_to_spec` — single-operation → AgentToolSpec.
* :func:`register_component_tools` — bulk registration with name map.
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from clawcodex_ext.agent.tool_authoring.call_handlers.bash import execute_bash
from extensions.sop_converter import tool_registry_bridge as trb
from extensions.sop_converter.core.agent_catalog_resolver import HOME_ROOT_ENV
from extensions.sop_converter.resource_catalog import (
    DUAL_WRITE_ENV,
    PAYLOAD_REF_ENV,
    SESSION_ID_ENV,
    ResourceCatalog,
)
from extensions.sop_converter.core.source_parser import (
    ParamSpec,
    SourceComponent,
    SourceOperation,
    SourceCodeParser,
)
from extensions.sop_converter.tool_registry_bridge import (
    _coerce_param_expression,
    _enrich_bridge_params,
    _generate_cli_handler_stub,
    _generate_method_stub,
    _generate_wrapper_script,
    _infer_extra_sys_path_entries,
    _is_cli_handler_op,
    _merge_init_and_method_params,
    _param_signature_parts,
    _parse_cli_dispatch_map,
    _resolve_module_path,
    _script_name_for_class,
    _script_name_for_functions,
    _strip_optional_union,
    _to_kebab_case,
    _type_hint_to_json_type,
    operation_to_spec,
    register_component_tools,
)


def _make_param(
    name: str,
    type_hint: str | None = None,
    *,
    required: bool = True,
    default: object = None,
    description: str = "",
) -> ParamSpec:
    return ParamSpec(
        name=name,
        type_hint=type_hint,
        default=default,
        required=required,
        description=description,
    )


def _make_op(
    name: str = "do_thing",
    *,
    description: str = "A method that does a thing.",
    parameters: list[ParamSpec] | None = None,
    return_type: str | None = None,
    class_name: str | None = None,
    file_stem: str = "things",
    is_async: bool = False,
    is_async_generator: bool = False,
) -> SourceOperation:
    return SourceOperation(
        name=name,
        description=description,
        parameters=parameters or [],
        return_type=return_type,
        class_name=class_name,
        file_stem=file_stem,
        is_async=is_async,
        is_async_generator=is_async_generator,
    )


def _make_component(
    name: str = "comp",
    file_path: str = "comp/things",
    operations: list[SourceOperation] | None = None,
) -> SourceComponent:
    return SourceComponent(
        name=name,
        file_path=file_path,
        description="A test component",
        operations=operations or [],
    )


def _catalog_payload(call_impl: str, flag: str) -> dict:
    parts = shlex.split(call_impl)
    idx = parts.index(flag)
    return json.loads(parts[idx + 1])


def _isolated_dirs() -> tuple[object, Path, Path]:
    """Return (cleanup, fake_tool_dir, fake_scripts_dir).

    Patches the module's TOOL_DIR and SCRIPTS_DIR to temp paths.
    """
    tmp = tempfile.TemporaryDirectory()
    tool_dir = Path(tmp.name) / "tools"
    scripts_dir = tool_dir / "scripts"
    tool_dir.mkdir()
    scripts_dir.mkdir()

    def _cleanup() -> None:
        for p in patches:
            p.stop()
        tmp.cleanup()

    patches = [
        patch.object(trb, "TOOL_DIR", tool_dir),
        patch.object(trb, "SCRIPTS_DIR", scripts_dir),
        # Also patch the imported name on the upstream module, since
        # SCRIPTS_DIR = TOOL_DIR / "scripts" evaluated at import time.
        patch(
            "clawcodex_ext.agent.tool_authoring.persistence.TOOL_DIR",
            tool_dir,
        ),
    ]
    for p in patches:
        p.start()
    return _cleanup, tool_dir, scripts_dir


# ---------------------------------------------------------------------------
# _strip_optional_union
# ---------------------------------------------------------------------------


class TestStripOptionalUnion(unittest.TestCase):
    def test_none_input(self) -> None:
        # None / empty → returns the empty string unchanged.
        self.assertEqual(_strip_optional_union(""), "")

    def test_passthrough_non_optional(self) -> None:
        self.assertEqual(_strip_optional_union("str"), "str")

    def test_optional_typing(self) -> None:
        self.assertEqual(_strip_optional_union("Optional[int]"), "int")

    def test_union_typing(self) -> None:
        self.assertEqual(
            _strip_optional_union("Union[str, None]"),
            "str",
        )

    def test_pipe_union(self) -> None:
        self.assertEqual(_strip_optional_union("int | None"), "int")

    def test_nested_pipe_union(self) -> None:
        # Recursive: outer pipe is split, inner is also handled.
        self.assertEqual(
            _strip_optional_union("dict[str, Foo] | None"),
            "dict[str, Foo]",
        )

    def test_nonetype_recognised(self) -> None:
        # "NoneType" is treated the same as "None".
        self.assertEqual(
            _strip_optional_union("Optional[NoneType, str]"),
            "str",
        )


# ---------------------------------------------------------------------------
# _type_hint_to_json_type
# ---------------------------------------------------------------------------


class TestTypeHintToJsonType(unittest.TestCase):
    def test_none_input_defaults_to_string(self) -> None:
        self.assertEqual(_type_hint_to_json_type(None), "string")

    def test_empty_string_defaults_to_string(self) -> None:
        self.assertEqual(_type_hint_to_json_type(""), "string")

    def test_primitive_types(self) -> None:
        for hint, expected in [
            ("str", "string"),
            ("int", "integer"),
            ("float", "number"),
            ("bool", "boolean"),
        ]:
            with self.subTest(hint=hint):
                self.assertEqual(_type_hint_to_json_type(hint), expected)

    def test_collection_types(self) -> None:
        for hint, expected in [
            ("list", "array"),
            ("List[int]", "array"),
            ("Dict[str, int]", "object"),
            ("Mapping[str, Any]", "object"),
        ]:
            with self.subTest(hint=hint):
                self.assertEqual(_type_hint_to_json_type(hint), expected)

    def test_unknown_type_falls_back_to_string(self) -> None:
        self.assertEqual(_type_hint_to_json_type("MyCustomType"), "string")

    def test_uuid_types_map_to_string(self) -> None:
        for hint in ["UUID", "UUID4", "Optional[UUID]", "UUID | None"]:
            with self.subTest(hint=hint):
                self.assertEqual(_type_hint_to_json_type(hint), "string")

    def test_optional_typing_reduces_first(self) -> None:
        self.assertEqual(
            _type_hint_to_json_type("Optional[int]"),
            "integer",
        )
        self.assertEqual(
            _type_hint_to_json_type("str | None"),
            "string",
        )

    def test_iterable_returns_array(self) -> None:
        self.assertEqual(
            _type_hint_to_json_type("Iterable[str]"),
            "array",
        )
        self.assertEqual(
            _type_hint_to_json_type("Sequence[int]"),
            "array",
        )


# ---------------------------------------------------------------------------
# _to_kebab_case
# ---------------------------------------------------------------------------


class TestToKebabCase(unittest.TestCase):
    def test_already_kebab(self) -> None:
        self.assertEqual(_to_kebab_case("docker-build"), "docker-build")

    def test_dot_separator(self) -> None:
        self.assertEqual(_to_kebab_case("LLM.invoke"), "llm-invoke")

    def test_underscore_separator(self) -> None:
        self.assertEqual(_to_kebab_case("video_ops.transcode"), "video-ops-transcode")

    def test_double_underscore(self) -> None:
        # "__" also acts as a separator.
        self.assertEqual(
            _to_kebab_case("utils__load_config"),
            "utils-load-config",
        )

    def test_multi_level_dot_path(self) -> None:
        self.assertEqual(
            _to_kebab_case("foundation.LLM.invoke"),
            "foundation-llm-invoke",
        )

    def test_camelcase_preserved_as_one_word(self) -> None:
        # "VideoProcessor" → "videoprocessor" (NOT "video-processor").
        self.assertEqual(
            _to_kebab_case("VideoProcessor.transcode"),
            "videoprocessor-transcode",
        )

    def test_strips_leading_trailing_hyphens(self) -> None:
        self.assertEqual(_to_kebab_case("-foo-"), "foo")

    def test_collapses_consecutive_hyphens(self) -> None:
        self.assertEqual(_to_kebab_case("foo--bar---baz"), "foo-bar-baz")


# ---------------------------------------------------------------------------
# _resolve_module_path
# ---------------------------------------------------------------------------


class TestResolveModulePath(unittest.TestCase):
    def test_strips_source_dir_prefix(self) -> None:
        comp = _make_component(file_path="openjiuwen/core/foundation")
        # source_dir = "/some/root/openjiuwen" — file_path is relative
        # to source_dir.parent, so we strip the "openjiuwen" segment.
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "openjiuwen"
            source_dir.mkdir()
            result = _resolve_module_path(comp, str(source_dir), "llm")
        self.assertEqual(result, "core.foundation.llm")

    def test_falls_back_to_full_path(self) -> None:
        # If file_path doesn't start with source_dir name, use it raw.
        comp = _make_component(file_path="other/location")
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "openjiuwen"
            source_dir.mkdir()
            result = _resolve_module_path(comp, str(source_dir), "llm")
        # The file_path is used as-is, then file_stem is appended.
        self.assertEqual(result, "other.location.llm")

    def test_with_dots_in_path(self) -> None:
        comp = _make_component(file_path="openjiuwen/core.sub/foo")
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "openjiuwen"
            source_dir.mkdir()
            result = _resolve_module_path(comp, str(source_dir), "bar")
        self.assertEqual(result, "core.sub.foo.bar")

    def test_with_dot_dir_path(self) -> None:
        # A file_path of "." should not become "..bar" — the parts
        # should just be empty before appending file_stem.
        comp = _make_component(file_path=".")
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "openjiuwen"
            source_dir.mkdir()
            result = _resolve_module_path(comp, str(source_dir), "x")
        # "." → empty parts → append "x" → "x".
        self.assertEqual(result, "x")


# ---------------------------------------------------------------------------
# _infer_extra_sys_path_entries (backend subproject imports)
# ---------------------------------------------------------------------------


class TestBackendSubprojectSysPath(unittest.TestCase):
    def test_detects_backend_subproject_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app_root = root / "AgentSDK" / "data_generation_platform"
            utils_dir = app_root / "backend" / "utils"
            models_dir = app_root / "backend" / "models"
            utils_dir.mkdir(parents=True)
            models_dir.mkdir(parents=True)
            (models_dir / "constants.py").write_text(
                "DEFAULT_CHUNK_SIZE = 500\n",
                encoding="utf-8",
            )
            (utils_dir / "text_utils.py").write_text(
                textwrap.dedent(
                    """\
                    from backend.models.constants import DEFAULT_CHUNK_SIZE

                    def split_text(text: str) -> list[str]:
                        return [text[:DEFAULT_CHUNK_SIZE]]
                    """
                ),
                encoding="utf-8",
            )
            module_name = "AgentSDK.data_generation_platform.backend.utils.text_utils"
            entries = _infer_extra_sys_path_entries(str(root), module_name)
            self.assertEqual(entries, [str(app_root.resolve())])

    def test_skips_openjiuwen_style_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkg = root / "openjiuwen" / "core" / "demo"
            pkg.mkdir(parents=True)
            (pkg / "demo.py").write_text(
                textwrap.dedent(
                    """\
                    def run() -> str:
                        return "ok"
                    """
                ),
                encoding="utf-8",
            )
            entries = _infer_extra_sys_path_entries(str(root), "openjiuwen.core.demo.demo")
            self.assertEqual(entries, [])

    def test_detects_sibling_src_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            demo = root / "llm_finetuning_demo"
            src = demo / "src"
            src.mkdir(parents=True)
            (src / "run_data_pipeline.py").write_text(
                "def run_data_pipeline():\n    return {}\n",
                encoding="utf-8",
            )
            (demo / "run_full_pipeline.py").write_text(
                textwrap.dedent(
                    """\
                    import os, sys
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    from run_data_pipeline import run_data_pipeline

                    def main():
                        run_data_pipeline()
                    """
                ),
                encoding="utf-8",
            )
            entries = _infer_extra_sys_path_entries(str(root), "llm_finetuning_demo.run_full_pipeline")
            self.assertEqual(entries, [str(src.resolve())])

    def test_no_sibling_src_means_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkg = root / "flat_sdk"
            pkg.mkdir()
            (pkg / "api.py").write_text("def ping():\n    return 1\n", encoding="utf-8")
            entries = _infer_extra_sys_path_entries(str(root), "flat_sdk.api")
            self.assertEqual(entries, [])


# ---------------------------------------------------------------------------
# _script_name_for_class / _script_name_for_functions
# ---------------------------------------------------------------------------


class TestScriptNameForClass(unittest.TestCase):
    def test_format(self) -> None:
        # Name is "{class}_{8-char-hash}.py".
        name = _script_name_for_class("foo.bar", "Baz")
        self.assertTrue(name.startswith("Baz_"))
        self.assertTrue(name.endswith(".py"))
        # Total: Baz_ + 8 hex chars + .py = 4 + 8 + 3 = 15 chars
        self.assertEqual(len(name), len("Baz_") + 8 + len(".py"))

    def test_deterministic(self) -> None:
        # Same inputs → same hash → same name.
        self.assertEqual(
            _script_name_for_class("a.b", "X"),
            _script_name_for_class("a.b", "X"),
        )

    def test_different_module_different_name(self) -> None:
        # Same class, different module path → different hash.
        self.assertNotEqual(
            _script_name_for_class("a.b", "X"),
            _script_name_for_class("a.c", "X"),
        )


class TestScriptNameForFunctions(unittest.TestCase):
    def test_format(self) -> None:
        # Format: "{file_stem}_fn_{hash}.py"
        name = _script_name_for_functions("foo.bar", "things")
        self.assertTrue(name.startswith("things_fn_"))
        self.assertTrue(name.endswith(".py"))

    def test_deterministic(self) -> None:
        self.assertEqual(
            _script_name_for_functions("a.b", "x"),
            _script_name_for_functions("a.b", "x"),
        )
