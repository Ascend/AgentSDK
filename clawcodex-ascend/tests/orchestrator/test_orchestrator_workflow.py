#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Unit tests for :mod:`extensions.orchestrator.workflow` and
:mod:`extensions.orchestrator.workflow_store`.

Covers:

* :func:`WorkflowLoader.load` — file-not-found raises
  :class:`WorkflowParseError`; the source path is attached to the
  returned config as ``_source_path``.
* :func:`WorkflowLoader.parse` — front-matter splitting, prompt
  extraction, invalid YAML, non-dict front matter, empty front matter.
* :func:`WorkflowLoader.default_path` — env-var override.
* :class:`WorkflowStore` — singleton semantics, ``load`` / ``current`` /
  ``force_reload`` / ``reset`` lifecycle.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from extensions.orchestrator.workflow import (
    WorkflowLoader,
    WorkflowParseError,
    _split_front_matter,
)
from extensions.orchestrator.workflow_store import (
    WorkflowStore,
    get_workflow_store,
)


# ---------------------------------------------------------------------------
# WorkflowLoader.parse
# ---------------------------------------------------------------------------


class TestWorkflowParse(unittest.TestCase):
    def test_empty_content(self) -> None:
        # Empty content → empty front matter, empty prompt.
        config, prompt = WorkflowLoader.parse("")
        # Default WorkflowConfig fields.
        self.assertIsNotNone(config)
        self.assertEqual(prompt, "")

    def test_no_front_matter(self) -> None:
        # Content without `---` markers → whole thing is the prompt.
        content = "Just a prompt.\nWith two lines."
        config, prompt = WorkflowLoader.parse(content)
        self.assertEqual(prompt, content)
        # No front matter to set anything from — config keeps defaults.
        self.assertIsNotNone(config)

    def test_basic_front_matter(self) -> None:
        content = textwrap.dedent(
            """\
            ---
            tracker:
              kind: github
              owner: octo
              repo: hello
            ---
            This is the prompt body.
            """
        )
        config, prompt = WorkflowLoader.parse(content)
        self.assertEqual(prompt, "This is the prompt body.")
        # The front-matter mapping was loaded into the nested config.
        self.assertEqual(config.tracker.kind, "github")
        self.assertEqual(config.tracker.owner, "octo")
        self.assertEqual(config.tracker.repo, "hello")

    def test_multiline_prompt_body(self) -> None:
        content = textwrap.dedent(
            """\
            ---
            name: wf
            ---
            Line 1.
            Line 2.
            Line 3.
            """
        )
        _, prompt = WorkflowLoader.parse(content)
        self.assertIn("Line 1.", prompt)
        self.assertIn("Line 2.", prompt)
        self.assertIn("Line 3.", prompt)

    def test_empty_front_matter(self) -> None:
        content = textwrap.dedent(
            """\
            ---
            ---
            Just a prompt.
            """
        )
        config, prompt = WorkflowLoader.parse(content)
        self.assertEqual(prompt, "Just a prompt.")
        # Config has its defaults.
        self.assertIsNotNone(config)

    def test_invalid_yaml_raises(self) -> None:
        # Use an obvious YAML error: unbalanced brackets.
        content = textwrap.dedent(
            """\
            ---
            name: [unclosed
            ---
            prompt
            """
        )
        with self.assertRaises(WorkflowParseError) as ctx:
            WorkflowLoader.parse(content)
        self.assertIn("YAML", str(ctx.exception))

    def test_non_dict_front_matter_raises(self) -> None:
        # A top-level list is not a mapping.
        content = textwrap.dedent(
            """\
            ---
            - one
            - two
            ---
            prompt
            """
        )
        with self.assertRaises(WorkflowParseError) as ctx:
            WorkflowLoader.parse(content)
        self.assertIn("mapping", str(ctx.exception))

    def test_missing_closing_marker_yaml_parsed_as_mapping(self) -> None:
        # Only an opening `---` with no closing `---` → the rest goes
        # into the front-matter buffer; YAML then parses it as a mapping
        # (or raises if it isn't valid YAML). Prompt is empty.
        content = textwrap.dedent(
            """\
            ---
            tracker:
              kind: github
            """
        )
        config, prompt = WorkflowLoader.parse(content)
        # The whole content is parsed as front matter; the prompt is empty.
        self.assertEqual(prompt, "")
        # And the front matter is loaded into config.
        self.assertEqual(config.tracker.kind, "github")


class TestSplitFrontMatter(unittest.TestCase):
    def test_empty(self) -> None:
        front, prompt = _split_front_matter("")
        self.assertEqual(front, [])
        self.assertEqual(prompt, [])

    def test_no_opening_marker(self) -> None:
        front, prompt = _split_front_matter("hello\nworld")
        self.assertEqual(front, [])
        self.assertEqual(prompt, ["hello", "world"])

    def test_complete_block(self) -> None:
        front, prompt = _split_front_matter("---\nkey: value\n---\nbody")
        self.assertEqual(front, ["key: value"])
        self.assertEqual(prompt, ["body"])

    def test_only_opening_marker(self) -> None:
        # No closing `---` → all subsequent lines stay in the front
        # buffer; prompt stays empty.
        front, prompt = _split_front_matter("---\nkey: value\nstill yaml")
        self.assertEqual(front, ["key: value", "still yaml"])
        self.assertEqual(prompt, [])


# ---------------------------------------------------------------------------
# WorkflowLoader.load
# ---------------------------------------------------------------------------


class TestWorkflowLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "WORKFLOW.md"

    def test_load_missing_file_raises(self) -> None:
        with self.assertRaises(WorkflowParseError) as ctx:
            WorkflowLoader.load(self.path)
        self.assertIn("not found", str(ctx.exception))

    def test_load_attaches_source_path(self) -> None:
        self.path.write_text(
            textwrap.dedent(
                """\
                ---
                name: wf
                ---
                prompt
                """
            ),
            encoding="utf-8",
        )
        config, _ = WorkflowLoader.load(self.path)
        self.assertEqual(config._source_path, str(self.path))
        self.assertEqual(config.source_path, str(self.path))

    def test_load_uses_path_object(self) -> None:
        self.path.write_text(
            textwrap.dedent(
                """\
                ---
                name: wf
                ---
                prompt
                """
            ),
            encoding="utf-8",
        )
        config, _ = WorkflowLoader.load(str(self.path))  # str input
        self.assertEqual(config._source_path, str(self.path))
        self.assertEqual(config.source_path, str(self.path))

    def test_load_returns_prompt(self) -> None:
        self.path.write_text(
            textwrap.dedent(
                """\
                ---
                name: wf
                ---
                This is the actual prompt.
                """
            ),
            encoding="utf-8",
        )
        _, prompt = WorkflowLoader.load(self.path)
        self.assertEqual(prompt, "This is the actual prompt.")

    def test_load_agent_env(self) -> None:
        self.path.write_text(
            textwrap.dedent(
                """\
                ---
                agent:
                  env:
                    PATH: "/custom/bin:$PATH"
                    MY_VAR: "value"
                ---
                prompt
                """
            ),
            encoding="utf-8",
        )
        config, _ = WorkflowLoader.load(self.path)
        self.assertEqual(config.agent.env["PATH"], "/custom/bin:$PATH")
        self.assertEqual(config.agent.env["MY_VAR"], "value")

    def test_load_agent_env_defaults_to_empty(self) -> None:
        self.path.write_text(
            textwrap.dedent(
                """\
                ---
                agent:
                  max_turns: 100
                ---
                prompt
                """
            ),
            encoding="utf-8",
        )
        config, _ = WorkflowLoader.load(self.path)
        self.assertEqual(config.agent.env, {})


# ---------------------------------------------------------------------------
# WorkflowLoader.default_path
# ---------------------------------------------------------------------------


class TestDefaultPath(unittest.TestCase):
    def test_default_path_is_cwd_workflow_md(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Make sure SYMPHONY_WORKFLOW_PATH is not set.
            os.environ.pop("SYMPHONY_WORKFLOW_PATH", None)
            path = WorkflowLoader.default_path()
            self.assertEqual(path, Path.cwd() / "WORKFLOW.md")

    def test_default_path_respects_env(self) -> None:
        with patch.dict(os.environ, {"SYMPHONY_WORKFLOW_PATH": "/some/custom/WORKFLOW.md"}):
            path = WorkflowLoader.default_path()
            self.assertEqual(path, Path("/some/custom/WORKFLOW.md"))


# ---------------------------------------------------------------------------
# WorkflowStore — singleton
# ---------------------------------------------------------------------------


class TestWorkflowStoreSingleton(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure clean state for every test.
        WorkflowStore.reset()
        self.addCleanup(WorkflowStore.reset)

    def test_two_instances_are_same(self) -> None:
        a = WorkflowStore()
        b = WorkflowStore()
        self.assertIs(a, b)

    def test_get_workflow_store_returns_singleton(self) -> None:
        a = get_workflow_store()
        b = get_workflow_store()
        self.assertIs(a, b)
        self.assertIsInstance(a, WorkflowStore)

    def test_initial_state_is_empty(self) -> None:
        store = WorkflowStore()
        self.assertIsNone(store.config)
        self.assertIsNone(store.prompt_template)
        self.assertIsNone(store.workflow_path)
        self.assertIsNone(store.current())

    def test_load_populates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "WORKFLOW.md"
            path.write_text(
                textwrap.dedent(
                    """\
                    ---
                    tracker:
                      kind: github
                      owner: octo
                      repo: hello
                    ---
                    Hello
                    """
                ),
                encoding="utf-8",
            )
            store = WorkflowStore()
            store.load(str(path))
            self.assertEqual(store.workflow_path, str(path))
            self.assertIsNotNone(store.config)
            self.assertEqual(store.prompt_template, "Hello")
            # current() returns the tuple.
            config, prompt = store.current()
            self.assertEqual(prompt, "Hello")
            self.assertEqual(config.tracker.kind, "github")

    def test_force_reload_uses_stored_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "WORKFLOW.md"
            path.write_text(
                textwrap.dedent(
                    """\
                    ---
                    tracker:
                      kind: github
                    ---
                    v1 prompt
                    """
                ),
                encoding="utf-8",
            )
            store = WorkflowStore()
            store.load(str(path))
            # Mutate the file on disk.
            path.write_text(
                textwrap.dedent(
                    """\
                    ---
                    tracker:
                      kind: linear
                    ---
                    v2 prompt
                    """
                ),
                encoding="utf-8",
            )
            # force_reload should pick up the new content.
            store.force_reload()
            self.assertEqual(store.prompt_template, "v2 prompt")
            self.assertEqual(store.config.tracker.kind, "linear")

    def test_force_reload_without_path_is_noop(self) -> None:
        # No load has happened → no path to reload.
        store = WorkflowStore()
        store.force_reload()  # must not raise
        self.assertIsNone(store.workflow_path)

    def test_reset_clears_class_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "WORKFLOW.md"
            path.write_text(
                textwrap.dedent(
                    """\
                    ---
                    tracker:
                      kind: github
                    ---
                    y
                    """
                ),
                encoding="utf-8",
            )
            store = WorkflowStore()
            store.load(str(path))
            self.assertIsNotNone(store.config)
            store.reset()
            # Class-level state is cleared — a *new* singleton reference
            # observes the empty state. The previous `store` object still
            # holds its own instance attributes set by `load()` (load
            # uses `self._config = ...` which shadows the class attr), so
            # we verify via a fresh reference.
            fresh = WorkflowStore()
            self.assertIsNone(fresh.config)
            self.assertIsNone(fresh.prompt_template)
            self.assertIsNone(fresh.workflow_path)

    def test_get_workflow_store_after_reset_returns_new(self) -> None:
        # First reference.
        first = get_workflow_store()
        WorkflowStore.reset()
        second = get_workflow_store()
        self.assertIsNot(first, second)


if __name__ == "__main__":
    unittest.main()
