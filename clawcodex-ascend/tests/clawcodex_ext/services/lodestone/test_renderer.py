#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for clawcodex_ext.services.lodestone.renderer."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import pytest

from clawcodex_ext.services.lodestone.models import (
    AnchorContext,
    LodestoneAnchor,
    LodestoneConfig,
)
from clawcodex_ext.services.lodestone.renderer import (
    AnchorRenderer,
    TemplateFormatError,
    _render_template,
    _tracker_full_host,
)


def _ctx(cfg: LodestoneConfig | None = None, **kwargs):
    base = LodestoneConfig()
    cfg = cfg or base
    root = kwargs.pop("workspace_root", None)
    if isinstance(root, str):
        from pathlib import Path

        root = Path(root)
    return AnchorContext(
        workspace_root=root,
        session_id="s",
        config=cfg,
        remote_url=kwargs.pop("remote_url", None),
        branch=kwargs.pop("branch", "main"),
        env={},
    )


# ---------------------------------------------------------------------------
# Template substitution
# ---------------------------------------------------------------------------


def test_render_template_basic():
    out = _render_template("vscode://file/{abs}:{line}:{col}", {"abs": "/x", "line": "1", "col": "2"})
    assert out == "vscode://file//x:1:2"


def test_render_template_unknown_placeholder():
    with pytest.raises(TemplateFormatError):
        _render_template("vscode://file/{abs}/{nope}", {"abs": "/x"})


def test_render_template_no_placeholders():
    assert _render_template("literal://x", {}) == "literal://x"


def test_tracker_full_host_short_to_domain():
    assert _tracker_full_host("gitcode", _ctx()) == "gitcode.com"
    assert _tracker_full_host("linear", _ctx()) == "linear.app"
    assert _tracker_full_host("FOO", _ctx()) == "foo.com"
    assert _tracker_full_host("already.full", _ctx()) == "already.full"


# ---------------------------------------------------------------------------
# Renderer with built-in targets
# ---------------------------------------------------------------------------


def test_renderer_text_sink_returns_raw_text():
    cfg = LodestoneConfig(default_editor="vscode")
    anchor = LodestoneAnchor(kind="file_path", raw="x.py:1", file_path="x.py", line=1, column=1)
    target = _ctx_registry().get("vscode", "file_path")
    rendered = AnchorRenderer().render(anchor, target=target, sink="text", ctx=_ctx(cfg))
    # When sink="text" the renderer emits the URL itself — convenient
    # for programmatic consumers like the ``open`` action that need a
    # stable string to feed to the OS default handler.
    assert rendered.rendered == "vscode://file/x.py:1:1"
    assert rendered.sink == "text"
    assert rendered.is_anchor is True


def test_renderer_markdown_sink_emits_link():
    cfg = LodestoneConfig(default_editor="vscode")
    anchor = LodestoneAnchor(kind="file_path", raw="x.py:1", file_path="x.py", line=1, column=1)
    target = _ctx_registry().get("vscode", "file_path")
    rendered = AnchorRenderer().render(anchor, target=target, sink="markdown", ctx=_ctx(cfg))
    assert rendered.rendered.startswith("[x.py:1:1](")
    assert "vscode://file/" in rendered.rendered
    assert rendered.rendered.endswith(":1:1)")


def test_renderer_url_anchor_bypasses_target_lookup():
    a = LodestoneAnchor(kind="url", raw="https://example.com", url="https://example.com")
    rendered = AnchorRenderer().render(a, target=None, sink="markdown")
    assert rendered.rendered == "[https://example.com](https://example.com)"


def test_renderer_no_target_falls_back_to_plain_text():
    a = LodestoneAnchor(kind="file_path", raw="x.py:1", file_path="x.py", line=1, column=1)
    rendered = AnchorRenderer().render(a, target=None, sink="markdown")
    assert rendered.is_anchor is False
    assert "x.py:1" in rendered.rendered


def test_renderer_normalize_auto_to_osc8_in_vscode_term():
    sink = AnchorRenderer().normalize_sink("auto", env={"TERM_PROGRAM": "vscode"})
    assert sink == "osc8"


def test_renderer_normalize_auto_to_markdown_in_normal_term():
    sink = AnchorRenderer().normalize_sink("auto", env={"TERM_PROGRAM": "xterm"})
    assert sink == "markdown"


def test_renderer_markdown_escape_url_with_brackets():
    a = LodestoneAnchor(kind="url", raw="", url="https://example.com/foo(bar)")
    rendered = AnchorRenderer().render(a, target=None, sink="markdown")
    # Parentheses trigger angle-bracket wrapping per CommonMark.
    assert "(" in rendered.rendered
    assert ">" in rendered.rendered


# ---------------------------------------------------------------------------
# Service.resolve_text — splice anchors into text
# ---------------------------------------------------------------------------


def test_service_resolve_text_replaces_anchor_with_markdown():
    from clawcodex_ext.services.lodestone import LodestoneService

    cfg = LodestoneConfig(default_editor="vscode", renderer="markdown")
    svc = LodestoneService(config=cfg)
    out = svc.resolve_text("see x.py:42", workspace_root="/abs")
    # col is defaulting to 1 when None.
    assert "vscode://file/" in out
    assert "[x.py:42]" in out


def test_service_disabled_returns_plain_text():
    from clawcodex_ext.services.lodestone import LodestoneService

    cfg = LodestoneConfig(enabled=False, renderer="markdown")
    svc = LodestoneService(config=cfg)
    out = svc.resolve_text("see x.py:42", workspace_root="/abs")
    assert out == "see x.py:42"


def _ctx_registry():
    from clawcodex_ext.services.lodestone.targets import build_default_registry

    return build_default_registry()
