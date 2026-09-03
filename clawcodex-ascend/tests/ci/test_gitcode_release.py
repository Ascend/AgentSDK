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

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "scripts" / "ci" / "gitcode_release.py").is_file(),
    reason="standalone ClawCodex release helpers are not part of the AgentSDK runtime migration",
)


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend("scripts/ci")
    return importlib.import_module("gitcode_release")


def test_display_url_redacts_signed_queries(monkeypatch):
    gitcode_release = _load_module(monkeypatch)

    assert (
        gitcode_release._display_url("https://file.gitcode.com/bucket/object?AccessKeyId=test&Signature=signed")
        == "https://file.gitcode.com/bucket/object?<redacted>"
    )
    assert (
        gitcode_release._display_url("https://api.gitcode.com/api/v5/repos/owner/repo?file_name=asset.whl")
        == "https://api.gitcode.com/api/v5/repos/owner/repo?file_name=asset.whl"
    )


def test_upload_file_uses_presigned_put_without_mutating_url(tmp_path, monkeypatch):
    gitcode_release = _load_module(monkeypatch)

    asset = tmp_path / "asset name.whl"
    asset.write_bytes(b"wheel-bytes")
    signed_url = "https://file.gitcode.com/bucket/object?AccessKeyId=test&Expires=1&Signature=signed"
    calls = []

    def fake_request(method, path_or_url, **kwargs):
        calls.append((method, path_or_url, kwargs))
        if method == "GET":
            return {
                "url": signed_url,
                "headers": {
                    "Content-Type": "application/octet-stream",
                    "x-obs-callback": "callback-payload",
                    "x-obs-meta-project-id": 123,
                },
            }
        return {}

    monkeypatch.setattr(gitcode_release, "_request", fake_request)

    gitcode_release._upload_file("owner", "repo", "v0.5.0", asset)

    assert calls[0][0] == "GET"
    assert calls[0][1] == ("/api/v5/repos/owner/repo/releases/v0.5.0/upload_url?file_name=asset%20name.whl")

    method, url, kwargs = calls[1]
    assert method == "PUT"
    assert url == signed_url
    assert kwargs["skip_auth"] is True
    assert kwargs["data"] == b"wheel-bytes"
    assert kwargs["content_type"] == "application/octet-stream"
    assert kwargs["extra_headers"] == {
        "x-obs-callback": "callback-payload",
        "x-obs-meta-project-id": "123",
    }
