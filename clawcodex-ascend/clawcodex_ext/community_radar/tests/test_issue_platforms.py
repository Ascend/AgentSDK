#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Tests for issue_platforms module — platform detection, target resolution, IssueClient."""

from __future__ import annotations

import os
from unittest import mock

from clawcodex_ext.community_radar.issue_platforms import (
    _PLATFORMS,
    IssueClient,
    ResolvedTarget,
    _infer_platform_from_url,
    _parse_explicit_repo,
    _resolve_token,
    resolve_target,
)

# ---------------------------------------------------------------------------
# Platform registry
# ---------------------------------------------------------------------------


class TestPlatformRegistry:
    def test_all_platforms_registered(self) -> None:
        assert "gitcode" in _PLATFORMS
        assert "github" in _PLATFORMS
        assert "gitee" in _PLATFORMS

    def test_gitcode_platform_config(self) -> None:
        plat = _PLATFORMS["gitcode"]
        assert plat.name == "gitcode"
        assert "gitcode.com" in plat.default_endpoint
        assert plat.auth_mode == "access_token"
        assert len(plat.token_env_vars) >= 1
        assert "GITCODE_TOKEN" in plat.token_env_vars

    def test_github_platform_config(self) -> None:
        plat = _PLATFORMS["github"]
        assert plat.name == "github"
        assert "github.com" in plat.default_endpoint
        assert plat.auth_mode == "bearer"
        assert "GITHUB_TOKEN" in plat.token_env_vars
        assert "GH_TOKEN" in plat.token_env_vars


# ---------------------------------------------------------------------------
# _infer_platform_from_url
# ---------------------------------------------------------------------------


class TestInferPlatform:
    def test_gitcode_ssh_url(self) -> None:
        assert _infer_platform_from_url("git@gitcode.com:hidden178/repo.git") == "gitcode"

    def test_gitcode_https_url(self) -> None:
        assert _infer_platform_from_url("https://gitcode.com/hidden178/repo") == "gitcode"

    def test_github_ssh_url(self) -> None:
        assert _infer_platform_from_url("git@github.com:owner/repo.git") == "github"

    def test_github_https_url(self) -> None:
        assert _infer_platform_from_url("https://github.com/owner/repo.git") == "github"

    def test_gitee_url(self) -> None:
        assert _infer_platform_from_url("https://gitee.com/owner/repo.git") == "gitee"

    def test_unknown_url(self) -> None:
        assert _infer_platform_from_url("https://unknown.example.com/foo/bar") is None


# ---------------------------------------------------------------------------
# _parse_explicit_repo
# ---------------------------------------------------------------------------


class TestParseExplicitRepo:
    def test_owner_repo(self) -> None:
        platform, owner, repo = _parse_explicit_repo("hidden178/my-repo")
        assert platform is None
        assert owner == "hidden178"
        assert repo == "my-repo"

    def test_platform_owner_repo(self) -> None:
        platform, owner, repo = _parse_explicit_repo("gitcode.com/hidden178/my-repo")
        assert platform == "gitcode"
        assert owner == "hidden178"
        assert repo == "my-repo"

    def test_with_git_suffix(self) -> None:
        _, owner, repo = _parse_explicit_repo("owner/repo.git")
        assert owner == "owner"
        assert repo == "repo"

    def test_trailing_slash(self) -> None:
        _, owner, repo = _parse_explicit_repo("owner/repo/")
        assert owner == "owner"
        assert repo == "repo"


# ---------------------------------------------------------------------------
# _resolve_token
# ---------------------------------------------------------------------------


class TestResolveToken:
    def test_env_var_priority(self) -> None:
        plat = _PLATFORMS["gitcode"]
        with mock.patch.dict(os.environ, {"GITCODE_TOKEN": "env-token"}, clear=True):
            assert _resolve_token(plat) == "env-token"

    def test_fallback_env_var(self) -> None:
        plat = _PLATFORMS["gitcode"]
        with mock.patch.dict(os.environ, {"GITCODE_API_KEY": "api-key-token"}, clear=True):
            assert _resolve_token(plat) == "api-key-token"

    def test_generic_env_fallback(self) -> None:
        plat = _PLATFORMS["gitcode"]
        with mock.patch.dict(os.environ, {"CLAWCODEX_ISSUE_TOKEN": "generic-token"}, clear=True):
            assert _resolve_token(plat) == "generic-token"

    def test_config_fallback(self) -> None:
        plat = _PLATFORMS["gitcode"]
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _resolve_token(plat, "config-token") == "config-token"

    def test_no_token(self) -> None:
        plat = _PLATFORMS["gitcode"]
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _resolve_token(plat) is None


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


class TestResolveTarget:
    def test_no_source_returns_none(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch(
                "clawcodex_ext.community_radar.issue_platforms._detect_from_git_remote",
                return_value=None,
            ),
        ):
            target = resolve_target()
            assert target is None

    def test_with_cli_repo(self) -> None:
        with mock.patch.dict(os.environ, {"GITCODE_TOKEN": "test-token"}, clear=True):
            target = resolve_target(cli_repo="hidden178/test-repo")
            assert target is not None
            assert target.owner == "hidden178"
            assert target.repo == "test-repo"
            assert target.platform.name == "gitcode"
            assert target.api_token == "test-token"

    def test_with_config_repo(self) -> None:
        with mock.patch.dict(os.environ, {"GITCODE_TOKEN": "test-token"}, clear=True):
            target = resolve_target(
                config_target_repo="myowner/myrepo",
            )
            assert target is not None
            assert target.owner == "myowner"
            assert target.repo == "myrepo"

    def test_with_platform_specified(self) -> None:
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "gh-token"}, clear=True):
            target = resolve_target(
                cli_repo="gh-owner/gh-repo",
                cli_platform="github",
            )
            assert target is not None
            assert target.platform.name == "github"
            assert target.owner == "gh-owner"
            assert target.repo == "gh-repo"
            assert target.api_token == "gh-token"

    def test_repo_slug_property(self) -> None:
        with mock.patch.dict(os.environ, {"GITCODE_TOKEN": "t"}, clear=True):
            target = resolve_target(cli_repo="owner/repo")
            assert target is not None
            assert "gitcode" in target.repo_slug
            assert "owner" in target.repo_slug
            assert "repo" in target.repo_slug

    def test_web_url_property(self) -> None:
        with mock.patch.dict(os.environ, {"GITCODE_TOKEN": "t"}, clear=True):
            target = resolve_target(cli_repo="o/r")
            assert target is not None
            assert "gitcode.com" in target.web_url
            assert "/o/r" in target.web_url


# ---------------------------------------------------------------------------
# IssueClient
# ---------------------------------------------------------------------------


class TestIssueClient:
    def test_create_issue_success(self) -> None:
        plat = _PLATFORMS["gitcode"]
        target = ResolvedTarget(platform=plat, owner="o", repo="r", api_token="t")
        client = IssueClient(target)
        with mock.patch.object(
            client,
            "_request",
            return_value={
                "number": 42,
                "html_url": "https://gitcode.com/o/r/issues/42",
            },
        ):
            resp = client.create_issue(title="Test", body="Body", labels=["community-radar"])
            assert resp is not None
            assert resp["number"] == 42
        client.close()

    def test_list_issues_success(self) -> None:
        plat = _PLATFORMS["gitcode"]
        target = ResolvedTarget(platform=plat, owner="o", repo="r", api_token="t")
        client = IssueClient(target)
        with mock.patch.object(
            client,
            "_request",
            return_value=[
                {"number": 1, "title": "Test", "body": "<!-- community-radar-id: abc -->"},
            ],
        ):
            issues = client.list_issues(label="community-radar")
            assert len(issues) == 1
            assert issues[0]["number"] == 1
        client.close()

    def test_create_issue_http_error(self) -> None:
        plat = _PLATFORMS["gitcode"]
        target = ResolvedTarget(platform=plat, owner="o", repo="r", api_token="t")
        client = IssueClient(target)
        with mock.patch.object(client, "_request", return_value=None):
            resp = client.create_issue(title="Test", body="Body")
            assert resp is None
        client.close()

    def test_get_issue(self) -> None:
        plat = _PLATFORMS["gitcode"]
        target = ResolvedTarget(platform=plat, owner="o", repo="r", api_token="t")
        client = IssueClient(target)
        with mock.patch.object(
            client,
            "_request",
            return_value={
                "number": 7,
                "title": "Found",
            },
        ):
            issue = client.get_issue(7)
            assert issue is not None
            assert issue["number"] == 7
        client.close()
