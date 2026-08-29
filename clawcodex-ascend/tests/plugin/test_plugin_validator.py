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

from src.plugins.validator import (
    is_valid_manifest,
    validate_manifest,
)


class TestValidateManifest:
    def test_valid_minimal(self):
        errors = validate_manifest({"name": "my-plugin"})
        assert errors == []

    def test_valid_full(self):
        errors = validate_manifest(
            {
                "name": "my-plugin",
                "description": "A test plugin",
                "version": "1.2.3",
                "hooks": {"PreToolUse": []},
                "mcp_servers": {"server1": {}},
                "permissions": ["read", "write"],
                "dependencies": {"other-plugin": "^1.0.0"},
            }
        )
        assert errors == []

    def test_missing_name(self):
        errors = validate_manifest({"description": "no name"})
        assert any(e.field == "name" for e in errors)

    def test_invalid_name_chars(self):
        errors = validate_manifest({"name": "bad name!"})
        assert any(e.field == "name" for e in errors)

    def test_name_starts_digit(self):
        errors = validate_manifest({"name": "1bad"})
        assert any(e.field == "name" for e in errors)

    def test_name_not_string(self):
        errors = validate_manifest({"name": 123})
        assert any(e.field == "name" for e in errors)

    def test_invalid_version(self):
        errors = validate_manifest({"name": "test", "version": "bad"})
        assert any(e.field == "version" for e in errors)

    def test_valid_version_prerelease(self):
        errors = validate_manifest({"name": "test", "version": "1.0.0-beta.1"})
        assert not any(e.field == "version" for e in errors)

    def test_version_not_string(self):
        errors = validate_manifest({"name": "test", "version": 100})
        assert any(e.field == "version" for e in errors)

    def test_description_not_string(self):
        errors = validate_manifest({"name": "test", "description": 42})
        assert any(e.field == "description" for e in errors)

    def test_hooks_not_dict(self):
        errors = validate_manifest({"name": "test", "hooks": "bad"})
        assert any(e.field == "hooks" for e in errors)

    def test_mcp_servers_not_dict(self):
        errors = validate_manifest({"name": "test", "mcp_servers": []})
        assert any(e.field == "mcp_servers" for e in errors)

    def test_permissions_not_list(self):
        errors = validate_manifest({"name": "test", "permissions": "read"})
        assert any(e.field == "permissions" for e in errors)

    def test_unknown_permission(self):
        errors = validate_manifest({"name": "test", "permissions": ["admin"]})
        assert any(e.field == "permissions" for e in errors)

    def test_valid_permissions(self):
        errors = validate_manifest(
            {
                "name": "test",
                "permissions": ["read", "write", "execute", "network", "mcp"],
            }
        )
        assert not any(e.field == "permissions" for e in errors)

    def test_dependencies_not_dict(self):
        errors = validate_manifest({"name": "test", "dependencies": ["dep1"]})
        assert any(e.field == "dependencies" for e in errors)

    def test_not_dict_root(self):
        errors = validate_manifest("not a dict")
        assert any(e.field == "root" for e in errors)


class TestIsValidManifest:
    def test_valid(self):
        assert is_valid_manifest({"name": "test"}) is True

    def test_invalid(self):
        assert is_valid_manifest({}) is False

    def test_invalid_name(self):
        assert is_valid_manifest({"name": "1bad"}) is False
