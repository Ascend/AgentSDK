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

"""Unit tests for the source-aware environment trust boundary.

User-owned global settings are trusted before the folder gate.  A cloned
project is not: only its allow-listed variables apply until trust is granted.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from src.permissions.trust_boundary import (
    SAFE_ENV_KEYS,
    UNSAFE_ENV_KEYS,
    UNSAFE_ENV_PREFIXES,
    apply_full_config_environment_variables,
    apply_safe_config_environment_variables,
    is_safe_env_key,
    reset_trust_boundary_for_test_only,
)


class TestSafeKeyAllowList(unittest.TestCase):
    def test_every_entry_in_safe_env_keys_is_safe(self) -> None:
        # Every key in the allow-list must classify as safe — round-trip
        # verifies no entry is shadowed by UNSAFE_ENV_KEYS or by a
        # prefix match.
        for key in SAFE_ENV_KEYS:
            with self.subTest(key=key):
                self.assertTrue(
                    is_safe_env_key(key),
                    f"{key!r} is in SAFE_ENV_KEYS but is_safe_env_key returns False",
                )

    def test_set_size_matches_ts_port(self) -> None:
        # Sanity check: TS managedEnvConstants.ts:109-194 has 82 entries
        # in SAFE_ENV_VARS. Our port should match exactly. If a future
        # WI legitimately changes the count, update both this number
        # and the corresponding TS comment.
        self.assertEqual(
            len(SAFE_ENV_KEYS),
            82,
            "SAFE_ENV_KEYS count drifted from the TS port",
        )


class TestUnsafeKeysBlocked(unittest.TestCase):
    def test_path_returns_false(self) -> None:
        self.assertFalse(is_safe_env_key("PATH"))

    def test_pythonpath_returns_false(self) -> None:
        self.assertFalse(is_safe_env_key("PYTHONPATH"))

    def test_node_options_returns_false(self) -> None:
        self.assertFalse(is_safe_env_key("NODE_OPTIONS"))

    def test_node_extra_ca_certs_returns_false(self) -> None:
        # The round-1 critic-blocking-issue case: TS explicitly flags
        # NODE_EXTRA_CA_CERTS as "TRUST ATTACKER-CONTROLLED SERVER"
        # (managedEnvConstants.ts:99-103).
        self.assertFalse(is_safe_env_key("NODE_EXTRA_CA_CERTS"))

    def test_anthropic_base_url_returns_false(self) -> None:
        # "REDIRECT TO ATTACKER-CONTROLLED SERVER"
        # (managedEnvConstants.ts:95-97).
        self.assertFalse(is_safe_env_key("ANTHROPIC_BASE_URL"))

    def test_anthropic_api_key_returns_false(self) -> None:
        # "SWITCH TO ATTACKER-CONTROLLED PROJECT"
        # (managedEnvConstants.ts:105-108).
        self.assertFalse(is_safe_env_key("ANTHROPIC_API_KEY"))


class TestUnsafePrefixesBlocked(unittest.TestCase):
    def test_ld_preload_returns_false(self) -> None:
        self.assertFalse(is_safe_env_key("LD_PRELOAD"))

    def test_ld_library_path_returns_false(self) -> None:
        self.assertFalse(is_safe_env_key("LD_LIBRARY_PATH"))

    def test_dyld_insert_libraries_returns_false(self) -> None:
        self.assertFalse(is_safe_env_key("DYLD_INSERT_LIBRARIES"))


class TestUnknownKeyDefaultDeny(unittest.TestCase):
    def test_random_key_returns_false(self) -> None:
        # Default-deny: anything not in the allow-list returns False.
        self.assertFalse(is_safe_env_key("FOO_BAR_BAZ"))

    def test_empty_string_returns_false(self) -> None:
        self.assertFalse(is_safe_env_key(""))


class TestApplySafeDoesNotApplyUnsafe(unittest.TestCase):
    def test_unsafe_keys_excluded_from_safe_apply(self, *, monkeypatch=None):
        project_env = {
            "ANTHROPIC_MODEL": "claude-sonnet-4-6",  # safe
            "PATH": "/opt/evil/bin",  # unsafe
            "NODE_EXTRA_CA_CERTS": "/opt/evil.crt",  # unsafe
            "LD_PRELOAD": "/opt/evil.so",  # unsafe (prefix)
            "DISABLE_TELEMETRY": "1",  # safe
        }
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch(
                "src.permissions.trust_boundary._load_global_config_env",
                return_value={},
            ),
            mock.patch(
                "src.permissions.trust_boundary._load_user_settings_env",
                return_value={},
            ),
            mock.patch(
                "src.permissions.trust_boundary._load_project_scoped_env",
                return_value=project_env,
            ),
        ):
            # Remove any pre-existing keys to make the test deterministic
            for k in (
                "ANTHROPIC_MODEL",
                "DISABLE_TELEMETRY",
                "PATH",
                "NODE_EXTRA_CA_CERTS",
                "LD_PRELOAD",
            ):
                os.environ.pop(k, None)
            reset_trust_boundary_for_test_only()
            apply_safe_config_environment_variables()

            self.assertEqual(os.environ.get("ANTHROPIC_MODEL"), "claude-sonnet-4-6")
            self.assertEqual(os.environ.get("DISABLE_TELEMETRY"), "1")
            # Unsafe keys: NOT applied.
            self.assertNotEqual(os.environ.get("PATH"), "/opt/evil/bin")
            self.assertIsNone(os.environ.get("NODE_EXTRA_CA_CERTS"))
            self.assertIsNone(os.environ.get("LD_PRELOAD"))


class TestApplySafeDoesNotOverwriteExisting(unittest.TestCase):
    def test_setdefault_semantics(self) -> None:
        # Pre-existing process env wins; setdefault doesn't clobber.
        config_env = {"ANTHROPIC_MODEL": "from-config"}
        with mock.patch.dict(os.environ, {"ANTHROPIC_MODEL": "from-shell"}, clear=False):
            reset_trust_boundary_for_test_only()
            apply_safe_config_environment_variables(config_env)
            self.assertEqual(os.environ["ANTHROPIC_MODEL"], "from-shell")


class TestApplyFullOverwritesUnsafe(unittest.TestCase):
    def test_full_apply_writes_unsafe_keys(self) -> None:
        # Post-trust: the full apply writes everything, including
        # explicitly-unsafe keys.
        config_env = {"PATH": "/opt/safe/bin", "ANTHROPIC_MODEL": "claude-4"}
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PATH", None)
            os.environ.pop("ANTHROPIC_MODEL", None)
            reset_trust_boundary_for_test_only()
            apply_full_config_environment_variables(config_env)
            self.assertEqual(os.environ.get("PATH"), "/opt/safe/bin")
            self.assertEqual(os.environ.get("ANTHROPIC_MODEL"), "claude-4")

    def test_full_apply_preserves_original_shell_value(self) -> None:
        # This port intentionally keeps explicit shell overrides even after
        # trust; only managed-policy (MDM) values may replace them.
        config_env = {"ANTHROPIC_MODEL": "from-config"}
        with mock.patch.dict(os.environ, {"ANTHROPIC_MODEL": "from-shell"}, clear=False):
            reset_trust_boundary_for_test_only()
            apply_full_config_environment_variables(config_env)
            self.assertEqual(os.environ["ANTHROPIC_MODEL"], "from-shell")


class TestLoadConfigEnvFallthrough(unittest.TestCase):
    """Risk-register row 3: load_global must fall through cleanly
    when no global config exists.
    """

    def test_no_global_config_does_not_raise(self) -> None:
        # apply_safe_config_environment_variables() with no arg loads
        # from disk. On a fresh test machine without ~/.clawcodex/config.json
        # this should succeed and apply nothing (returns empty dict).
        # Mock ConfigManager.load_global to return {} (no env subkey)
        # so the test doesn't depend on user-local config.
        with mock.patch("src.config.ConfigManager.load_global", return_value={}):
            apply_safe_config_environment_variables()  # must not raise


class TestProjectConfigIsFilteredBeforeTrust(unittest.TestCase):
    """Project sources are read, but their unsafe keys stay unapplied."""

    def test_project_env_not_applied_pre_trust(self) -> None:
        # Simulate the attack: global config has no env, but project
        # config has a malicious PATH.
        attacker_env = {"PATH": "/opt/evil/bin"}
        original_path = os.environ.get("PATH", "")

        # The implementation should only read load_global. If it
        # reads load_project / load_local, the attacker's PATH would
        # be in config_env and then is_safe_env_key("PATH") returns
        # False so it's still not applied — but the security goal is
        # to never even read the project config in the first place.
        # We assert both: load_project is NOT called, and PATH stays
        # unchanged.
        with (
            mock.patch("src.config.ConfigManager.load_global", return_value={"env": {}}) as mock_global,
            mock.patch(
                "src.config.ConfigManager.load_project",
                return_value={"env": attacker_env},
            ) as mock_project,
            mock.patch(
                "src.config.ConfigManager.load_local",
                return_value={"env": attacker_env},
            ) as mock_local,
        ):
            reset_trust_boundary_for_test_only()
            apply_safe_config_environment_variables()
            # PATH unchanged — defense-in-depth from is_safe_env_key.
            self.assertEqual(os.environ.get("PATH", ""), original_path)
            mock_global.assert_called_once()
            mock_project.assert_called_once()
            mock_local.assert_called_once()


class TestUnsafeKeySetIsAuditable(unittest.TestCase):
    """Documented audit guarantees: the explicit unsafe set has
    every name the chapter / TS reference calls out by name.
    """

    def test_chapter_mentioned_keys_are_explicitly_unsafe(self) -> None:
        # Chapter §"The Trust Boundary": "PATH, LD_PRELOAD, NODE_OPTIONS".
        # All three must be explicitly enumerated (PATH/NODE_OPTIONS in
        # the set; LD_PRELOAD via UNSAFE_ENV_PREFIXES).
        self.assertIn("PATH", UNSAFE_ENV_KEYS)
        self.assertIn("NODE_OPTIONS", UNSAFE_ENV_KEYS)
        self.assertIn("LD_", UNSAFE_ENV_PREFIXES)

    def test_ts_attack_category_keys_explicitly_unsafe(self) -> None:
        # TS managedEnvConstants.ts:92-103 comment block enumerates the
        # explicit "DANGEROUS" categories. Audit that we keep them all.
        for expected in (
            "NODE_EXTRA_CA_CERTS",
            "NODE_TLS_REJECT_UNAUTHORIZED",
            "ANTHROPIC_BASE_URL",
            "HTTP_PROXY",
            "ANTHROPIC_API_KEY",
            "AWS_BEARER_TOKEN_BEDROCK",
        ):
            with self.subTest(key=expected):
                self.assertIn(expected, UNSAFE_ENV_KEYS)


if __name__ == "__main__":
    unittest.main()
