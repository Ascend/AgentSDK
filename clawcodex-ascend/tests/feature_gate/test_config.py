#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for ``clawcodex_ext.feature_gate.config``."""

import json


from clawcodex_ext.feature_gate.config import ConfigStore


class TestConfigStore:
    """Test JSON config persistence."""

    def test_get_returns_none_when_empty(self):
        store = ConfigStore()
        assert store.get("any_key") is None

    def test_save_and_get(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path)
        store.save({"feat_a": True, "feat_b": False})
        assert store.get("feat_a") is True
        assert store.get("feat_b") is False

    def test_save_creates_directory(self, tmp_path):
        subdir = tmp_path / "nested" / "deep"
        store = ConfigStore(config_dir=subdir)
        store.save({"x": True})
        assert subdir.exists()
        assert (subdir / "features.json").exists()

    def test_save_overwrites_existing(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path)
        store.save({"key": True})
        store.save({"key": False, "new_key": True})
        assert store.get("key") is False
        assert store.get("new_key") is True

    def test_reload_clears_cache(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path)
        store.save({"a": True})
        assert store.get("a") is True

        # Modify the file on disk
        config_file = store.config_file
        with open(config_file, "w") as f:
            json.dump({"a": False, "b": True}, f)

        store.reload()
        assert store.get("a") is False
        assert store.get("b") is True

    def test_load_from_existing_file(self, tmp_path):
        config_file = tmp_path / "features.json"
        with open(config_file, "w") as f:
            json.dump({"persisted": True, "also_persisted": False}, f)

        store = ConfigStore(config_dir=tmp_path)
        assert store.get("persisted") is True
        assert store.get("also_persisted") is False

    def test_load_invalid_json(self, tmp_path):
        config_file = tmp_path / "features.json"
        config_file.write_text("not valid json {{{")

        store = ConfigStore(config_dir=tmp_path)
        # Should not raise; returns None for all keys
        assert store.get("anything") is None

    def test_config_file_property(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path)
        assert store.config_file == tmp_path / "features.json"

    def test_config_dir_property(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path)
        assert store.config_dir == tmp_path

    def test_save_to_nonexistent_dir(self, tmp_path):
        """Saving should create the directory if it doesn't exist."""
        new_dir = tmp_path / "brand_new"
        store = ConfigStore(config_dir=new_dir)
        store.save({"test": True})
        assert (new_dir / "features.json").exists()

    def test_best_effort_io_error(self, tmp_path):
        """IO errors should be logged but not raised."""
        # Create a file at the expected path that is a directory,
        # causing the open() to fail.
        blocker = tmp_path / "features.json"
        blocker.mkdir()

        store = ConfigStore(config_dir=tmp_path)
        # Should not raise
        store.save({"should_fail": True})
