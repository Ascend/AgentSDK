#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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

"""Unit tests for aura.memory.memory_summary.MemorySummary."""

from types import SimpleNamespace

import pytest

from aura.memory.memory_summary import MemorySummary


# ---------------------------------------------------------------------------
# __init__ / default config
# ---------------------------------------------------------------------------

class TestInitDefaults:
    def test_default_config_values(self):
        mem = MemorySummary()
        assert mem.config.use_summary is False
        assert mem.config.max_prompt_length == 8192
        assert mem.config.chat_model_name == "qwen3_4b"
        assert mem.config.train_model_tokenizer_path is None

    def test_default_messages_empty(self):
        mem = MemorySummary()
        assert mem.messages == []

    def test_tokenizer_stored(self):
        tok = object()
        mem = MemorySummary(tokenizer=tok)
        assert mem.tokenizer is tok

    def test_none_config_treated_as_empty(self):
        mem = MemorySummary(config=None)
        assert mem.config.use_summary is False


# ---------------------------------------------------------------------------
# update_configs
# ---------------------------------------------------------------------------

class TestUpdateConfigs:
    def test_overrides_existing_keys(self):
        mem = MemorySummary()
        mem.update_configs({"use_summary": True, "max_prompt_length": 4096})
        assert mem.config.use_summary is True
        assert mem.config.max_prompt_length == 4096

    def test_ignores_unknown_keys(self):
        mem = MemorySummary()
        mem.update_configs({"custom_key": "abc"})
        assert not hasattr(mem.config, "custom_key")

    def test_empty_dict_noop(self):
        mem = MemorySummary()
        before = mem.config.max_prompt_length
        mem.update_configs({})
        assert mem.config.max_prompt_length == before

    def test_init_config_param_applied(self):
        mem = MemorySummary(config={"chat_model_name": "qwen3_8b", "use_summary": True})
        assert mem.config.chat_model_name == "qwen3_8b"
        assert mem.config.use_summary is True


# ---------------------------------------------------------------------------
# clear_memory
# ---------------------------------------------------------------------------

class TestClearMemory:
    def test_empty_content_clears_messages(self):
        mem = MemorySummary()
        mem.add_message({"role": "user", "content": "hi"})
        assert mem.messages != []
        mem.clear_memory()
        assert mem.messages == []

    def test_with_content_writes_single_message(self):
        mem = MemorySummary()
        mem.clear_memory(role="system", content="reset prompt")
        assert mem.messages == [{"role": "system", "content": "reset prompt"}]

    def test_custom_role(self):
        mem = MemorySummary()
        mem.clear_memory(role="user", content="hello")
        assert mem.messages[0]["role"] == "user"

    def test_clear_replaces_existing_messages(self):
        mem = MemorySummary()
        mem.add_message({"role": "user", "content": "old"})
        mem.clear_memory(content="new")
        assert len(mem.messages) == 1
        assert mem.messages[0]["content"] == "new"


# ---------------------------------------------------------------------------
# add_message
# ---------------------------------------------------------------------------

class TestAddMessage:
    def test_single_dict(self):
        mem = MemorySummary()
        mem.add_message({"role": "user", "content": "a"})
        assert mem.messages == [{"role": "user", "content": "a"}]

    def test_list_of_dicts(self):
        mem = MemorySummary()
        mem.add_message([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
        assert len(mem.messages) == 2

    def test_ignores_none(self):
        mem = MemorySummary()
        mem.add_message(None)
        assert mem.messages == []

    def test_ignores_empty(self):
        mem = MemorySummary()
        mem.add_message("")
        assert mem.messages == []

    def test_filters_non_dict_items_in_list(self):
        mem = MemorySummary()
        mem.add_message([{"role": "user", "content": "a"}, None, "str", 123, {"role": "assistant", "content": "b"}])
        assert len(mem.messages) == 2
        assert mem.messages[0]["role"] == "user"
        assert mem.messages[1]["role"] == "assistant"

    def test_insert_id_appends_when_none(self):
        mem = MemorySummary()
        mem.add_message({"role": "user", "content": "first"})
        mem.add_message({"role": "user", "content": "second"})
        assert mem.messages[1]["content"] == "second"

    def test_insert_id_inserts_at_position(self):
        mem = MemorySummary()
        mem.add_message([{"role": "user", "content": "a"}, {"role": "user", "content": "c"}])
        mem.add_message({"role": "user", "content": "b"}, insert_id=1)
        assert [m["content"] for m in mem.messages] == ["a", "b", "c"]

    def test_insert_id_with_list(self):
        mem = MemorySummary()
        mem.add_message([{"role": "user", "content": "a"}, {"role": "user", "content": "d"}])
        mem.add_message(
            [{"role": "user", "content": "b"}, {"role": "user", "content": "c"}],
            insert_id=1,
        )
        assert [m["content"] for m in mem.messages] == ["a", "b", "c", "d"]

    def test_add_message_copies_dict(self):
        mem = MemorySummary()
        original = {"role": "user", "content": "a"}
        mem.add_message(original)
        original["content"] = "mutated"
        assert mem.messages[0]["content"] == "a"

    def test_add_message_applies_metadata(self):
        mem = MemorySummary()
        mem.add_message({"role": "user", "content": "a"}, metadata={"step": 1})
        assert mem.messages[0]["metadata"] == {"step": 1}


# ---------------------------------------------------------------------------
# get_prompt_messages / get_messages
# ---------------------------------------------------------------------------

class TestGetters:
    def test_get_prompt_messages_returns_shallow_copy(self):
        mem = MemorySummary()
        mem.add_message({"role": "user", "content": "a"})
        snapshot = mem.get_prompt_messages()
        snapshot.append({"role": "user", "content": "b"})
        assert len(mem.messages) == 1

    def test_get_prompt_messages_type(self):
        mem = MemorySummary()
        result = mem.get_prompt_messages()
        assert isinstance(result, list)

    def test_get_messages_returns_shallow_copy(self):
        mem = MemorySummary()
        mem.add_message({"role": "user", "content": "a"})
        snapshot = mem.get_messages()
        snapshot.append({"role": "user", "content": "b"})
        assert len(mem.messages) == 1

    def test_get_prompt_messages_reflects_changes(self):
        mem = MemorySummary()
        mem.add_message({"role": "user", "content": "a"})
        first = mem.get_prompt_messages()
        mem.add_message({"role": "user", "content": "b"})
        second = mem.get_prompt_messages()
        assert len(first) == 1
        assert len(second) == 2


# ---------------------------------------------------------------------------
# config namespace type
# ---------------------------------------------------------------------------

class TestConfigNamespace:
    def test_config_is_simplenamespace(self):
        mem = MemorySummary()
        assert isinstance(mem.config, SimpleNamespace)

    def test_config_attrs_settable_after_init(self):
        mem = MemorySummary()
        mem.config.use_summary = True
        assert mem.config.use_summary is True
