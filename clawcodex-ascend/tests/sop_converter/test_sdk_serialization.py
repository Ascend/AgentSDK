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

"""Tests for SDK wrapper result serialization."""

from __future__ import annotations

import dataclasses
import json
import unittest

from extensions.sop_converter.core.sdk_serialization import (
    WRAPPER_COERCION_HELPERS,
    WRAPPER_SERIALIZATION_HELPERS,
    coerce_mapping_value,
    coerce_sdk_type,
    dumps_sdk_result,
    to_jsonable,
)


@dataclasses.dataclass
class _DemoConfig:
    name: str
    count: int = 0


class TestSdkSerialization(unittest.TestCase):
    def test_dataclass_to_jsonable(self) -> None:
        payload = to_jsonable(_DemoConfig(name="verify-bot", count=2))
        self.assertEqual(payload, {"name": "verify-bot", "count": 2})

    def test_dumps_sdk_result_is_valid_json(self) -> None:
        raw = dumps_sdk_result(_DemoConfig(name="x"))
        parsed = json.loads(raw)
        self.assertEqual(parsed["name"], "x")

    def test_to_jsonable_nested_function_does_not_recurse(self) -> None:
        def factory(config):
            def loader():
                return config

            return loader

        payload = to_jsonable(factory({}))
        self.assertIsInstance(payload, str)
        self.assertIn("loader", payload)

        try:
            from pydantic import BaseModel
        except ImportError:
            self.skipTest("pydantic not installed")

        class Card(BaseModel):
            id: str

        payload = to_jsonable(Card(id="verify-bot"))
        self.assertEqual(payload, {"id": "verify-bot"})

    def test_dumps_sdk_result_redacts_sensitive_fields(self) -> None:
        raw = dumps_sdk_result({"name": "verify-bot", "api_key": "sk-secret"})
        parsed = json.loads(raw)
        self.assertEqual(parsed["name"], "verify-bot")
        self.assertEqual(parsed["api_key"], "<redacted>")

    def test_coerce_mapping_value_rejects_invalid_json(self) -> None:
        with self.assertRaises(TypeError):
            coerce_mapping_value('{"name": ')

    def test_coerce_sdk_type_sequence_rejects_non_iterable(self) -> None:
        with self.assertRaises(TypeError):
            coerce_sdk_type(list[str], 0)

    def test_wrapper_serialization_helpers_are_self_contained(self) -> None:
        namespace: dict[str, object] = {}
        # The helpers are rendered from annotated module-level functions; the
        # future-import flag keeps the annotations lazy when exec'd standalone.
        prelude = "from __future__ import annotations\n"
        exec(prelude + WRAPPER_SERIALIZATION_HELPERS, namespace)  # noqa: S102
        exec(prelude + WRAPPER_COERCION_HELPERS, namespace)  # noqa: S102
        self.assertEqual(
            namespace["_to_jsonable"](_DemoConfig(name="verify-bot", count=2)),
            {"name": "verify-bot", "count": 2},
        )
        self.assertEqual(namespace["_normalize_mapping_inputs"]('{"a": 1}'), {"a": 1})
        self.assertEqual(namespace["_coerce_mapping_value"]('{"a": 1}'), {"a": 1})

    def test_inline_resolve_env_references_shares_module_impl(self) -> None:
        # The inlined coercion helper is an alias of the module-level
        # ``resolve_env_references``, so it supports the same ``environ``
        # override parameter (no hardcoded ``os.environ`` copy).
        namespace: dict[str, object] = {}
        prelude = "from __future__ import annotations\n"
        exec(prelude + WRAPPER_SERIALIZATION_HELPERS, namespace)  # noqa: S102
        exec(prelude + WRAPPER_COERCION_HELPERS, namespace)  # noqa: S102
        inline = namespace["_resolve_env_references"]
        self.assertIs(inline, namespace["resolve_env_references"])
        self.assertEqual(inline("env:FAKE_VAR", environ={"FAKE_VAR": "value"}), "value")


if __name__ == "__main__":
    unittest.main()
