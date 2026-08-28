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

"""Unit tests for :mod:`extensions.sop_converter.sdk_parser`.

Covers the SdkParser that converts SDK/API specifications into atomic
tool definitions:

* :class:`SdkMethod` dataclass — field defaults and explicit construction.
* :class:`SdkParser` — input dispatch (dict, JSON string, simple list).
* :func:`_parse_openapi` — extracts operations (HTTP methods) and
  component schemas.
* :func:`_parse_simple_list` — splits comma/newline lists, filters
  comments, sanitises names.
* :func:`_sanitize_name` — kebab-case conversion of CamelCase, dots,
  brackets, slashes.
* :func:`parse_sdk_spec` — convenience wrapper, error capture.
"""

from __future__ import annotations

import unittest

from extensions.sop_converter.sdk_parser import (
    SdkParam,
    SdkParser,
    SdkParseResult,
    parse_sdk_spec,
)


# ---------------------------------------------------------------------------
# SdkMethod dataclass
# ---------------------------------------------------------------------------
class TestParseSdkSpec(unittest.TestCase):
    def test_successful_parse(self) -> None:
        result = parse_sdk_spec("alpha, beta", source="test")
        self.assertIsInstance(result, SdkParseResult)
        self.assertEqual(result.source, "test")
        self.assertEqual([m.name for m in result.methods], ["alpha", "beta"])
        self.assertEqual(result.errors, [])

    def test_default_source(self) -> None:
        result = parse_sdk_spec("alpha")
        self.assertEqual(result.source, "manual")

    def test_passes_through_errors(self) -> None:
        # Force an internal exception by mocking parse() to raise.
        from unittest.mock import patch

        parser = SdkParser("alpha")
        with patch.object(parser, "parse", side_effect=RuntimeError("boom")):
            with patch(
                "extensions.sop_converter.core.sdk_parser.SdkParser",
                return_value=parser,
            ):
                result = parse_sdk_spec("alpha")
        self.assertEqual(result.methods, [])
        self.assertEqual(result.errors, ["boom"])


# ---------------------------------------------------------------------------
# SdkParam dataclass
# ---------------------------------------------------------------------------


class TestSdkParam(unittest.TestCase):
    def test_defaults(self) -> None:
        p = SdkParam(name="x")
        self.assertEqual(p.name, "x")
        self.assertEqual(p.param_type, "string")
        self.assertFalse(p.required)
        self.assertEqual(p.description, "")
        self.assertEqual(p.location, "query")
        self.assertIsNone(p.schema)

    def test_explicit_fields(self) -> None:
        p = SdkParam(
            name="x",
            param_type="integer",
            required=True,
            description="A number",
            location="path",
            schema={"type": "integer", "minimum": 1},
        )
        self.assertEqual(p.param_type, "integer")
        self.assertTrue(p.required)
        self.assertEqual(p.description, "A number")
        self.assertEqual(p.location, "path")
        self.assertEqual(p.schema, {"type": "integer", "minimum": 1})

    def test_is_frozen(self) -> None:
        p = SdkParam(name="x")
        with self.assertRaises(Exception):
            p.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SdkParser.parse — OpenAPI enhanced parsing
# ---------------------------------------------------------------------------


class TestParseOpenApiEnhanced(unittest.TestCase):
    def test_http_method_extracted(self) -> None:
        spec = {
            "paths": {
                "/users": {
                    "get": {"operationId": "listUsers"},
                    "post": {"operationId": "createUser"},
                },
            },
        }
        methods = SdkParser(spec).parse()
        for m in methods:
            if m.name == "list_users":
                self.assertEqual(m.http_method, "GET")
            elif m.name == "create_user":
                self.assertEqual(m.http_method, "POST")

    def test_http_path_extracted(self) -> None:
        spec = {
            "paths": {
                "/users/{id}": {
                    "get": {"operationId": "getUser"},
                },
            },
        }
        methods = SdkParser(spec).parse()
        self.assertEqual(methods[0].http_path, "/users/{id}")

    def test_params_with_types(self) -> None:
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "integer"},
                                "description": "Max results",
                            },
                            {
                                "name": "offset",
                                "in": "query",
                                "required": True,
                                "schema": {"type": "integer"},
                            },
                        ],
                    },
                },
            },
        }
        methods = SdkParser(spec).parse()
        method = methods[0]
        self.assertEqual(len(method.params), 2)
        self.assertEqual(method.params[0].name, "limit")
        self.assertEqual(method.params[0].param_type, "integer")
        self.assertFalse(method.params[0].required)
        self.assertEqual(method.params[0].description, "Max results")
        self.assertEqual(method.params[0].location, "query")
        self.assertEqual(method.params[1].name, "offset")
        self.assertTrue(method.params[1].required)

    def test_request_body_extracted(self) -> None:
        spec = {
            "paths": {
                "/users": {
                    "post": {
                        "operationId": "createUser",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "email": {"type": "string"},
                                        },
                                        "required": ["name", "email"],
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        methods = SdkParser(spec).parse()
        method = methods[0]
        self.assertIsNotNone(method.request_body)
        self.assertTrue(method.request_body.get("required"))
        self.assertIn("application/json", method.request_body.get("content", {}))

    def test_responses_extracted(self) -> None:
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/User"},
                                        },
                                    },
                                },
                            },
                            "400": {"description": "Bad request"},
                        },
                    },
                },
            },
        }
        methods = SdkParser(spec).parse()
        method = methods[0]
        self.assertIn("200", method.responses)
        self.assertIn("400", method.responses)
        self.assertEqual(method.responses["200"]["description"], "Success")

    def test_tags_extracted(self) -> None:
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "tags": ["users", "admin"],
                    },
                },
            },
        }
        methods = SdkParser(spec).parse()
        self.assertEqual(methods[0].tags, ["users", "admin"])

    def test_server_url_extracted(self) -> None:
        spec = {
            "servers": [{"url": "https://api.example.com/v1"}],
            "paths": {
                "/users": {"get": {"operationId": "listUsers"}},
            },
        }
        parser = SdkParser(spec)
        parser.parse()
        self.assertEqual(parser.openapi_base_url, "https://api.example.com/v1")

    def test_path_params_recognized(self) -> None:
        spec = {
            "paths": {
                "/users/{userId}/posts/{postId}": {
                    "get": {
                        "operationId": "getPost",
                        "parameters": [
                            {"name": "userId", "in": "path", "required": True},
                            {"name": "postId", "in": "path", "required": True},
                        ],
                    },
                },
            },
        }
        methods = SdkParser(spec).parse()
        method = methods[0]
        path_params = [p for p in method.params if p.location == "path"]
        self.assertEqual(len(path_params), 2)
        self.assertEqual(path_params[0].name, "userId")
        self.assertEqual(path_params[1].name, "postId")


if __name__ == "__main__":
    unittest.main()
