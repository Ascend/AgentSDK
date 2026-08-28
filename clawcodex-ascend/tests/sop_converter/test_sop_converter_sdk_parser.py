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

"""Unit tests for :mod:`extensions.sop_converter.core.sdk_parser`.

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

import socket
import tempfile
import unittest
import urllib.error
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from extensions.sop_converter.core.sdk_parser import (
    SdkMethod,
    SdkParser,
    parse_sdk_spec,
)


def _public_addrinfo(host, port, *args, **kwargs):
    """Return a public IPv4 address, decoupling URL safety checks from DNS."""
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port))]


class _FakeResponse:
    """Context-manager response stub returning a minimal OpenAPI JSON."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def read(self) -> bytes:
        return (
            b'{"openapi": "3.0.0", "info": {"title": "T", "version": "1"},'
            b' "paths": {"/hello": {"get": {"operationId": "getHello", "summary": "Hello"}}}}'
        )


# ---------------------------------------------------------------------------
# SdkMethod dataclass
# ---------------------------------------------------------------------------


class TestSdkMethod(unittest.TestCase):
    def test_defaults(self) -> None:
        m = SdkMethod(name="x", description="d")
        self.assertEqual(m.name, "x")
        self.assertEqual(m.description, "d")
        self.assertEqual(m.parameters, [])
        self.assertEqual(m.required_params, [])
        self.assertIsNone(m.return_type)
        self.assertIsNone(m.original_class)

    def test_explicit_fields(self) -> None:
        m = SdkMethod(
            name="x",
            description="d",
            parameters=["a", "b"],
            required_params=["a"],
            return_type="dict",
            original_class="Foo",
        )
        self.assertEqual(m.parameters, ["a", "b"])
        self.assertEqual(m.required_params, ["a"])
        self.assertEqual(m.return_type, "dict")
        self.assertEqual(m.original_class, "Foo")

    def test_is_frozen(self) -> None:
        # Field assignment should raise because the dataclass is frozen.
        m = SdkMethod(name="x", description="d")
        with self.assertRaises(FrozenInstanceError):
            m.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------


class TestSanitizeName(unittest.TestCase):
    def test_already_kebab(self) -> None:
        self.assertEqual(SdkParser._sanitize_name("docker-build"), "docker-build")

    def test_camelcase_split(self) -> None:
        # CamelCase → snake_case → kebab-case.
        self.assertEqual(SdkParser._sanitize_name("DockerBuild"), "docker_build")

    def test_slash_replaced(self) -> None:
        # Slashes become underscores then collapse.
        self.assertEqual(
            SdkParser._sanitize_name("foo/bar"),
            "foo_bar",
        )

    def test_brackets_replaced(self) -> None:
        # Bracket chars are scrubbed → underscores → collapsed → lower.
        # "a[b]" → "a_b_" → "a_b" → "a_b".
        self.assertEqual(SdkParser._sanitize_name("a[b]"), "a_b")

    def test_underscores_collapsed(self) -> None:
        # Multiple consecutive underscores collapse to one.
        self.assertEqual(
            SdkParser._sanitize_name("a__b___c"),
            "a_b_c",
        )

    def test_empty_name_falls_back(self) -> None:
        # After stripping all non-word chars, an empty name → "sdk_method".
        self.assertEqual(SdkParser._sanitize_name("///"), "sdk_method")

    def test_lowercases_result(self) -> None:
        self.assertEqual(SdkParser._sanitize_name("FOO"), "f_o_o")


# ---------------------------------------------------------------------------
# SdkParser.parse — simple list path
# ---------------------------------------------------------------------------


class TestParseSimpleList(unittest.TestCase):
    def test_comma_separated(self) -> None:
        methods = SdkParser("docker_build, docker_tag, docker_push").parse()
        names = [m.name for m in methods]
        self.assertIn("docker_build", names)
        self.assertIn("docker_tag", names)
        self.assertIn("docker_push", names)

    def test_newline_separated(self) -> None:
        methods = SdkParser("foo\nbar\nbaz").parse()
        self.assertEqual(
            [m.name for m in methods],
            ["foo", "bar", "baz"],
        )

    def test_mixed_separators(self) -> None:
        methods = SdkParser("foo,bar\nbaz").parse()
        self.assertEqual(
            [m.name for m in methods],
            ["foo", "bar", "baz"],
        )

    def test_empty_input(self) -> None:
        self.assertEqual(SdkParser("").parse(), [])

    def test_whitespace_input(self) -> None:
        self.assertEqual(SdkParser("   \n  \n  ").parse(), [])

    def test_comment_lines_skipped(self) -> None:
        methods = SdkParser("# this is a comment\nreal_method").parse()
        self.assertEqual([m.name for m in methods], ["real_method"])

    def test_each_method_has_description(self) -> None:
        methods = SdkParser("alpha, beta").parse()
        for m in methods:
            self.assertIn("SDK method:", m.description)

    def test_each_method_has_empty_params(self) -> None:
        methods = SdkParser("alpha").parse()
        self.assertEqual(methods[0].parameters, [])
        self.assertEqual(methods[0].required_params, [])

    def test_dedupes_via_sanitize(self) -> None:
        # "foo" and "Foo" sanitise identically. "FOO" sanitises to
        # "f_o_o" because each uppercase letter is underscore-prefixed.
        methods = SdkParser("foo, Foo").parse()
        self.assertEqual([m.name for m in methods], ["foo", "foo"])

    def test_uppercase_letters_each_prefixed(self) -> None:
        # "FOO" → "F_O_O" because CamelCase splitting puts an underscore
        # before every uppercase letter.
        self.assertEqual(SdkParser._sanitize_name("FOO"), "f_o_o")


# ---------------------------------------------------------------------------
# SdkParser.parse — OpenAPI dict path
# ---------------------------------------------------------------------------


class TestSchemaResolution(unittest.TestCase):
    def test_recursive_schema_reference_is_short_circuited(self) -> None:
        ref = "#/components/schemas/Node"
        schemas = {
            "Node": {
                "type": "object",
                "properties": {
                    "children": {
                        "type": "array",
                        "items": {"$ref": ref},
                    },
                },
            },
        }

        resolved = SdkParser._resolve_schema({"$ref": ref}, schemas)

        self.assertEqual(resolved["type"], "object")
        self.assertEqual(resolved["properties"]["children"]["items"], {"$ref": ref})

    def test_recursive_schema_type_reference_is_short_circuited(self) -> None:
        schemas = {
            "First": {"$ref": "#/components/schemas/Second"},
            "Second": {"$ref": "#/components/schemas/First"},
        }

        self.assertEqual(
            SdkParser._resolve_schema_type(
                {"$ref": "#/components/schemas/First"},
                schemas,
            ),
            "object",
        )

    def test_schema_resolution_does_not_mutate_input(self) -> None:
        ref = "#/components/schemas/Child"
        schema = {
            "type": "object",
            "properties": {"child": {"$ref": ref}},
        }
        schemas = {"Child": {"type": "string"}}
        original = deepcopy(schema)

        SdkParser._resolve_schema(schema, schemas)

        self.assertEqual(schema, original)


class TestParseOpenApiDict(unittest.TestCase):
    def test_basic_operations_extracted(self) -> None:
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "summary": "List all users",
                        "parameters": [
                            {"name": "limit", "required": False},
                            {"name": "offset", "required": True},
                        ],
                    },
                    "post": {
                        "operationId": "createUser",
                        "summary": "Create a user",
                    },
                },
            },
        }
        methods = SdkParser(spec).parse()
        names = [m.name for m in methods]
        self.assertIn("list_users", names)
        self.assertIn("create_user", names)

    def test_required_params_collected(self) -> None:
        spec = {
            "paths": {
                "/x": {
                    "get": {
                        "operationId": "x",
                        "parameters": [
                            {"name": "a", "required": True},
                            {"name": "b", "required": False},
                            {"name": "c", "required": True},
                        ],
                    },
                },
            },
        }
        methods = SdkParser(spec).parse()
        self.assertEqual(methods[0].required_params, ["a", "c"])
        self.assertEqual(methods[0].parameters, ["a", "b", "c"])

    def test_description_truncated(self) -> None:
        long_desc = "x" * 500
        spec = {
            "paths": {
                "/x": {
                    "get": {
                        "operationId": "long",
                        "description": long_desc,
                    },
                },
            },
        }
        methods = SdkParser(spec).parse()
        self.assertEqual(len(methods[0].description), 200)

    def test_summary_preferred_over_description(self) -> None:
        spec = {
            "paths": {
                "/x": {
                    "get": {
                        "operationId": "x",
                        "summary": "Short summary",
                        "description": "Longer description.",
                    },
                },
            },
        }
        methods = SdkParser(spec).parse()
        self.assertEqual(methods[0].description, "Short summary")

    def test_non_http_methods_ignored(self) -> None:
        # "summary" and "description" keys inside path dicts are not
        # HTTP methods and should be skipped.
        spec = {
            "paths": {
                "/x": {
                    "summary": "path summary",  # not a real method
                    "get": {"operationId": "real"},
                },
            },
        }
        methods = SdkParser(spec).parse()
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0].name, "real")

    def test_components_schemas_extracted(self) -> None:
        spec = {
            "components": {
                "schemas": {
                    "User": {
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                        "required": ["id"],
                    },
                },
            },
        }
        methods = SdkParser(spec).parse()
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0].name, "user")
        self.assertEqual(methods[0].parameters, ["id", "name"])
        self.assertEqual(methods[0].required_params, ["id"])
        self.assertEqual(methods[0].original_class, "User")

    def test_default_operation_id(self) -> None:
        # No operationId → use "{method}_{path}" as fallback.
        spec = {
            "paths": {
                "/items": {
                    "get": {"summary": "Get items"},
                },
            },
        }
        methods = SdkParser(spec).parse()
        # Sanitised name: "get_/items" → "/" becomes "_" → "get__items"
        # → underscores collapsed to single → "get_items" → trailing
        # underscores stripped → "get_items".
        self.assertIn("get_items", [m.name for m in methods])

    def test_all_four_http_methods(self) -> None:
        spec = {
            "paths": {
                "/r": {
                    "get": {"operationId": "g"},
                    "post": {"operationId": "p"},
                    "put": {"operationId": "u"},
                    "delete": {"operationId": "d"},
                    "patch": {"operationId": "pa"},
                },
            },
        }
        methods = SdkParser(spec).parse()
        names = sorted(m.name for m in methods)
        self.assertEqual(names, ["d", "g", "p", "pa", "u"])

    def test_empty_spec_returns_empty(self) -> None:
        self.assertEqual(SdkParser({}).parse(), [])

    def test_methods_cached(self) -> None:
        # Second parse() call returns the same list (idempotent).
        spec = {"paths": {"/x": {"get": {"operationId": "x"}}}}
        parser = SdkParser(spec)
        first = parser.parse()
        second = parser.parse()
        self.assertIs(first, second)


# ---------------------------------------------------------------------------
# SdkParser.parse — JSON string path
# ---------------------------------------------------------------------------


class TestParseJsonString(unittest.TestCase):
    def test_json_object_string(self) -> None:
        # Strings starting with `{` are parsed as JSON.
        spec = '{"paths": {"/x": {"get": {"operationId": "getX"}}}}'
        methods = SdkParser(spec).parse()
        self.assertEqual([m.name for m in methods], ["get_x"])

    def test_json_with_https_prefix_falls_back_to_list(self) -> None:
        # Starts with "https://" → tried as JSON first, fails → simple
        # list path.
        spec = "https://example.com/api/v1, foo, bar"
        with (
            patch("socket.getaddrinfo", side_effect=_public_addrinfo),
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError("offline"),
            ) as urlopen,
        ):
            methods = SdkParser(spec).parse()
        urlopen.assert_called_once_with(spec, timeout=30)
        # The first segment is the URL (treated as one name), plus foo,
        # bar.
        names = [m.name for m in methods]
        self.assertIn("foo", names)
        self.assertIn("bar", names)

    def test_url_unexpected_errors_are_not_silently_downgraded(self) -> None:
        with (
            patch("socket.getaddrinfo", side_effect=_public_addrinfo),
            patch(
                "urllib.request.urlopen",
                side_effect=RuntimeError("unexpected parser failure"),
            ),
            self.assertRaises(RuntimeError),
        ):
            SdkParser("https://example.com/spec").parse()

    def test_private_loopback_url_is_blocked(self) -> None:
        # SSRF guard: loopback targets must never be fetched.
        with patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("urlopen must not be called"),
        ) as urlopen:
            methods = SdkParser("http://127.0.0.1:8500/openapi.json").parse()
        urlopen.assert_not_called()
        self.assertIsInstance(methods, list)

    def test_link_local_url_is_blocked(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("urlopen must not be called"),
        ) as urlopen:
            methods = SdkParser("http://169.254.169.254/latest/meta-data/").parse()
        urlopen.assert_not_called()
        self.assertIsInstance(methods, list)

    def test_file_scheme_url_is_not_fetched(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("urlopen must not be called"),
        ) as urlopen:
            methods = SdkParser("file:///etc/passwd").parse()
        urlopen.assert_not_called()
        self.assertIsInstance(methods, list)

    def test_public_url_is_fetched(self) -> None:
        with (
            patch("socket.getaddrinfo", side_effect=_public_addrinfo),
            patch(
                "urllib.request.urlopen",
                return_value=_FakeResponse(),
            ) as urlopen,
        ):
            methods = SdkParser("http://93.184.216.34/openapi.json").parse()
        urlopen.assert_called_once()
        self.assertTrue(any(m.name == "get_hello" for m in methods))

    def test_json_file_with_non_dict_content_falls_back_to_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "spec.json"
            file_path.write_text("[1, 2, 3]", encoding="utf-8")

            methods = SdkParser(str(file_path)).parse()

        self.assertEqual(len(methods), 1)

    def test_invalid_json_falls_back_to_simple_list(self) -> None:
        # Doesn't start with `{`, so goes directly to simple list.
        spec = "alpha, beta"
        methods = SdkParser(spec).parse()
        self.assertEqual([m.name for m in methods], ["alpha", "beta"])


# ---------------------------------------------------------------------------
# SdkParser.raw property
# ---------------------------------------------------------------------------


class TestSdkParserRaw(unittest.TestCase):
    def test_string_input_stored_stripped(self) -> None:
        parser = SdkParser("  alpha, beta  ")
        self.assertEqual(parser.raw, "alpha, beta")

    def test_dict_input_stored_as_is(self) -> None:
        d = {"paths": {}}
        parser = SdkParser(d)
        self.assertIs(parser.raw, d)

    def test_source_attribute(self) -> None:
        parser = SdkParser("x", source="custom")
        self.assertEqual(parser._source, "custom")


# ---------------------------------------------------------------------------
# parse_sdk_spec convenience function
# ---------------------------------------------------------------------------


class TestParseSdkSpec(unittest.TestCase):
    def test_successful_parse(self) -> None:
        result = parse_sdk_spec("alpha, beta", source="test")

        self.assertEqual(result.source, "test")
        self.assertEqual([method.name for method in result.methods], ["alpha", "beta"])
        self.assertEqual(result.errors, [])
