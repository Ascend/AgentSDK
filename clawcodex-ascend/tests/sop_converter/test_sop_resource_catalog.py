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

"""Tests for the SOP ResourceCatalog."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extensions.sop_converter.core.agent_catalog import AgentCatalogEntry
from extensions.sop_converter.core.agent_catalog_resolver import HOME_ONLY_ENV, HOME_ROOT_ENV
from extensions.sop_converter.resource_catalog import (
    DUAL_WRITE_ENV,
    RESOURCE_CATALOG_AMBIGUOUS,
    RESOURCE_CATALOG_MISSING,
    RESOURCE_CATALOG_WRITE_FAILED,
    RESOURCE_VERSION_UNSUPPORTED,
    SCHEMA_VERSION,
    SESSION_ID_ENV,
    CatalogExecutionContext,
    ResourceCatalogError,
    ResourceCatalog,
    ResourceRecord,
    agent_entry_to_resource_record,
    context_from_env,
    get_agent_record,
    get_resource_record,
    plan_write_targets,
    resolve_record,
    resolve_resource_catalog_path,
    resource_error,
    write_record,
)


def _tmp_path() -> Path:
    return Path(tempfile.mkdtemp(prefix="resource_catalog_test_"))


def _make_record(*, resource_id: str = "agent-1") -> ResourceRecord:
    return ResourceRecord(
        resource_type="AgentConfig",
        resource_id=resource_id,
        bundle_id="core",
        source_tool="agentbuilder-build-agent",
        materializer={
            "kind": "python_class",
            "module": "fake_sdk.agent",
            "class_name": "DemoAgent",
        },
        invoker={"kind": "python_method", "method": "invoke", "input_param": "query"},
        payload={
            "kind": "inline",
            "dsl": {"name": "verify-bot", "api_key": "sk-secret"},
            "model": "gpt-4o",
        },
        sdk={"source_dir": "/tmp/sdk"},
    )


class TestResourceCatalogRoundTrip(unittest.TestCase):
    def test_save_load_and_lookup_round_trip(self) -> None:
        path = _tmp_path() / "resource-catalog.json"
        cat = ResourceCatalog()
        cat.upsert(_make_record())
        cat.save(path)

        raw = path.read_text(encoding="utf-8")
        self.assertNotIn("sk-secret", raw)
        self.assertIn("CLAWCODEX_CORE_API_KEY", raw)

        with patch.dict(os.environ, {"CLAWCODEX_CORE_API_KEY": "sk-restored"}):
            loaded = ResourceCatalog.load(path)
            record = loaded.get("agentconfig", "agent-1")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.payload["dsl"]["api_key"], "sk-restored")
        self.assertEqual(record.materializer["class_name"], "DemoAgent")
        self.assertEqual(record.invoker["input_param"], "query")

    def test_upsert_is_idempotent_and_preserves_created_at(self) -> None:
        cat = ResourceCatalog()
        first = cat.upsert(_make_record())
        second_record = _make_record()
        second_record.metadata = {"extra": "x"}
        second = cat.upsert(second_record)
        self.assertEqual(first.created_at, second.created_at)
        self.assertIsNotNone(cat.get("agentconfig", "agent-1"))
        self.assertEqual(cat.get("agentconfig", "agent-1").metadata["extra"], "x")  # type: ignore[union-attr]

    def test_missing_and_bad_files_degrade_to_empty(self) -> None:
        missing = _tmp_path() / "missing.json"
        self.assertEqual(ResourceCatalog.load(missing).records, {})
        bad = _tmp_path() / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        self.assertEqual(ResourceCatalog.load(bad).records, {})

    def test_unsupported_catalog_version_raises_resource_version_unsupported(self) -> None:
        future = _tmp_path() / "future.json"
        future.write_text(json.dumps({"version": 999, "records": {}}), encoding="utf-8")
        with self.assertRaises(ResourceCatalogError) as raised:
            ResourceCatalog.load(future)
        self.assertEqual(raised.exception.error_code, RESOURCE_VERSION_UNSUPPORTED)
        self.assertIn("999", str(raised.exception))
        self.assertIn(str(SCHEMA_VERSION), str(raised.exception))

    def test_get_agent_record_propagates_unsupported_catalog_version(self) -> None:
        root = _tmp_path()
        bundle = root / "bundle"
        bundle.mkdir()
        location = resolve_resource_catalog_path(bundle)
        location.path.parent.mkdir(parents=True, exist_ok=True)
        location.path.write_text(
            json.dumps({"version": 999, "records": {}}),
            encoding="utf-8",
        )
        with self.assertRaises(ResourceCatalogError) as raised:
            get_agent_record(
                agent_ref="verify-bot",
                catalog_context=CatalogExecutionContext(
                    bundle_path=bundle,
                    bundle_id="bundle",
                ),
            )
        self.assertEqual(raised.exception.error_code, RESOURCE_VERSION_UNSUPPORTED)


class TestResourceCatalogResolver(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {
            HOME_ROOT_ENV: os.environ.pop(HOME_ROOT_ENV, None),
            HOME_ONLY_ENV: os.environ.pop(HOME_ONLY_ENV, None),
        }

    def tearDown(self) -> None:
        for key, value in self._saved_env.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    def test_bundle_user_and_session_paths(self) -> None:
        root = _tmp_path()
        bundle = root / "bundle"
        bundle.mkdir()
        os.environ[HOME_ROOT_ENV] = str(root)

        bundle_loc = resolve_resource_catalog_path(bundle)
        self.assertEqual(bundle_loc.reason, "bundle-local")
        self.assertEqual(bundle_loc.path, bundle / ".clawcodex" / "resource-catalog.json")

        user_loc = resolve_resource_catalog_path(bundle, bundle_id="core", scope="user")
        self.assertEqual(user_loc.reason, "user-local")
        self.assertEqual(user_loc.path, root / "sop-resources" / "core" / "catalog.json")

        session_loc = resolve_resource_catalog_path(bundle, session_id="s1", scope="session")
        self.assertEqual(session_loc.reason, "session-local")
        self.assertEqual(session_loc.path, root / "sessions" / "s1" / "sop-resources.json")

    def test_home_only_forces_user_catalog(self) -> None:
        root = _tmp_path()
        bundle = root / "bundle"
        bundle.mkdir()
        os.environ[HOME_ROOT_ENV] = str(root)
        os.environ[HOME_ONLY_ENV] = "1"
        loc = resolve_resource_catalog_path(bundle, bundle_id="core")
        self.assertEqual(loc.reason, "home-forced")
        self.assertEqual(loc.path, root / "sop-resources" / "core" / "catalog.json")


class TestResourceCatalogAgentBridge(unittest.TestCase):
    def setUp(self) -> None:
        self._home = _tmp_path() / "clawcodex-home"
        self._home.mkdir()
        self._env_patch = patch.dict(os.environ, {HOME_ROOT_ENV: str(self._home)})
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()

    def test_get_agent_record_resolves_saved_agent_name(self) -> None:
        root = _tmp_path()
        bundle = root / "bundle"
        bundle.mkdir()
        location = resolve_resource_catalog_path(bundle)
        catalog = ResourceCatalog()
        catalog.upsert(_make_record(resource_id="agent-verify"))
        catalog.save(location.path)

        record = get_agent_record(agent_ref="verify-bot", bundle_path=bundle)

        self.assertEqual(record.resource_id, "agent-verify")
        self.assertEqual(record.payload["dsl"]["name"], "verify-bot")

    def test_get_agent_record_uses_catalog_execution_context(self) -> None:
        root = _tmp_path()
        bundle = root / "bundle"
        bundle.mkdir()
        location = resolve_resource_catalog_path(bundle)
        catalog = ResourceCatalog()
        catalog.upsert(_make_record(resource_id="agent-context"))
        catalog.save(location.path)

        record = get_agent_record(
            agent_id="agent-context",
            catalog_context=CatalogExecutionContext(
                bundle_path=bundle,
                bundle_id="bundle",
            ),
        )

        self.assertEqual(record.resource_id, "agent-context")

    def test_get_agent_record_rejects_ambiguous_saved_name(self) -> None:
        root = _tmp_path()
        bundle = root / "bundle"
        bundle.mkdir()
        location = resolve_resource_catalog_path(bundle)
        catalog = ResourceCatalog()
        catalog.upsert(_make_record(resource_id="agent-one"))
        catalog.upsert(_make_record(resource_id="agent-two"))
        catalog.save(location.path)

        with self.assertRaisesRegex(Exception, "multiple records") as raised:
            get_agent_record(agent_ref="verify-bot", bundle_path=bundle)
        self.assertEqual(getattr(raised.exception, "error_code", ""), RESOURCE_CATALOG_AMBIGUOUS)

    def test_agent_entry_to_resource_record(self) -> None:
        entry = AgentCatalogEntry(
            agent_id="agent-1",
            sdk_source_dir="/tmp/sdk",
            dsl={"name": "verify-bot"},
            model="gpt-4o",
            class_name="DemoAgent",
            module_name="fake_sdk.agent",
            init_kwargs={"temperature": 0.2},
            metadata={"source_tool": "agentbuilder-build-agent"},
            resource_type="AgentConfig",
            handle_field="agent_id",
        )
        record = agent_entry_to_resource_record(entry, bundle_id="core")
        self.assertEqual(record.resource_type, "agentconfig")
        self.assertEqual(record.resource_id, "agent-1")
        self.assertEqual(record.payload["agent_catalog_entry"]["agent_id"], "agent-1")
        self.assertEqual(record.materializer["class_name"], "DemoAgent")

    def test_build_resource_record_from_create_omits_legacy_blob(self) -> None:
        from extensions.sop_converter.resource_catalog import (
            build_resource_record_from_create,
        )

        record = build_resource_record_from_create(
            resource_id="bot-1",
            resource_type="agent",
            handle_field="agent_id",
            snapshot={"id": "bot-1", "name": "verify-bot"},
            class_name="DemoAgent",
            module_name="fake.agent",
            model="deepseek-chat",
            provider="deepseek",
            source_tool="create-llm-agent",
        )
        self.assertEqual(record.resource_id, "bot-1")
        self.assertEqual(record.materializer["class_name"], "DemoAgent")
        self.assertEqual(record.payload["model"], "deepseek-chat")
        self.assertNotIn("agent_catalog_entry", record.payload)

    def test_resource_error_envelope(self) -> None:
        out = resource_error(
            RESOURCE_CATALOG_MISSING,
            "missing",
            resource_type="AgentConfig",
            resource_id="agent-1",
        )
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_code"], RESOURCE_CATALOG_MISSING)
        self.assertEqual(out["resource_type"], "agentconfig")


class TestCatalogMutateAndMgmt(unittest.TestCase):
    def test_delete_does_not_resurrect_with_merge_false(self) -> None:
        path = _tmp_path() / "catalog.json"
        cat = ResourceCatalog()
        cat.upsert(_make_record(resource_id="a"))
        cat.save(path)
        from extensions.sop_converter.resource_catalog import delete_resource_at

        self.assertTrue(delete_resource_at(path, "agentconfig", "a"))
        loaded = ResourceCatalog.load(path)
        self.assertIsNone(loaded.records.get("agentconfig:a"))

    def test_latest_active_only(self) -> None:
        cat = ResourceCatalog()
        r1 = cat.upsert(_make_record(resource_id="old"))
        r1.updated_at = "2020-01-01T00:00:00+00:00"
        cat.records[r1.key()] = r1
        cat.upsert(_make_record(resource_id="new"))
        cat.mark_failed("agentconfig", "new", reason="boom")
        latest = cat.latest("agentconfig")
        self.assertEqual(latest.resource_id, "old")

    def test_put_prepared_preserves_updated_at(self) -> None:
        cat = ResourceCatalog()
        rec = _make_record()
        rec.updated_at = "2024-01-01T00:00:00+00:00"
        cat.put_prepared(rec)
        self.assertEqual(cat.records[rec.key()].updated_at, "2024-01-01T00:00:00+00:00")

    def test_delete_resource_at_missing_key_returns_false(self) -> None:
        path = _tmp_path() / "catalog.json"
        ResourceCatalog().save(path)
        from extensions.sop_converter.resource_catalog import delete_resource_at

        self.assertFalse(delete_resource_at(path, "agentconfig", "missing"))


class TestCrossLayerResolve(unittest.TestCase):
    def test_session_shadows_older_bundle(self) -> None:
        home = _tmp_path()
        bundle = _tmp_path() / "b"
        bundle.mkdir()
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home), SESSION_ID_ENV: "s1"}):
            ctx = context_from_env(bundle_path=bundle, bundle_id="b", session_id="s1")
            sess = resolve_resource_catalog_path(bundle, bundle_id="b", session_id="s1", scope="session")
            bun = resolve_resource_catalog_path(bundle, bundle_id="b", scope="bundle")
            for loc, marker, ts in (
                (bun, "bundle", "2020-01-01T00:00:00+00:00"),
                (sess, "session", "2025-01-01T00:00:00+00:00"),
            ):
                cat = ResourceCatalog()
                rec = _make_record(resource_id="x")
                rec.payload["model"] = marker
                rec.updated_at = ts
                cat.put_prepared(rec)
                loc.ensure_parent()
                cat.save(loc.path, merge=False)
            resolved = resolve_record("x", resource_type="agentconfig", catalog_context=ctx)
            self.assertEqual(resolved.record.payload["model"], "session")
            self.assertEqual(resolved.location.reason, "session-local")

    def test_same_key_picks_newer_updated_at_not_ambiguous(self) -> None:
        home = _tmp_path()
        bundle = _tmp_path() / "b"
        bundle.mkdir()
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            ctx = context_from_env(bundle_path=bundle, bundle_id="b")
            bun = resolve_resource_catalog_path(bundle, bundle_id="b", scope="bundle")
            usr = resolve_resource_catalog_path(bundle, bundle_id="b", scope="user")
            for loc, marker, ts in (
                (bun, "bundle", "2020-01-01T00:00:00+00:00"),
                (usr, "user", "2025-01-01T00:00:00+00:00"),
            ):
                cat = ResourceCatalog()
                rec = _make_record(resource_id="x")
                rec.payload["model"] = marker
                rec.updated_at = ts
                cat.put_prepared(rec)
                loc.ensure_parent()
                cat.save(loc.path, merge=False)
            resolved = resolve_record("x", resource_type="agentconfig", catalog_context=ctx)
            self.assertEqual(resolved.record.payload["model"], "user")
            self.assertEqual(resolved.location.reason, "user-local")

    def test_tie_prefers_bundle_over_user(self) -> None:
        home = _tmp_path()
        bundle = _tmp_path() / "b"
        bundle.mkdir()
        ts = "2024-01-01T00:00:00+00:00"
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            ctx = context_from_env(bundle_path=bundle, bundle_id="b")
            bun = resolve_resource_catalog_path(bundle, bundle_id="b", scope="bundle")
            usr = resolve_resource_catalog_path(bundle, bundle_id="b", scope="user")
            for loc, marker in (
                (bun, "bundle"),
                (usr, "user"),
            ):
                cat = ResourceCatalog()
                rec = _make_record(resource_id="x")
                rec.payload["model"] = marker
                rec.updated_at = ts
                rec.created_at = ts
                cat.put_prepared(rec)
                loc.ensure_parent()
                cat.save(loc.path, merge=False)
            resolved = resolve_record("x", resource_type="agentconfig", catalog_context=ctx)
            self.assertEqual(resolved.record.payload["model"], "bundle")
            self.assertEqual(resolved.location.reason, "bundle-local")

    def test_get_resource_record_honors_session_env_without_context(self) -> None:
        home = _tmp_path()
        bundle = _tmp_path() / "b"
        bundle.mkdir()
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home), SESSION_ID_ENV: "s1"}):
            sess = resolve_resource_catalog_path(
                bundle,
                bundle_id="b",
                session_id="s1",
                scope="session",
            )
            cat = ResourceCatalog()
            rec = _make_record(resource_id="x")
            rec.payload["model"] = "session-only"
            cat.put_prepared(rec)
            sess.ensure_parent()
            cat.save(sess.path, merge=False)
            record = get_resource_record(
                "x",
                resource_type="agentconfig",
                bundle_path=bundle,
            )
            self.assertEqual(record.payload["model"], "session-only")


class TestPlanWriteTargets(unittest.TestCase):
    def test_default_single_write_bundle(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        ctx = context_from_env(bundle_path=bundle, bundle_id="mybundle")
        targets = plan_write_targets(ctx)
        self.assertEqual([t.reason for t in targets], ["bundle-local"])

    def test_dual_write_bundle_and_user(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        with patch.dict(os.environ, {DUAL_WRITE_ENV: "1", HOME_ROOT_ENV: str(_tmp_path())}):
            ctx = context_from_env(bundle_path=bundle, bundle_id="mybundle")
            reasons = [t.reason for t in plan_write_targets(ctx)]
        self.assertEqual(reasons, ["bundle-local", "user-local"])

    def test_session_additive(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        home = _tmp_path()
        with patch.dict(os.environ, {SESSION_ID_ENV: "sess-1", HOME_ROOT_ENV: str(home)}):
            ctx = context_from_env(bundle_path=bundle, bundle_id="mybundle")
            reasons = [t.reason for t in plan_write_targets(ctx)]
        self.assertEqual(reasons[0], "session-local")
        self.assertIn("bundle-local", reasons)


class TestPayloadRefAndStorageView(unittest.TestCase):
    def test_get_stored_does_not_restore_secrets(self) -> None:
        path = _tmp_path() / "c.json"
        cat = ResourceCatalog()
        cat.upsert(_make_record())
        cat.save(path)
        with patch.dict(os.environ, {"CLAWCODEX_CORE_API_KEY": "sk-live"}):
            loaded = ResourceCatalog.load(path)
            stored = loaded.get_stored("agentconfig", "agent-1")
            runtime = loaded.get("agentconfig", "agent-1")
        assert stored is not None
        self.assertNotEqual(stored.payload["dsl"]["api_key"], "sk-live")
        self.assertEqual(runtime.payload["dsl"]["api_key"], "sk-live")

    def test_spill_uses_relative_hashed_ref_and_name_index(self) -> None:
        from extensions.sop_converter.resource_catalog import spill_payload_if_needed

        catalog_dir = _tmp_path()
        rec = _make_record()
        rec.payload["dsl"] = {"name": "verify-bot", "blob": "x" * 70000}
        out = spill_payload_if_needed(rec, catalog_dir, force_ref=True)
        self.assertEqual(out.payload["kind"], "payload_ref")
        self.assertFalse(Path(out.payload["ref"]).is_absolute())
        self.assertIn("verify-bot", out.payload["name_index"])
        self.assertTrue((catalog_dir / out.payload["ref"]).is_file())

    def test_rejects_path_escape(self) -> None:
        from extensions.sop_converter.resource_catalog import safe_payload_path

        with self.assertRaises(ResourceCatalogError):
            safe_payload_path(_tmp_path(), "../secret.json")

    def test_name_match_uses_name_index_without_opening_ref(self) -> None:
        cat = ResourceCatalog()
        rec = _make_record(resource_id="agent-spilled")
        rec.payload = {
            "kind": "payload_ref",
            "ref": "payloads/missing-file.json",
            "name_index": ["verify-bot"],
            "handle_field": "agent_id",
        }
        cat.put_prepared(rec)
        matches = cat.find_by_agent_reference("verify-bot")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].resource_id, "agent-spilled")

    def test_get_resource_record_resolves_payload_ref_without_catalog_dir(self) -> None:
        from extensions.sop_converter.resource_catalog import spill_payload_if_needed

        home = _tmp_path()
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            bundle_loc = resolve_resource_catalog_path(
                bundle,
                bundle_id="mybundle",
                scope="bundle",
            )
            catalog_dir = bundle_loc.path.parent
            rec = _make_record(resource_id="spilled-agent")
            rec.payload["dsl"] = {"name": "verify-bot", "blob": "x" * 70000}
            spilled = spill_payload_if_needed(rec, catalog_dir, force_ref=True)
            self.assertEqual(spilled.payload["kind"], "payload_ref")

            cat = ResourceCatalog()
            cat.put_prepared(spilled)
            bundle_loc.ensure_parent()
            cat.save(bundle_loc.path, merge=False)

            record = get_resource_record(
                "spilled-agent",
                resource_type="agentconfig",
                bundle_path=bundle,
            )
            self.assertEqual(record.payload["kind"], "inline")
            self.assertIn("dsl", record.payload)
            self.assertEqual(record.payload["dsl"]["name"], "verify-bot")

    def test_get_resource_record_restores_secrets_after_payload_ref_resolve(self) -> None:
        from extensions.sop_converter.resource_catalog import spill_payload_if_needed

        home = _tmp_path()
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            bundle_loc = resolve_resource_catalog_path(
                bundle,
                bundle_id="mybundle",
                scope="bundle",
            )
            catalog_dir = bundle_loc.path.parent
            rec = _make_record(resource_id="spilled-secret")
            # Keep api_key so spill redacts it into the external payload file.
            spilled = spill_payload_if_needed(rec, catalog_dir, force_ref=True)
            self.assertEqual(spilled.payload["kind"], "payload_ref")
            payload_path = catalog_dir / spilled.payload["ref"]
            spilled_raw = payload_path.read_text(encoding="utf-8")
            self.assertNotIn("sk-secret", spilled_raw)
            self.assertIn("CLAWCODEX_CORE_API_KEY", spilled_raw)

            cat = ResourceCatalog()
            cat.put_prepared(spilled)
            bundle_loc.ensure_parent()
            cat.save(bundle_loc.path, merge=False)

            with patch.dict(os.environ, {"CLAWCODEX_CORE_API_KEY": "sk-restored-live"}):
                record = get_resource_record(
                    "spilled-secret",
                    resource_type="agentconfig",
                    bundle_path=bundle,
                )
            self.assertEqual(record.payload["kind"], "inline")
            self.assertEqual(record.payload["dsl"]["api_key"], "sk-restored-live")

    def test_delete_resource_at_unlinks_spilled_payload_file(self) -> None:
        from extensions.sop_converter.resource_catalog import (
            delete_resource_at,
            spill_payload_if_needed,
        )

        catalog_dir = _tmp_path()
        catalog_path = catalog_dir / "catalog.json"
        rec = _make_record(resource_id="spilled-agent")
        rec.payload["dsl"] = {"name": "verify-bot", "blob": "x" * 70000}
        spilled = spill_payload_if_needed(rec, catalog_dir, force_ref=True)
        cat = ResourceCatalog()
        cat.put_prepared(spilled)
        cat.save(catalog_path)

        payload_path = catalog_dir / spilled.payload["ref"]
        self.assertTrue(payload_path.is_file())

        self.assertTrue(delete_resource_at(catalog_path, "agentconfig", "spilled-agent"))
        self.assertFalse(payload_path.is_file())
        loaded = ResourceCatalog.load(catalog_path)
        self.assertIsNone(loaded.get("agentconfig", "spilled-agent"))


class TestWriteRecord(unittest.TestCase):
    def test_dual_write_identical_updated_at(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        home = _tmp_path()
        with patch.dict(os.environ, {DUAL_WRITE_ENV: "1", HOME_ROOT_ENV: str(home)}):
            ctx = context_from_env(bundle_path=bundle, bundle_id="mybundle")
            result = write_record(_make_record(resource_id="dual-agent"), ctx)

        self.assertIsNone(result.failed_layer)
        self.assertFalse(result.retryable)
        self.assertEqual(sorted(result.written_layers), ["bundle", "user"])
        bundle_path = Path(result.catalog_paths["bundle"])
        user_path = Path(result.catalog_paths["user"])
        self.assertTrue(bundle_path.is_file())
        self.assertTrue(user_path.is_file())
        bundle_rec = ResourceCatalog.load(bundle_path).get_stored("agentconfig", "dual-agent")
        user_rec = ResourceCatalog.load(user_path).get_stored("agentconfig", "dual-agent")
        assert bundle_rec is not None and user_rec is not None
        self.assertEqual(bundle_rec.updated_at, user_rec.updated_at)
        self.assertEqual(result.resource_catalog_path, str(bundle_path))

    def test_partial_failure_envelope(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        home = _tmp_path()
        original_save = ResourceCatalog._save_unlocked

        def patched_save(self, path: Path, *, merge: bool = True) -> None:
            if "sop-resources" in str(path).replace("\\", "/"):
                raise OSError("simulated user write failure")
            original_save(self, path, merge=merge)

        with patch.dict(os.environ, {DUAL_WRITE_ENV: "1", HOME_ROOT_ENV: str(home)}):
            ctx = context_from_env(bundle_path=bundle, bundle_id="mybundle")
            with patch.object(ResourceCatalog, "_save_unlocked", patched_save):
                result = write_record(_make_record(resource_id="partial-agent"), ctx)

        self.assertEqual(result.written_layers, ["bundle"])
        self.assertEqual(result.failed_layer, "user")
        self.assertTrue(result.retryable)
        self.assertIn("bundle", result.catalog_paths)
        self.assertNotIn("user", result.catalog_paths)
        self.assertEqual(result.resource_catalog_path, result.catalog_paths["bundle"])
        self.assertIn(RESOURCE_CATALOG_WRITE_FAILED, result.error)
        bundle_path = Path(result.catalog_paths["bundle"])
        stored = ResourceCatalog.load(bundle_path).get_stored("agentconfig", "partial-agent")
        self.assertIsNotNone(stored)

    def test_write_record_honors_payload_kind_metadata(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        home = _tmp_path()
        forced = _make_record(resource_id="force-ref")
        forced.metadata = {"payload_kind": "payload_ref"}
        forced.payload["dsl"] = {"name": "tiny"}
        kept_inline = _make_record(resource_id="force-inline")
        kept_inline.metadata = {"payload_kind": "inline"}
        kept_inline.payload["dsl"] = {"name": "huge", "blob": "x" * 70000}

        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            ctx = context_from_env(bundle_path=bundle, bundle_id="mybundle")
            ref_result = write_record(forced, ctx)
            inline_result = write_record(kept_inline, ctx)

        ref_path = Path(ref_result.catalog_paths["bundle"])
        inline_path = Path(inline_result.catalog_paths["bundle"])
        ref_rec = ResourceCatalog.load(ref_path).get_stored("agentconfig", "force-ref")
        inline_rec = ResourceCatalog.load(inline_path).get_stored("agentconfig", "force-inline")
        assert ref_rec is not None and inline_rec is not None
        self.assertEqual(ref_rec.payload["kind"], "payload_ref")
        self.assertEqual(inline_rec.payload["kind"], "inline")
        self.assertIn("blob", inline_rec.payload["dsl"])

    def test_write_record_overwrite_preserves_created_at_and_merges_env_refs(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        home = _tmp_path()
        first = _make_record(resource_id="overwrite-1")
        first.secrets = {"env_refs": ["CLAWCODEX_CORE_TOKEN"]}
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            ctx = context_from_env(bundle_path=bundle, bundle_id="mybundle")
            first_result = write_record(first, ctx)
            catalog_path = Path(first_result.catalog_paths["bundle"])
            stored_first = ResourceCatalog.load(catalog_path).get_stored("agentconfig", "overwrite-1")
            assert stored_first is not None
            created = stored_first.created_at

            second = _make_record(resource_id="overwrite-1")
            second.payload["dsl"] = {"name": "verify-bot", "api_key": "sk-new"}
            second.secrets = {"env_refs": ["CLAWCODEX_CORE_OTHER"]}
            write_record(second, ctx)

        stored = ResourceCatalog.load(catalog_path).get_stored("agentconfig", "overwrite-1")
        assert stored is not None
        self.assertEqual(stored.created_at, created)
        self.assertEqual(
            stored.secrets.get("env_refs"),
            ["CLAWCODEX_CORE_API_KEY", "CLAWCODEX_CORE_OTHER", "CLAWCODEX_CORE_TOKEN"],
        )


if __name__ == "__main__":
    unittest.main()
