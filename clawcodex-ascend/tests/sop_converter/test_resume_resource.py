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

"""resume-resource: generic ResourceHandler recovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extensions.sop_converter.composite_runtime import CompositeWorkflowRunner
from extensions.sop_converter.composite_workflows import (
    invoke_existing_agent_workflow,
    resume_resource_workflow,
)
from extensions.sop_converter.resource_catalog import (
    CatalogExecutionContext,
    ResourceCatalog,
    ResourceRecord,
    resolve_resource_catalog_path,
    spill_payload_if_needed,
)
from extensions.sop_converter.resource_handlers import (
    RESOURCE_TYPE_UNREGISTERED,
    ResourceHandler,
    register_resource_handler,
)
from extensions.sop_converter.resource_runtime import materialize_resource


def _register_demo_handler() -> None:
    def materialize(record: ResourceRecord) -> dict:
        return {"demo": {"id": record.resource_id, "payload": record.payload}}

    def invoke(
        record: ResourceRecord,
        query: str = "",
        inputs: object = None,
    ) -> dict:
        del inputs
        resource = materialize(record)
        return {
            "text": query or "ok",
            "resource_id": resource["demo"]["id"],
            "output": {"echo": query or "ok"},
            "raw": {"echo": query or "ok"},
        }

    register_resource_handler(
        ResourceHandler(
            resource_type="DemoHandle",
            materialize=materialize,
            invoke=invoke,
            public_output_schema={
                "type": "object",
                "required": ["text", "resource_id"],
                "properties": {
                    "text": {"type": "string"},
                    "resource_id": {"type": "string"},
                },
            },
            error_codes=frozenset({"resource_materialize_failed"}),
        ),
        replace=True,
    )


class TestResumeResourceWorkflow(unittest.TestCase):
    def test_resource_record_identity_is_shared_with_core_handler(self) -> None:
        """Guard against dual ResourceRecord copies breaking resume-resource."""
        from extensions.sop_converter.core.resource_catalog import (
            ResourceRecord as CoreRecord,
        )
        from extensions.sop_converter.resource_handlers import require_resource_handler

        self.assertIs(ResourceRecord, CoreRecord)
        handler = require_resource_handler("agent")
        self.assertEqual(
            handler.materialize.__module__,
            "extensions.sop_converter.core.agent_runtime",
        )

    def test_workflow_is_trusted_with_private_catalog_lane(self) -> None:
        spec = resume_resource_workflow()
        self.assertEqual(spec.name, "resume-resource")
        self.assertTrue(spec.trusted)
        self.assertEqual(
            {step.id for step in spec.steps},
            {"load_resource_record", "materialize_resource", "invoke_resource"},
        )
        private_ids = {step.id for step in spec.steps if step.visibility == "private"}
        self.assertEqual(private_ids, {"load_resource_record", "materialize_resource"})
        public = next(step for step in spec.steps if step.id == "invoke_resource")
        self.assertEqual(public.visibility, "public")

    def test_demo_handle_resume_via_registry(self) -> None:
        _register_demo_handler()
        record = ResourceRecord(
            resource_type="DemoHandle",
            resource_id="demo-1",
            bundle_id="test",
            source_tool="create-demo",
            materializer={"kind": "demo"},
            invoker={"kind": "demo"},
            payload={"handle_field": "id", "value": "demo-1"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            path = resolve_resource_catalog_path(bundle).path
            catalog = ResourceCatalog()
            catalog.upsert(record)
            catalog.save(path)

            result = CompositeWorkflowRunner().run(
                resume_resource_workflow(),
                {
                    "resource_type": "DemoHandle",
                    "resource_ref": "demo-1",
                    "query": "ping",
                },
                resources={
                    "catalog": CatalogExecutionContext(
                        bundle_path=bundle,
                        bundle_id="test",
                    )
                },
            )

        self.assertFalse(result.is_error, msg=result.error)
        self.assertEqual(result.output["text"], "ping")
        self.assertEqual(result.output["resource_id"], "demo-1")
        self.assertEqual(result.output["resource_type"], "demohandle")
        self.assertEqual(result.output["resource_ref"], "demo-1")
        self.assertEqual(
            [step.status for step in result.trace],
            ["success", "success", "success"],
        )

    def test_unregistered_type_does_not_silently_use_agent(self) -> None:
        result = CompositeWorkflowRunner().run(
            resume_resource_workflow(),
            {
                "resource_type": "totally-unknown-type",
                "resource_ref": "x",
                "query": "ping",
            },
            resources={"catalog": CatalogExecutionContext(bundle_id="test")},
        )
        self.assertTrue(result.is_error)
        self.assertEqual(result.error_code, RESOURCE_TYPE_UNREGISTERED)
        self.assertIn(RESOURCE_TYPE_UNREGISTERED, result.error)

    def test_builtin_macro_and_composite_tool_registered(self) -> None:
        from extensions.sop_converter.runtime.composite_tools.builtin import (
            builtin_composite_tools,
        )
        from extensions.sop_converter.runtime.macros.catalog import ensure_builtin_macros

        catalog = ensure_builtin_macros()
        self.assertIsNotNone(catalog.get("builtin:resume-resource"))

        names = {spec.name for spec in builtin_composite_tools()}
        self.assertIn("resume_resource", names)
        resume = next(s for s in builtin_composite_tools() if s.name == "resume_resource")
        self.assertEqual(resume.call_type, "workflow")
        self.assertEqual(
            resume.call_impl,
            {"catalog_id": "builtin:resume-resource"},
        )

    def test_workflows_bind_catalog_dir_from_resolved_location(self) -> None:
        from extensions.sop_converter.runtime.composite_workflows import (
            invoke_existing_agent_workflow as runtime_invoke_workflow,
            resume_resource_workflow as runtime_resume_workflow,
        )

        for resume in (resume_resource_workflow(), runtime_resume_workflow()):
            load = next(step for step in resume.steps if step.id == "load_resource_record")
            materialize = next(step for step in resume.steps if step.id == "materialize_resource")
            self.assertEqual(
                load.callable_ref,
                "extensions.sop_converter.resource_catalog:resolve_record",
            )
            self.assertEqual(
                materialize.args["catalog_dir"],
                "$private.load_resource_record.output.location.path.parent",
            )

        for invoke in (invoke_existing_agent_workflow(), runtime_invoke_workflow()):
            agent_load = next(step for step in invoke.steps if step.id == "load_agent_record")
            agent_materialize = next(step for step in invoke.steps if step.id == "materialize_agent")
            self.assertEqual(
                agent_load.callable_ref,
                "extensions.sop_converter.resource_catalog:resolve_agent_record",
            )
            self.assertEqual(
                agent_materialize.args["catalog_dir"],
                "$private.load_agent_record.output.location.path.parent",
            )

    def test_spilled_payload_ref_resume_passes_catalog_dir(self) -> None:
        _register_demo_handler()
        record = ResourceRecord(
            resource_type="DemoHandle",
            resource_id="demo-spill",
            bundle_id="test",
            source_tool="create-demo",
            materializer={"kind": "demo"},
            invoker={"kind": "demo"},
            payload={
                "handle_field": "id",
                "value": "demo-spill",
                "blob": "x" * 70000,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            location = resolve_resource_catalog_path(bundle)
            catalog_dir = location.path.parent
            spilled = spill_payload_if_needed(record, catalog_dir, force_ref=True)
            self.assertEqual(spilled.payload["kind"], "payload_ref")

            catalog = ResourceCatalog()
            catalog.upsert(spilled)
            catalog.save(location.path)

            captured: dict[str, object] = {}

            def _spy(
                record: ResourceRecord,
                resource_type: str = "",
                *,
                catalog_dir: Path | str | None = None,
            ) -> dict:
                captured["catalog_dir"] = catalog_dir
                return materialize_resource(
                    record,
                    resource_type,
                    catalog_dir=catalog_dir,
                )

            with patch(
                "extensions.sop_converter.resource_runtime.materialize_resource",
                side_effect=_spy,
            ):
                result = CompositeWorkflowRunner().run(
                    resume_resource_workflow(),
                    {
                        "resource_type": "DemoHandle",
                        "resource_ref": "demo-spill",
                        "query": "spill-ok",
                    },
                    resources={
                        "catalog": CatalogExecutionContext(
                            bundle_path=bundle,
                            bundle_id="test",
                        )
                    },
                )

        self.assertFalse(result.is_error, msg=result.error)
        self.assertEqual(result.output["text"], "spill-ok")
        self.assertEqual(result.output["resource_id"], "demo-spill")
        self.assertIsNotNone(captured.get("catalog_dir"))
        self.assertEqual(Path(str(captured["catalog_dir"])), catalog_dir)


if __name__ == "__main__":
    unittest.main()
