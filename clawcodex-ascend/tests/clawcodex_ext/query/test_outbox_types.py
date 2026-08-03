"""Tests for clawcodex_ext/query/outbox_types.py (P102-C)."""

from __future__ import annotations


from clawcodex_ext.query.outbox_types import (
    CronMissedEvent,
    CronPromptEvent,
    GenericOutboxEvent,
    outbox_event_from_dict,
)


class TestCronPromptEvent:
    def test_get(self):
        e = CronPromptEvent(prompt="hello", task_id="abc", run_id="123")
        assert e.get("prompt") == "hello"
        assert e.get("task_id") == "abc"
        assert e.get("run_id") == "123"
        assert e.get("missing") is None
        assert e.get("missing", "default") == "default"

    def test_getitem(self):
        e = CronPromptEvent(prompt="hello")
        assert e["prompt"] == "hello"

    def test_contains(self):
        e = CronPromptEvent(prompt="hello")
        assert "prompt" in e
        assert "missing" not in e


class TestCronMissedEvent:
    def test_get(self):
        e = CronMissedEvent(tasks=["t1", "t2"], notification="missed")
        assert e.get("tasks") == ["t1", "t2"]
        assert e.get("notification") == "missed"


class TestGenericOutboxEvent:
    def test_get(self):
        e = GenericOutboxEvent(payload={"tool": "Read", "path": "/tmp"})
        assert e.get("tool") == "Read"
        assert e.get("path") == "/tmp"
        assert e.get("missing") is None

    def test_getitem(self):
        e = GenericOutboxEvent(payload={"tool": "Read"})
        assert e["tool"] == "Read"

    def test_contains(self):
        e = GenericOutboxEvent(payload={"tool": "Read"})
        assert "tool" in e
        assert "missing" not in e

    def test_from_dict(self):
        e = GenericOutboxEvent.from_dict({"tool": "Read", "path": "/tmp"})
        assert e.get("tool") == "Read"
        assert e.get("path") == "/tmp"


class TestOutboxEventFromDict:
    def test_cron_prompt(self):
        e = outbox_event_from_dict({"type": "cron_prompt", "prompt": "p", "task_id": "t"})
        assert isinstance(e, CronPromptEvent)
        assert e.prompt == "p"
        assert e.task_id == "t"

    def test_cron_missed(self):
        e = outbox_event_from_dict({"type": "cron_missed", "tasks": ["a"], "notification": "n"})
        assert isinstance(e, CronMissedEvent)
        assert e.tasks == ["a"]

    def test_generic(self):
        e = outbox_event_from_dict({"tool": "Read", "path": "/tmp"})
        assert isinstance(e, GenericOutboxEvent)
        assert e.get("tool") == "Read"

    def test_empty_dict(self):
        e = outbox_event_from_dict({})
        assert isinstance(e, GenericOutboxEvent)


class TestTypeAnnotation:
    def test_tool_context_outbox_type(self):
        from clawcodex_ext.tool_system.context import ToolContext
        from dataclasses import fields

        outbox_field = next(f for f in fields(ToolContext) if f.name == "outbox")
        # The annotation should be list[OutboxEvent]
        assert "OutboxEvent" in str(outbox_field.type)
