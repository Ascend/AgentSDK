"""CronDispatchBridge unit tests (F-22-G-2).

The bridge must:
- pop typed ``CronPromptEvent`` / ``CronMissedEvent`` from the outbox;
- also accept legacy dict-style entries (``{"type": "cron_prompt", ...}``);
- leave non-cron entries (Generic / Proactive / unknown) untouched;
- apply the accumulation guard (duplicate task_id → cancel run) at the
  caller side, not inside the bridge — the bridge is pure dispatch.
"""

from __future__ import annotations

from clawcodex_ext.cron_system.dispatch import (
    CronDispatchBridge,
    CronMissedDispatch,
)
from clawcodex_ext.cron_system.runs import (
    CreateCronRunParams,
    create_queued_run,
    finalize_cron_run,
)
from clawcodex_ext.cron_system.scheduler import now_ms
from clawcodex_ext.query.outbox_types import (  # pylint: disable=no-name-in-module
    CronMissedEvent,
    CronPromptEvent,
    ProactivePromptEvent,
)


def _wrap(prompt: str, task_id: str, run_id: str) -> str:
    return f"[{task_id}/{run_id}] {prompt}"


def test_drain_typed_cron_prompt(tmp_path):
    bridge = CronDispatchBridge(tmp_path, wrap_prompt=_wrap)
    outbox: list = [
        CronPromptEvent(prompt="hello", task_id="t1", run_id="r1"),
    ]
    events = bridge.drain(outbox)
    assert len(events) == 1
    assert events[0].prompt == "hello"
    assert events[0].task_id == "t1"
    assert events[0].run_id == "r1"
    assert events[0].wrapped_prompt == "[t1/r1] hello"
    assert not outbox  # drained entries removed


def test_drain_legacy_dict_cron_prompt(tmp_path):
    """Backward compat — pre-typed outbox producers kept working."""
    bridge = CronDispatchBridge(tmp_path, wrap_prompt=_wrap)
    outbox: list = [
        {"type": "cron_prompt", "prompt": "legacy", "task_id": "t2", "run_id": "r2"},
    ]
    events = bridge.drain(outbox)
    assert len(events) == 1
    assert events[0].task_id == "t2"
    assert events[0].run_id == "r2"
    assert not outbox


def test_drain_mixed_typed_and_dict(tmp_path):
    bridge = CronDispatchBridge(tmp_path, wrap_prompt=_wrap)
    outbox: list = [
        CronPromptEvent(prompt="a", task_id="t1", run_id="r1"),
        {"type": "cron_prompt", "prompt": "b", "task_id": "t2", "run_id": "r2"},
        ProactivePromptEvent(prompt="ignored", source="tick"),
        {"type": "user_question", "questions": ["q1"]},
        {"tool": "Brief", "text": "preview"},
    ]
    events = bridge.drain(outbox)
    assert [e.task_id for e in events] == ["t1", "t2"]
    # non-cron events kept in outbox in original order
    assert len(outbox) == 3
    assert isinstance(outbox[0], ProactivePromptEvent)
    assert outbox[1] == {"type": "user_question", "questions": ["q1"]}
    assert outbox[2] == {"tool": "Brief", "text": "preview"}


def test_drain_skips_blank_prompts(tmp_path):
    bridge = CronDispatchBridge(tmp_path, wrap_prompt=_wrap)
    outbox: list = [
        CronPromptEvent(prompt="   ", task_id="t1", run_id="r1"),
        CronPromptEvent(prompt="ok", task_id="t2", run_id="r2"),
    ]
    events = bridge.drain(outbox)
    assert [e.task_id for e in events] == ["t2"]
    # blank entry stays in outbox
    assert len(outbox) == 1
    assert outbox[0].prompt == "   "


def test_drain_missed_typed_and_dict(tmp_path):
    bridge = CronDispatchBridge(tmp_path, wrap_prompt=_wrap)
    outbox: list = [
        CronMissedEvent(tasks=["t1"], notification="missed one-shot t1"),
        {"type": "cron_missed", "tasks": ["t2", "t3"], "notification": "missed two"},
        CronPromptEvent(prompt="kept", task_id="t4", run_id="r4"),
    ]
    missed = bridge.drain_missed(outbox)
    assert len(missed) == 2
    assert isinstance(missed[0], CronMissedDispatch)
    assert missed[0].task_ids == ["t1"]
    assert missed[0].notification == "missed one-shot t1"
    assert missed[1].task_ids == ["t2", "t3"]
    # cron_prompt preserved in outbox for a separate drain() call
    assert len(outbox) == 1
    assert outbox[0].task_id == "t4"


def test_drain_missed_skips_blank_notifications(tmp_path):
    bridge = CronDispatchBridge(tmp_path, wrap_prompt=_wrap)
    outbox: list = [
        CronMissedEvent(tasks=["t1"], notification="   "),
        CronMissedEvent(tasks=["t2"], notification="real"),
    ]
    missed = bridge.drain_missed(outbox)
    assert [m.task_ids for m in missed] == [["t2"]]
    assert len(outbox) == 1


def test_claim_and_finalize_lifecycle(tmp_path):
    """Bridge.claim() and bridge.finalize() delegate to runs.py."""
    bridge = CronDispatchBridge(tmp_path, wrap_prompt=_wrap)
    run = create_queued_run(
        tmp_path,
        CreateCronRunParams(
            task_id="t1",
            prompt="hi",
            queued_at=now_ms(),
        ),
    )
    assert run is not None
    assert bridge.claim("t1", run.id) == "t1"
    bridge.finalize("t1", run.id, "completed")
    # second claim after finalize is a no-op (run no longer queued)
    assert bridge.claim("t1", run.id) is None
    # finalize is idempotent for the "cancelled" path
    finalize_cron_run(tmp_path, run.id, "cancelled")
    bridge.finalize("t1", run.id, "cancelled")


def test_default_wrap_prompt_includes_task_id_and_time(tmp_path):
    """The bridge's default wrapper reproduces the legacy header format."""
    bridge = CronDispatchBridge(tmp_path)  # wrap_prompt=None → default
    outbox: list = [CronPromptEvent(prompt="do thing", task_id="abc12345", run_id="r1")]
    events = bridge.drain(outbox)
    assert len(events) == 1
    wrapped = events[0].wrapped_prompt
    assert wrapped.startswith("✻ Running scheduled task")
    assert "· abc12345" in wrapped
    assert "do thing" in wrapped
