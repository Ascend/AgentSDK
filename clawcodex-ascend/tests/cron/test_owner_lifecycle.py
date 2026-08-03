"""F-22-F-4 / F-22-F-5 stub tests — teammate integration awaits.

The teammate subsystem (``TeammateManager``) is not yet wired into the
cron pipeline, so these hooks are stubs:

- :meth:`CronScheduler.notify_owner_exited` only invokes the optional
  ``on_owner_exited`` hook and logs; no runs are auto-finalized.
- :meth:`CronScheduler.cleanup_orphaned_tasks` only runs when
  ``active_agents_provider`` is set, and never auto-deletes.

These tests pin that contract so the future integration can be done
without breaking existing behavior. Once teammate is wired, the
tests here will be extended to assert auto-finalization.
"""

from __future__ import annotations

from clawcodex_ext.cron_system.scheduler import CronScheduler
from clawcodex_ext.cron_system.tasks import (
    add_cron_task,
    read_all_cron_tasks,
    write_cron_tasks,
)


def _outbox(prompt: str) -> None:
    return None  # placeholder; no scheduler-side fire for these tests


def _make_scheduler(tmp_path, **kwargs) -> CronScheduler:
    return CronScheduler(tmp_path, on_fire=_outbox, **kwargs)


def test_notify_owner_exited_invokes_hook(tmp_path):
    seen: list[str] = []
    scheduler = _make_scheduler(tmp_path, on_owner_exited=seen.append)
    assert scheduler.notify_owner_exited("agent-7") == []
    assert seen == ["agent-7"]


def test_notify_owner_exited_without_hook_is_noop(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    # No hook set — must not raise, returns empty list.
    assert scheduler.notify_owner_exited("agent-7") == []


def test_notify_owner_exited_rejects_blank_agent(tmp_path):
    called: list[str] = []
    scheduler = _make_scheduler(tmp_path, on_owner_exited=called.append)
    assert scheduler.notify_owner_exited("") == []
    assert not called  # blank agent id is ignored, hook not invoked


def test_notify_owner_exited_swallows_hook_exception(tmp_path):
    def _boom(agent_id: str) -> None:
        raise RuntimeError("simulated hook failure")

    scheduler = _make_scheduler(tmp_path, on_owner_exited=_boom)
    # Defensive: hook exceptions must not crash the scheduler call site.
    assert scheduler.notify_owner_exited("agent-7") == []


def test_cleanup_orphaned_tasks_requires_provider(tmp_path):
    """No provider set → no work done, returns empty list."""
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="orphan?",
        recurring=True,
        durable=True,
        created_at=1_000,
    )
    # Tag the task with an agent id so the provider would matter.
    tasks = read_all_cron_tasks(tmp_path)
    tagged = []
    for t in tasks:
        if hasattr(t, "agent_id"):
            from dataclasses import replace

            tagged.append(replace(t, agent_id="ghost-agent"))
        else:
            tagged.append(t)
    write_cron_tasks(tmp_path, tagged)

    scheduler = _make_scheduler(tmp_path)
    assert scheduler.cleanup_orphaned_tasks() == []


def test_cleanup_orphaned_tasks_uses_provider(tmp_path):
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="orphan?",
        recurring=True,
        durable=True,
        created_at=1_000,
    )
    tasks = read_all_cron_tasks(tmp_path)
    from dataclasses import replace

    tagged = [replace(t, agent_id="alive-agent") if hasattr(t, "agent_id") else t for t in tasks]
    write_cron_tasks(tmp_path, tagged)

    scheduler = _make_scheduler(
        tmp_path,
        active_agents_provider=lambda: {"alive-agent"},
    )
    assert scheduler.cleanup_orphaned_tasks() == []  # alive-agent is active

    scheduler2 = _make_scheduler(
        tmp_path,
        active_agents_provider=lambda: {"other-agent"},
    )
    orphaned = scheduler2.cleanup_orphaned_tasks()
    assert len(orphaned) == 1
    assert orphaned[0].agent_id == "alive-agent"


def test_cleanup_orphaned_tasks_swallows_provider_exception(tmp_path):
    def _boom() -> set[str]:
        raise RuntimeError("provider crashed")

    scheduler = _make_scheduler(
        tmp_path,
        active_agents_provider=_boom,
    )
    # Defensive: a failing provider must not break the scheduler.
    assert scheduler.cleanup_orphaned_tasks() == []


def test_scheduler_default_has_no_owner_hooks(tmp_path):
    """Default construction has no F-4/F-5 callbacks wired."""
    scheduler = _make_scheduler(tmp_path)
    assert scheduler.on_owner_exited is None
    assert scheduler.active_agents_provider is None
