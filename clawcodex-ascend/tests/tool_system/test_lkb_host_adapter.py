"""Tests for the LKB host adapter (Phase 4/8 host integration).

Verifies:
* Flag-off: try_handle returns (False, None) -> native Task-v2 path runs.
* Flag-on: TaskCreate/Get/List/Update route through the LKB Plan Graph
  Store and hydrate context.tasks from the Store (LKB-ADAPT-002/003/004).
* TaskOutput is never routed (LKB-ADAPT-014).
"""

from __future__ import annotations


# pylint: disable=W0108,W0621
from pathlib import Path
from types import SimpleNamespace

import pytest

from lkb.clawcodex_task_adapter import try_handle


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CLAWCODEX_HOME", str(home))
    return home


def _ctx(tasks: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(tasks=tasks or {}, agent_id="agent-a")


def test_flag_off_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: False)
    handled, result = try_handle("TaskCreate", {"subject": "x"}, _ctx())
    assert handled is False
    assert result is None


def test_task_output_never_routed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: True)
    handled, result = try_handle("TaskOutput", {"task_id": "T-1"}, _ctx())
    assert handled is False  # LKB-ADAPT-014
    assert result is None


def test_board_resolution_uses_workspace_and_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from lkb.clawcodex_task_adapter import _board_id

    calls: list[tuple[object, object]] = []

    class Repo:
        def resolve_board(self, workspace_root=None, *, session_id=None):
            calls.append((workspace_root, session_id))
            return SimpleNamespace(board_id="board")

    monkeypatch.setattr(
        "lkb.clawcodex_task_adapter._repo",
        lambda: Repo(),
    )
    context = SimpleNamespace(workspace_root=Path("project"), session_id="session-1")
    assert _board_id(context) == "board"
    assert calls == [(Path("project"), "session-1")]


def test_board_resolution_real_repository_accepts_session(
    tmp_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the concrete repository API aligned with the host adapter."""
    from lkb.clawcodex_task_adapter import _board_id
    from lkb.repository import JsonFileLkbRepository

    workspace = tmp_path / "project"
    workspace.mkdir()
    repository = JsonFileLkbRepository(home=tmp_home)
    monkeypatch.setattr("lkb.repository.get_repository", lambda *, home=None: repository)

    context = SimpleNamespace(workspace_root=workspace, session_id="session-1")
    board_id = _board_id(context)

    assert board_id
    assert repository.resolve_board(workspace, session_id="session-1").board_id == board_id


def test_board_resolution_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_plan_graph(monkeypatch)
    monkeypatch.setattr(
        "lkb.clawcodex_task_adapter._board_id",
        lambda context: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    handled, result = try_handle("TaskUpdate", {"taskId": "T-1", "status": "in_progress"}, _ctx())
    assert handled is True
    assert result is not None
    assert result.is_error is True
    assert result.output["success"] is False
    assert result.output["lkb"]["decision"] == "denied"


def test_create_routes_to_store_and_hydrates(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: True)
    from lkb.repository import JsonFileLkbRepository

    test_repo = JsonFileLkbRepository(home=tmp_home)
    board_id = test_repo.resolve_board(explicit_id="adapt").board_id
    monkeypatch.setattr("lkb.repository.get_repository", lambda *, home=None: test_repo)
    monkeypatch.setattr(test_repo, "resolve_board", lambda *a, **kw: SimpleNamespace(board_id=board_id))

    ctx = _ctx()
    handled, result = try_handle("TaskCreate", {"subject": "Hello", "description": "d"}, ctx)
    assert handled is True
    assert result.name == "TaskCreate"
    new_id = result.output["task"]["id"]
    # context.tasks hydrated from the Store.
    assert new_id in ctx.tasks
    assert ctx.tasks[new_id]["subject"] == "Hello"
    assert ctx.tasks[new_id]["status"] == "pending"


def test_get_and_list_after_create(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: True)
    from lkb.repository import JsonFileLkbRepository

    test_repo = JsonFileLkbRepository(home=tmp_home)
    board_id = test_repo.resolve_board(explicit_id="adapt2").board_id
    monkeypatch.setattr("lkb.repository.get_repository", lambda *, home=None: test_repo)
    monkeypatch.setattr(test_repo, "resolve_board", lambda *a, **kw: SimpleNamespace(board_id=board_id))

    ctx = _ctx()
    _, create_res = try_handle("TaskCreate", {"subject": "A", "description": "d"}, ctx)
    new_id = create_res.output["task"]["id"]

    # TaskGet (camelCase taskId).
    handled, get_res = try_handle("TaskGet", {"taskId": new_id}, ctx)
    assert handled is True
    assert get_res.output["task"]["id"] == new_id

    # TaskList.
    handled, list_res = try_handle("TaskList", {}, ctx)
    assert handled is True
    ids = {t["id"] for t in list_res.output["tasks"]}
    assert new_id in ids


def test_update_status_routes_to_start_task(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: True)
    from lkb.repository import JsonFileLkbRepository

    test_repo = JsonFileLkbRepository(home=tmp_home)
    board_id = test_repo.resolve_board(explicit_id="adapt3").board_id
    monkeypatch.setattr("lkb.repository.get_repository", lambda *, home=None: test_repo)
    monkeypatch.setattr(test_repo, "resolve_board", lambda *a, **kw: SimpleNamespace(board_id=board_id))

    ctx = _ctx()
    _, create_res = try_handle("TaskCreate", {"subject": "A", "description": "d"}, ctx)
    new_id = create_res.output["task"]["id"]
    # Claim then start (status=in_progress).
    try_handle("TaskUpdate", {"taskId": new_id, "owner": "agent-a"}, ctx)
    handled, upd_res = try_handle("TaskUpdate", {"taskId": new_id, "status": "in_progress"}, ctx)
    assert handled is True
    assert upd_res.output["task"]["updated"] is True
    assert upd_res.output["success"] is True
    assert upd_res.output["taskId"] == new_id
    assert upd_res.output["updatedFields"] == ["status"]
    # Hydrated context reflects the new status.
    assert ctx.tasks[new_id]["status"] == "in_progress"


def test_update_routes_all_sub_intents_without_dropping_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_plan_graph(monkeypatch)
    monkeypatch.setattr(
        "lkb.clawcodex_task_adapter._board_id",
        lambda context: "board",
    )
    monkeypatch.setattr(
        "lkb.clawcodex_task_adapter._plan_id",
        lambda context, board_id: "plan",
    )
    monkeypatch.setattr(
        "lkb.clawcodex_task_adapter._hydrate",
        lambda context, board_id, plan_id: None,
    )
    calls: list[tuple[str, dict]] = []
    from lkb.commands import CommandResult

    def fake_execute(kind, board_id, plan_id, payload, *, actor, command_id, reason=None, roles=()):
        calls.append((kind, dict(payload)))
        return CommandResult(decision="committed", command_id=command_id)

    monkeypatch.setattr(
        "lkb.clawcodex_task_adapter._execute",
        fake_execute,
    )
    context = _ctx({"T-1": {"id": "T-1", "status": "pending"}})

    _, result = try_handle(
        "TaskUpdate",
        {
            "taskId": "T-1",
            "addBlockedBy": ["T-2", "T-3"],
            "removeBlocks": ["T-4"],
        },
        context,
    )
    assert calls[-1] == (
        "patch_task",
        {
            "task_id": "T-1",
            "addBlockedBy": ["T-2", "T-3"],
            "removeBlocks": ["T-4"],
        },
    )
    assert result.output["updatedFields"] == ["blockedBy", "blocks"]

    try_handle(
        "TaskUpdate",
        {"taskId": "T-1", "owner": "agent-b", "subject": "keep this too"},
        context,
    )
    assert calls[-1][0] == "patch_task"
    assert calls[-1][1]["owner"] == "agent-b"
    assert calls[-1][1]["subject"] == "keep this too"

    try_handle("TaskUpdate", {"taskId": "T-1", "status": "deleted"}, context)
    assert calls[-1] == ("delete_task", {"task_id": "T-1", "status": "deleted"})


# ── C6 / LKB-ADAPT-009: denial shape keeps native fields consumable ──


def _enable_plan_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: True)


def _install_repo(monkeypatch: pytest.MonkeyPatch, tmp_home: Path, board_key: str):
    from lkb.repository import JsonFileLkbRepository

    test_repo = JsonFileLkbRepository(home=tmp_home)
    board_id = test_repo.resolve_board(explicit_id=board_key).board_id
    monkeypatch.setattr("lkb.repository.get_repository", lambda *, home=None: test_repo)
    monkeypatch.setattr(test_repo, "resolve_board", lambda *a, **kw: SimpleNamespace(board_id=board_id))
    return test_repo, board_id


def test_taskget_missing_returns_native_task_none(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C6/LKB-ADAPT-009: missing-task TaskGet returns ``{"task": None}``
    so legacy clients that read ``output["task"]`` don't KeyError.
    """
    _enable_plan_graph(monkeypatch)
    _install_repo(monkeypatch, tmp_home, "adapt-get-miss")
    ctx = _ctx()
    handled, result = try_handle("TaskGet", {"taskId": "T-nope"}, ctx)
    assert handled is True
    assert result.output["task"] is None
    # ``error`` must NOT be the only key — native fields are consumable.
    assert "error" not in result.output


def test_taskcreate_denial_keeps_native_task_shape(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C6/LKB-ADAPT-009: a denied TaskCreate still returns ``task.id`` and
    ``task.subject`` at the top level so legacy clients can read them.
    """
    _enable_plan_graph(monkeypatch)
    _install_repo(monkeypatch, tmp_home, "adapt-create-denied")

    # Force the LKB command to deny by making validate raise via a corrupted
    # subject (empty after strip) — the host adapter still builds the result.
    ctx = _ctx()
    # Subject is non-empty here; we deny by stubbing the dispatcher to return
    # a denied CommandResult.
    from lkb.commands import CommandResult

    def _fake_execute(kind, board_id, plan_id, payload, *, actor, command_id, reason=None, roles=()):
        return CommandResult(
            decision="denied",
            command_id=command_id,
            reason="validation_failed",
            validation_run_id="V-denied",
        )

    monkeypatch.setattr("lkb.clawcodex_task_adapter._execute", _fake_execute)
    handled, result = try_handle("TaskCreate", {"subject": "Hello", "description": "d"}, ctx)
    assert handled is True
    out = result.output
    # Native TaskCreate fields still consumable.
    assert out["task"]["id"]
    assert out["task"]["subject"] == "Hello"
    assert out["success"] is False
    assert out["status"] == "denied"
    # LKB detail isolated under ``lkb``.
    assert out["lkb"]["decision"] == "denied"
    assert out["lkb"]["validationRunId"] == "V-denied"
    # Legacy client that ignores ``lkb`` sees no ``error`` key.
    assert "error" not in out


def test_taskupdate_denial_keeps_native_taskid_updatedfields(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C6/LKB-ADAPT-009: a denied TaskUpdate still returns ``taskId`` and
    ``updatedFields`` so legacy clients can read them.
    """
    _enable_plan_graph(monkeypatch)
    _install_repo(monkeypatch, tmp_home, "adapt-update-denied")

    ctx = _ctx()
    # First create a real task so the update has a target.
    _, create_res = try_handle("TaskCreate", {"subject": "A", "description": "d"}, ctx)
    new_id = create_res.output["task"]["id"]

    from lkb.commands import CommandResult

    def _fake_execute(kind, board_id, plan_id, payload, *, actor, command_id, reason=None, roles=()):
        return CommandResult(
            decision="denied",
            command_id=command_id,
            reason="dependency_not_satisfied",
            validation_run_id="V-upd-denied",
        )

    monkeypatch.setattr("lkb.clawcodex_task_adapter._execute", _fake_execute)
    handled, result = try_handle("TaskUpdate", {"taskId": new_id, "status": "completed"}, ctx)
    assert handled is True
    out = result.output
    # Native TaskUpdate fields still consumable.
    assert out["taskId"] == new_id
    assert out["updatedFields"] == []
    assert out["success"] is False
    assert out["status"] == "denied"
    # LKB detail isolated under ``lkb``.
    assert out["lkb"]["decision"] == "denied"
    assert out["lkb"]["validationRunId"] == "V-upd-denied"
    # Legacy client that ignores ``lkb`` sees no ``error`` key.
    assert "error" not in out


def test_adapter_exception_returns_denial_shape(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C6/LKB-ADAPT-009: an adapter-level exception must not surface as
    ``{"error": ...}``; it returns a denial-shaped payload so legacy
    clients keep seeing native fields.
    """
    _enable_plan_graph(monkeypatch)
    _install_repo(monkeypatch, tmp_home, "adapt-exc")

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("lkb.clawcodex_task_adapter._execute", _boom)
    ctx = _ctx()
    # TaskUpdate path uses _adapter_error_payload.
    handled, result = try_handle("TaskUpdate", {"taskId": "T-x", "status": "in_progress"}, ctx)
    assert handled is True
    out = result.output
    assert out["success"] is False
    assert out["status"] == "denied"
    assert out["taskId"] == "T-x"
    assert out["updatedFields"] == []
    assert "boom" in out["lkb"]["adapterError"]
    assert "error" not in out

    # TaskCreate path returns native task shape + denial fields.
    handled, result = try_handle("TaskCreate", {"subject": "Hi", "description": "d"}, ctx)
    assert handled is True
    out = result.output
    assert out["task"]["subject"] == "Hi"
    assert out["success"] is False
    assert out["status"] == "denied"
    assert "error" not in out

    # TaskGet path doesn't call _execute, so it returns task=None from the
    # hydrate path (no exception). Verify it still has no ``error`` key.
    handled, result = try_handle("TaskGet", {"taskId": "T-y"}, ctx)
    assert handled is True
    assert result.output["task"] is None
    assert "error" not in result.output


# ── metadata.lkb type validation (parity with native tasks_v2) ───────


def test_taskcreate_rejects_non_dict_lkb_metadata(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``metadata.lkb`` must be an object: a scalar would be stored verbatim
    and later crash downstream readers (e.g. ``lkb.context_adapter``).  The
    native path rejects it via ``_validate_lkb_metadata``; the adapter must
    deny it too instead of persisting a corrupt shape.
    """
    _enable_plan_graph(monkeypatch)
    _install_repo(monkeypatch, tmp_home, "adapt-md-create")
    ctx = _ctx()
    handled, result = try_handle("TaskCreate", {"subject": "A", "metadata": {"lkb": "oops"}}, ctx)
    assert handled is True
    assert result.is_error is True
    out = result.output
    assert out["success"] is False
    assert out["reason"]["code"] == "invalid_metadata"
    assert "metadata.lkb" in out["reason"]["message"]
    assert out["task"]["subject"] == "A"  # native shape stays consumable
    assert ctx.tasks == {}  # nothing persisted


def test_taskupdate_rejects_non_dict_lkb_metadata(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_plan_graph(monkeypatch)
    _install_repo(monkeypatch, tmp_home, "adapt-md-update")
    ctx = _ctx()
    _, create_res = try_handle("TaskCreate", {"subject": "A", "description": "d"}, ctx)
    new_id = create_res.output["task"]["id"]

    handled, result = try_handle("TaskUpdate", {"taskId": new_id, "metadata": {"lkb": ["not", "a", "dict"]}}, ctx)
    assert handled is True
    assert result.is_error is True
    out = result.output
    assert out["success"] is False
    assert out["taskId"] == new_id
    assert out["updatedFields"] == []
    assert out["reason"]["code"] == "invalid_metadata"
    assert "metadata.lkb" in out["reason"]["message"]
    assert ctx.tasks[new_id]["metadata"] == {}  # stored metadata untouched


def test_taskupdate_rejects_non_dict_metadata(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_plan_graph(monkeypatch)
    _install_repo(monkeypatch, tmp_home, "adapt-md-scalar")
    ctx = _ctx()
    _, create_res = try_handle("TaskCreate", {"subject": "A", "description": "d"}, ctx)
    new_id = create_res.output["task"]["id"]

    handled, result = try_handle("TaskUpdate", {"taskId": new_id, "metadata": "oops"}, ctx)
    assert handled is True
    assert result.is_error is True
    out = result.output
    assert out["reason"]["code"] == "invalid_metadata"
    assert out["reason"]["message"] == "metadata must be an object when provided"


def test_taskcreate_accepts_dict_lkb_metadata(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed ``metadata.lkb`` object still flows through to the Store."""
    _enable_plan_graph(monkeypatch)
    _install_repo(monkeypatch, tmp_home, "adapt-md-ok")
    ctx = _ctx()
    handled, result = try_handle(
        "TaskCreate",
        {"subject": "A", "metadata": {"lkb": {"assertions": ["a1"]}, "team": "core"}},
        ctx,
    )
    assert handled is True
    assert result.is_error is False
    new_id = result.output["task"]["id"]
    assert list(ctx.tasks[new_id]["metadata"]["lkb"]["assertions"]) == ["a1"]
    assert ctx.tasks[new_id]["metadata"]["team"] == "core"
