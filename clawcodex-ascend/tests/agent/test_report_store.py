"""Unit tests for :mod:`src.agent.report_store` (F-88 P88-D).

Covers the on-disk report store: dual MD+JSON write, atomic
write primitive, session-scoped paths, and the
``### Critical Files`` parser.
"""

from __future__ import annotations


# pylint: disable=C1803,E0611
import json
from pathlib import Path

import pytest

from src.agent.report_store import (
    ExploreReport,
    PlanDocument,
    ReportStore,
    now_iso_utc,
    parse_critical_files,
)
from clawcodex_ext.agent.report_store import _atomic_write_json


# ---------------------------------------------------------------------------
# parse_critical_files — pure function
# ---------------------------------------------------------------------------


def test_parse_critical_files_missing_section() -> None:
    assert parse_critical_files("") == ()
    assert parse_critical_files("Just some prose, no section.") == ()


def test_parse_critical_files_basic() -> None:
    md = (
        "# Plan\n"
        "Some prose here.\n\n"
        "### Critical Files for Implementation\n"
        "- src/foo.py\n"
        "- src/bar/baz.py\n"
        "- tests/test_foo.py\n"
    )
    files = parse_critical_files(md)
    assert files == ("src/foo.py", "src/bar/baz.py", "tests/test_foo.py")


def test_parse_critical_files_alternate_heading() -> None:
    """The parser is case-insensitive on the heading and tolerant of
    different heading levels (``#`` / ``##`` / ``###``).
    """
    md = "## critical files\n- alpha.py\n- beta.py\n"
    assert parse_critical_files(md) == ("alpha.py", "beta.py")


def test_parse_critical_files_strips_inline_backticks() -> None:
    md = "### Critical Files\n- `src/x.py`\n- `src/y.py`\n"
    assert parse_critical_files(md) == ("src/x.py", "src/y.py")


def test_parse_critical_files_skips_blank_bullets() -> None:
    md = "### Critical Files\n- a.py\n- \n- b.py\n"
    assert parse_critical_files(md) == ("a.py", "b.py")


# ---------------------------------------------------------------------------
# Atomic write primitive
# ---------------------------------------------------------------------------


def test_atomic_write_json_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    _atomic_write_json(target, {"hello": "world"})
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"hello": "world"}


def test_atomic_write_json_no_torn_file_on_repeat(tmp_path: Path) -> None:
    """Two consecutive writes to the same path produce a single
    well-formed JSON file; the temp file is cleaned up.
    """
    target = tmp_path / "x.json"
    _atomic_write_json(target, {"v": 1})
    _atomic_write_json(target, {"v": 2})
    leftovers = list(tmp_path.glob(".x.json.*.tmp"))
    assert leftovers == []
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}


def test_atomic_write_json_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out.json"
    _atomic_write_json(target, {"ok": True})
    assert target.exists()


# ---------------------------------------------------------------------------
# ReportStore.save_explore / save_plan
# ---------------------------------------------------------------------------


def _make_explore_report(**overrides) -> ExploreReport:
    defaults = dict(
        agent_id="a-001",
        session_id="s-001",
        title="Demo Explore",
        summary="A demo explore report.",
        findings=("Found A", "Found B"),
        critical_files=("src/foo.py",),
        raw_markdown="# Demo\n- Found A\n- Found B",
        created_at="2026-06-21T10:31:00Z",
    )
    defaults.update(overrides)
    return ExploreReport(**defaults)


def _make_plan_document(**overrides) -> PlanDocument:
    defaults = dict(
        agent_id="p-001",
        session_id="s-002",
        title="Demo Plan",
        summary="A demo plan.",
        steps=("Step 1: gather requirements", "Step 2: design"),
        critical_files=("src/bar.py",),
        raw_markdown="# Plan\n1. Step 1\n2. Step 2",
        created_at="2026-06-21T10:32:00Z",
    )
    defaults.update(overrides)
    return PlanDocument(**defaults)


def test_save_explore_writes_md_and_json(tmp_path: Path) -> None:
    store = ReportStore(base_dir=tmp_path)
    path = store.save_explore(_make_explore_report())
    assert path.name == "a-001.md"
    assert path.exists()
    json_path = path.with_suffix(".json")
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["title"] == "Demo Explore"
    assert payload["findings"] == ["Found A", "Found B"]
    assert payload["critical_files"] == ["src/foo.py"]
    assert payload["kind"] == "explore"


def test_save_plan_writes_md_and_json(tmp_path: Path) -> None:
    store = ReportStore(base_dir=tmp_path)
    path = store.save_plan(_make_plan_document())
    assert path.name == "p-001.md"
    assert path.exists()
    json_path = path.with_suffix(".json")
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["title"] == "Demo Plan"
    assert payload["steps"] == ["Step 1: gather requirements", "Step 2: design"]
    assert payload["critical_files"] == ["src/bar.py"]
    assert payload["kind"] == "plan"


def test_session_scoped_paths(tmp_path: Path) -> None:
    """Different ``session_id`` values produce different directories
    under the same ``base_dir``.
    """
    store = ReportStore(base_dir=tmp_path)
    store.save_explore(_make_explore_report(session_id="sess-A"))
    store.save_explore(_make_explore_report(session_id="sess-B"))
    assert (tmp_path / "explore" / "sess-A" / "a-001.md").exists()
    assert (tmp_path / "explore" / "sess-B" / "a-001.md").exists()


def test_explore_and_plan_have_distinct_subdirs(tmp_path: Path) -> None:
    store = ReportStore(base_dir=tmp_path)
    store.save_explore(_make_explore_report())
    store.save_plan(_make_plan_document())
    assert (tmp_path / "explore").exists()
    assert (tmp_path / "plan").exists()


def test_markdown_render_contains_title_and_summary(tmp_path: Path) -> None:
    store = ReportStore(base_dir=tmp_path)
    path = store.save_explore(_make_explore_report())
    text = path.read_text(encoding="utf-8")
    assert "# Demo Explore" in text
    assert "A demo explore report." in text
    assert "Found A" in text
    assert "src/foo.py" in text


def test_plan_markdown_renders_numbered_steps(tmp_path: Path) -> None:
    store = ReportStore(base_dir=tmp_path)
    path = store.save_plan(_make_plan_document())
    text = path.read_text(encoding="utf-8")
    assert "1. Step 1: gather requirements" in text
    assert "2. Step 2: design" in text


def test_missing_critical_files_is_empty_list(tmp_path: Path) -> None:
    store = ReportStore(base_dir=tmp_path)
    report = _make_explore_report(critical_files=(), raw_markdown="no critical files section here")
    path = store.save_explore(report)
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["critical_files"] == []


def test_save_explore_is_thread_safe(tmp_path: Path) -> None:
    """Concurrent writes do not corrupt the JSON file. Smoke test —
    we are not validating ordering, just absence of torn writes.
    """
    import threading

    store = ReportStore(base_dir=tmp_path)
    errors: list[Exception] = []

    def _worker(i: int) -> None:
        try:
            store.save_explore(_make_explore_report(agent_id=f"a-{i:03d}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    # All 8 files exist.
    files = list((tmp_path / "explore" / "s-001").glob("*.json"))
    assert len(files) == 8


# ---------------------------------------------------------------------------
# now_iso_utc
# ---------------------------------------------------------------------------


def test_now_iso_utc_format() -> None:
    """The timestamp is a 20-char ISO-8601 string with a ``Z`` suffix."""
    ts = now_iso_utc()
    assert ts.endswith("Z")
    assert len(ts) == 20  # YYYY-MM-DDTHH:MM:SSZ
    # Round-trip via fromisoformat (drop the trailing Z).
    from datetime import datetime

    datetime.fromisoformat(ts.rstrip("Z"))


# ---------------------------------------------------------------------------
# Path resolution — CLAWCODEX_HOME override
# ---------------------------------------------------------------------------


def test_base_dir_uses_clawcodex_home_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    store = ReportStore()
    assert store.base_dir == tmp_path / "reports"


def test_explicit_base_dir_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``base_dir`` wins over ``CLAWCODEX_HOME``."""
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "ignored"))
    explicit = tmp_path / "explicit"
    store = ReportStore(base_dir=explicit)
    assert store.base_dir == explicit


# ---------------------------------------------------------------------------
# Sanity — frozen dataclasses
# ---------------------------------------------------------------------------


def test_explore_report_is_frozen() -> None:
    """``ExploreReport`` is a frozen dataclass."""
    import dataclasses

    report = _make_explore_report()
    assert dataclasses.is_dataclass(report)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.title = "mutated"  # type: ignore[misc]


def test_plan_document_is_frozen() -> None:
    """``PlanDocument`` is a frozen dataclass."""
    import dataclasses

    plan = _make_plan_document()
    assert dataclasses.is_dataclass(plan)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.title = "mutated"  # type: ignore[misc]
