"""Drive a real orchestrator run + visualizer tick into one ``.cast``.

This module is the implementation behind ``clawcodex record --auto``. It
spins up a stub :class:`Orchestrator` against ``LocalTrackerAdapter`` +
bare origin, then
runs a batch of issues through ``GitSyncService.sync`` while emitting
two streams into one shared :class:`AsciicastWriter`:

* **orchestrator markers** — every per-session
  :class:`AsciicastSink` attached via
  ``Orchestrator.__init__(asciicast_capture=writer.capture)`` translates
  ``PhaseComplete`` / ``SessionComplete`` events into ``[phase N/T]``
  and ``session:<reason>`` markers.
* **dashboard tick snapshots** — every ``frame_delay_s`` seconds the
  script pulls the registered :class:`DashboardSource` instances (a
  real :class:`OrchestratorDashboardSource` was wired in at start-up)
  and feeds the merged entries into
  :meth:`AsciicastDashboardSource.record_snapshot`, which renders the
  ASCII panel frames via :func:`extensions.recording.renderers.panel`.

The :class:`AsciicastWriter` owns the lock that serialises both streams,
so order is well-defined and ``validate_cast`` is the single
acceptance check on the output file.

Design choices and their rationales
-----------------------------------
* **No code change to** ``AsciicastDashboardSource.pull`` — ``pull()``
  remains the recording-only stub it has been since before extraction. The
  auto-demo path calls ``record_snapshot(capture, entries, title=...)``
  directly with whatever entries the registry produced. Wiring
  ``pull()`` to the registry is left to a separate PR to keep the diff
  inside ``extensions/recording/`` and avoid crossing the protocol
  boundary into ``extensions/visualizer/``.
* **Manual** ``composite.on_phase_complete(...)`` **rather than real
  ``Orchestrator._run_issue``** — ``_run_issue`` pulls the model
  query engine / prompt / sandbox which is heavyweight and
  non-deterministic. ``GitSyncService.sync`` is the deterministic
  surface the verification report E2E suite already exercises; the demo's purpose is
  to show the recorder handles "an orchestrator-style batch", not to
  verify model behaviour.
* **Process-global registry cleanup** — the dashboard registry is
  shared across the process; this module registers an
  ``OrchestratorDashboardSource`` at start and *always* unregisters
  it in ``finally`` so test order doesn't pollute neighbours.
* **Deadline-based termination** — the main loop checks a
  monotonic deadline each tick rather than racing ``wait_for`` against
  the sync tasks. Cancelling a mid-flight sync is undesirable; we let
  the in-flight sync finish naturally and break out as soon as the
  panel rendering for that sync has been recorded.

The module is intentionally runnable both as
``python -m extensions.recording.auto_demo`` (smoke) and as a library
function (``run(...)`` is the public entry point used by the CLI
subcommand and the test suite).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from extensions.agent_dashboard.sources.orchestrator_source import (
    OrchestratorDashboardSource,
)
from extensions.agent_dashboard.source_registry import (
    get_default_registry,
    unregister_dashboard_source,
)
from extensions.api.query import PhaseComplete, SessionComplete
from extensions.capabilities.dashboard_entry import (
    DASHBOARD_STATUS_COMPLETED,
    DASHBOARD_STATUS_FAILED,
    DASHBOARD_STATUS_IN_PROGRESS,
    DASHBOARD_STATUS_PENDING,
    DashboardEntry,
)
from extensions.capabilities.recorder import AsciicastCapture, AsciicastHeader
from extensions.orchestrator.config.schema import (
    AgentConfig,
    HooksConfig,
    WorkflowConfig,
)
from extensions.orchestrator.git_sync import GitSyncService
from extensions.orchestrator.issue import Issue
from extensions.orchestrator.local_tracker.adapter import LocalTrackerAdapter
from extensions.orchestrator.orchestrator import Orchestrator
from extensions.orchestrator.workspace import (
    WorkspaceConfig,
    WorkspaceManager,
)
from extensions.recording.asciicast_writer import AsciicastWriter
from extensions.recording.validate_cast import validate_cast
from extensions.recording.visualizer_dashboard_source import (
    AsciicastDashboardSource,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DASHBOARD_PULL_SOURCES",
    "build_parser",
    "issues_for_auto_demo",
    "main",
    "run",
]


# ---------------------------------------------------------------------------
# Demo scaffolding — local tracker + bare origin
# ---------------------------------------------------------------------------


class _AutoDemoSession:
    """Minimal production session consumed by :class:`GitSyncService`.

    The recording command is shipped without the repository test suite, so
    its runtime scaffolding must remain self-contained.
    """

    def __init__(
        self,
        issue: Issue,
        workspace: Any,
        run_id: str,
        summary_comment_id: str,
    ) -> None:
        self.issue = issue
        self.workspace = workspace
        self.status = "completed"
        self.run_id = run_id
        self.summary_comment_id = summary_comment_id
        self.turn_count = 1
        self.tool_count = 1
        self.verification_status = None
        self.verification_output = None
        self.output_text = "agent output excerpt"


def _git(args: list[str], cwd: Path) -> None:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git executable is required for the automatic recording demo")
    subprocess.run(
        [git_executable, *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def _build_origin(base: Path) -> Path:
    origin = base / "origin.git"
    seed = base / "seed"
    seed.mkdir(parents=True)
    _git(["init", "--bare", str(origin)], base)
    _git(["init"], seed)
    _git(["config", "user.email", "demo@example.com"], seed)
    _git(["config", "user.name", "Auto Demo"], seed)
    (seed / "README.md").write_text("main branch\n", encoding="utf-8")
    _git(["add", "README.md"], seed)
    _git(["commit", "-m", "initial"], seed)
    _git(["branch", "-M", "main"], seed)
    _git(["remote", "add", "origin", str(origin)], seed)
    _git(["push", "-u", "origin", "main"], seed)
    _git(["symbolic-ref", "HEAD", "refs/heads/main"], origin)
    return origin


def _write_issue(issues_dir: Path, issue_id: str, identifier: str, title: str) -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    path = issues_dir / f"{identifier}.md"
    path.write_text(
        "---\n"
        f"id: {issue_id}\n"
        f"identifier: {identifier}\n"
        f"state: open\n"
        f"labels: [agent:run]\n"
        "---\n"
        f"# {title}\n\n"
        "Add a small verifiable change to the repository.\n",
        encoding="utf-8",
    )
    return path


def issues_for_auto_demo() -> list[dict[str, str]]:
    """Three issues with varying outcomes so the demo exercises several paths.

    Issue 0: success path (``test_command='true'``, no hooks).
    Issue 1: verification failure path (``test_command='false'``).
    Issue 2: pre-commit hook that adds a file — exercises the auto-amend
             branch in ``git_sync`` and surfaces the workspace change
             in the dashboard panel.
    """
    return [
        {
            "id": "auto-1",
            "identifier": "AUTO-1",
            "title": "Smoke run with passing verification",
        },
        {
            "id": "auto-2",
            "identifier": "AUTO-2",
            "title": "Trigger verification_failed path",
        },
        {
            "id": "auto-3",
            "identifier": "AUTO-3",
            "title": "Pre-commit hook adds a formatted.txt file",
        },
    ]


def _agent_hook_configs_for(issue_index: int) -> tuple[AgentConfig, HooksConfig]:
    """Pick configs that drive distinct dashboard states for issue N."""
    if issue_index == 0:
        return (
            AgentConfig(test_command="true", build_command="", lint_command=""),
            HooksConfig(pre_commit="", pre_push="", post_sync=""),
        )
    if issue_index == 1:
        return (
            AgentConfig(test_command="false", build_command="", lint_command=""),
            HooksConfig(),
        )
    # issue_index == 2
    formatter = (
        f"{sys.executable} -c "
        '"from pathlib import Path; '
        "Path(\\'formatted.txt\\').write_text(\\'formatted by pre_commit hook\\\\n\\')\""
    )
    return (
        AgentConfig(test_command="true"),
        HooksConfig(pre_commit=formatter, pre_push="", post_sync=""),
    )


# ---------------------------------------------------------------------------
# Issue → DashboardEntry projection (so the panel shows what was attempted)
# ---------------------------------------------------------------------------


def _issue_to_entry(
    issue_id: str,
    identifier: str,
    title: str,
    status: str,
    detail: str = "",
    progress_pct: float | None = None,
) -> DashboardEntry:
    return DashboardEntry(
        id=f"local:{issue_id}",
        source="local",
        title=identifier,
        status=status,
        detail=detail,
        progress_pct=progress_pct,
        owner="auto_demo",
    )


def _seed_entries(issues: list[dict[str, str]]) -> list[DashboardEntry]:
    """Initial snapshot — all three issues in pending."""
    return [
        _issue_to_entry(
            i["id"],
            i["identifier"],
            i["title"],
            status=DASHBOARD_STATUS_PENDING,
            progress_pct=0.0,
        )
        for i in issues
    ]


def _update_entry(
    entries: list[DashboardEntry],
    issue_id: str,
    *,
    status: str | None = None,
    progress_pct: float | None = None,
    detail: str | None = None,
) -> list[DashboardEntry]:
    """Mutate-by-replace the entry whose ``id`` ends with ``issue_id``."""
    for idx, entry in enumerate(entries):
        if entry.id.endswith(issue_id):
            updated = entry
            if status is not None:
                updated = replace(updated, status=status)
            if progress_pct is not None:
                updated = replace(updated, progress_pct=progress_pct)
            if detail is not None:
                updated = replace(updated, detail=detail)
            entries[idx] = updated
            return entries
    return entries


# Sources whose pull() is non-empty (orchestrator has a live provider-based
# source; the others may not be wired in this minimal scaffold). The auto
# demo also *concatenates* the local synthetic entries so the panel
# never goes blank even before OrchestratorDashboardSource is ready.
DASHBOARD_PULL_SOURCES = ("orchestrator",)


def pull_dashboard_entries(
    registry: Any,
    fallback: list[DashboardEntry],
) -> list[DashboardEntry]:
    """Aggregate entries from every registered pull-capable source.

    Sources whose ``pull`` raises are skipped — the dashboard would
    otherwise blank during the brief instant the orchestrator is in
    the middle of registering its registry. Empty snapshots are also
    skipped so the fallback synthetic entries remain visible.
    """
    aggregated: list[DashboardEntry] = list(fallback)
    for source in registry:
        name = getattr(source, "source_name", "")
        if name not in DASHBOARD_PULL_SOURCES:
            continue
        try:
            pulled = source.pull()
        except Exception as exc:  # noqa: BLE001
            logger.debug("dashboard source %s pull failed: %s", name, exc)
            continue
        if pulled:
            aggregated.extend(pulled)
    return aggregated


# ---------------------------------------------------------------------------
# Tick loop — render one panel per N seconds
# ---------------------------------------------------------------------------


async def _tick_loop(
    capture: AsciicastCapture,
    registry: Any,
    get_entries: Any,
    stop_event: asyncio.Event,
    frame_delay_s: float,
    make_title: Any,
) -> None:
    """1Hz-ish tick that renders a dashboard snapshot per loop iteration.

    Loops until ``stop_event`` is set. Calls ``get_entries()`` (a
    callable returning the *current* entries list) and ``make_title()``
    so the panel shows the latest progress — including between-sync
    intermediate states.
    """
    source = AsciicastDashboardSource()
    tick = 0
    while not stop_event.is_set():
        entries = pull_dashboard_entries(registry, get_entries())
        source.record_snapshot(
            capture,
            entries,
            title=make_title(tick),
        )
        tick += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=frame_delay_s)
        except asyncio.TimeoutError:
            # Not an error — just means the wait_for timed out without
            # the stop event being set; continue the loop.
            continue


# ---------------------------------------------------------------------------
# Public API — used by the CLI subcommand and the test suite
# ---------------------------------------------------------------------------


async def run(
    out_path: Path,
    *,
    duration_s: float = 30.0,
    issue_count: int = 3,
    frame_delay_s: float = 1.0,
) -> int:
    """Drive an end-to-end auto demo and return the process exit code.

    Parameters
    ----------
    out_path
        Destination ``.cast`` path; parents are created on demand.
    duration_s
        Wall-clock budget for the entire run. In-flight syncs are
        allowed to finish naturally so the final panel reflects them.
    issue_count
        Number of issues to dispatch (truncated to
        ``len(issues_for_auto_demo())`` — currently 3).
    frame_delay_s
        Sleep between visualizer ticks. 1 Hz is the recommended default
        for 30 s demos.
    """
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Defensive validation — the CLI already enforces these via argparse
    # types, but ``run()`` is a public library entry point and must not
    # accept silent no-op or busy-loop inputs (negative delays, NaN
    # timeouts, non-positive issue counts).
    import math

    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError(f"duration_s must be a non-negative finite number, got {duration_s!r}")
    if not math.isfinite(frame_delay_s) or frame_delay_s <= 0:
        raise ValueError(f"frame_delay_s must be a positive finite number, got {frame_delay_s!r}")
    if int(issue_count) < 1:
        raise ValueError(f"issue_count must be >= 1, got {issue_count!r}")

    issues = issues_for_auto_demo()[: max(1, int(issue_count))]
    # Use a uniquely-named run dir so multiple invocations against the
    # same --out path don't collide on the seed/origin filesystem.
    run_dir = Path(tempfile.mkdtemp(prefix=f".auto-demo-{out_path.stem}-"))

    base = run_dir
    origin = _build_origin(base)
    issues_dir = base / "issues"
    for issue in issues:
        _write_issue(issues_dir, issue["id"], issue["identifier"], issue["title"])

    tracker = LocalTrackerAdapter(issues_path=issues_dir)
    workspace_root = base / "workspaces"
    manager = WorkspaceManager(
        WorkspaceConfig(
            root=workspace_root,
            repo_clone_url=str(origin),
            checkout_issue_branch=True,
        )
    )

    header = AsciicastHeader(
        width=120,
        height=36,
        timestamp=int(time.time()),
        command="clawcodex record --auto",
        title=f"Auto demo — {len(issues)} issue(s)",
    )
    writer = AsciicastWriter(out_path, header)
    writer.open()
    capture = writer.capture

    # OrchestratorDashboardSource is provider-based; we pass a closure
    # that returns the live orchestrator once it is constructed below.
    orch_holder: dict[str, Orchestrator | None] = {"orch": None}

    orch_source = OrchestratorDashboardSource(orchestrator_provider=lambda: orch_holder["orch"])
    registry = get_default_registry()
    registry.register(orch_source)

    try:
        synth_entries = _seed_entries(issues)

        # Build a minimal orchestrator with a capture handle. The actual
        # sync work happens via GitSyncService directly — we don't run
        # the LLM agent runner here.
        orch = Orchestrator.__new__(Orchestrator)
        orch.workflow = WorkflowConfig.from_dict(
            {
                "tracker": {
                    "kind": "local",
                    "issues_path": str(issues_dir),
                },
                "agent": {
                    "max_concurrent_agents": 1,
                    "phases": ["analysis", "design", "impl", "test", "review"],
                },
            }
        )
        orch._progress_context = _make_progress_context()
        orch.asciicast_capture = capture
        orch_holder["orch"] = orch

        async def sync_one(issue_spec: dict[str, str], index: int) -> str:
            """Run GitSyncService.sync for one issue. Returns a status word."""
            agent_cfg, hook_cfg = _agent_hook_configs_for(index)
            issue_obj = Issue(
                id=issue_spec["id"],
                identifier=issue_spec["identifier"],
                title=issue_spec["title"],
                url=f"file://{issues_dir / (issue_spec['identifier'] + '.md')}",
            )
            workspace = await manager.create_for_issue(issue_obj)
            placeholder = await tracker.create_comment(
                issue_spec["id"],
                "## ClawCodex Run Summary\n\n⏳ Run in progress.",
            )
            assert placeholder is not None

            # Drive per-issue orchestrator sinks so the .cast has phase
            # + session markers.
            composite = orch._build_session_sink(issue_spec["id"])
            composite.on_phase_complete(
                PhaseComplete(phase=1, turn_count=1),
                None,
            )

            # Edit the workspace so git has something to commit.
            (workspace.path / "README.md").write_text(f"changes for {issue_spec['identifier']}\n", encoding="utf-8")

            try:
                service = GitSyncService(
                    tracker,
                    agent_config=agent_cfg,
                    hooks_config=hook_cfg,
                )
                run_id = f"auto-{issue_spec['identifier']}-{int(time.time())}"
                session = _AutoDemoSession(issue_obj, workspace, run_id, placeholder.id)
                # The .cast-space session_complete reason depends on
                # whether sync raised or returned cleanly.
                reason = "exit_code=0"
                try:
                    await service.sync(session)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "GitSyncService.sync raised for %s: %s",
                        issue_spec["id"],
                        exc,
                    )
                    reason = f"sync_error={exc.__class__.__name__}"
                composite.on_session_complete(
                    SessionComplete(reason=reason),
                    None,
                )
                return reason
            finally:
                # Workspace cleanup: WorkspaceManager does not expose a
                # remove_for_issue method, and the demo uses a fresh
                # ``run_dir`` per ``run()`` invocation (via
                # ``tempfile.mkdtemp``) so workspaces do not need to be
                # torn down between issues. Leaving the workspace in
                # place lets the post-sync dashboard panel reflect the
                # final state of each branch for visual inspection.
                pass

        issue_count_label = len(issues)

        def _make_title(tick: int) -> str:
            return f"Logical Kanban (tick {tick} of {issue_count_label})"

        async def drive() -> list[str]:
            stop_event = asyncio.Event()
            tick_task = asyncio.create_task(
                _tick_loop(
                    capture,
                    registry,
                    lambda: synth_entries,
                    stop_event,
                    frame_delay_s,
                    make_title=_make_title,
                )
            )

            async def with_progress(idx: int, spec: dict[str, str]) -> str:
                _update_entry(synth_entries, spec["id"], status=DASHBOARD_STATUS_IN_PROGRESS)
                try:
                    result = await sync_one(spec, idx)
                except Exception as exc:  # noqa: BLE001
                    import traceback

                    logger.debug(
                        "with_progress sync exception for %s:\n%s",
                        spec["id"],
                        "".join(traceback.format_exception(exc)),
                    )
                    _update_entry(
                        synth_entries,
                        spec["id"],
                        status=DASHBOARD_STATUS_FAILED,
                        detail=str(exc)[:80],
                    )
                    return f"sync_error={exc.__class__.__name__}"
                status = DASHBOARD_STATUS_COMPLETED if result == "exit_code=0" else DASHBOARD_STATUS_FAILED
                _update_entry(
                    synth_entries,
                    spec["id"],
                    status=status,
                    progress_pct=1.0 if status == DASHBOARD_STATUS_COMPLETED else None,
                    detail=result,
                )
                return result

            sync_tasks = [asyncio.create_task(with_progress(i, spec)) for i, spec in enumerate(issues)]

            deadline_loop = asyncio.create_task(asyncio.sleep(max(0.0, float(duration_s))))
            done_deadline, _ = await asyncio.wait(
                {deadline_loop},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if deadline_loop in done_deadline:
                stop_event.set()

            sync_results = await asyncio.gather(*sync_tasks, return_exceptions=True)
            await tick_task
            return [r if isinstance(r, str) else "sync_error" for r in sync_results]

        results = await drive()

    finally:
        try:
            unregister_dashboard_source("orchestrator")
        except Exception as exc:
            logger.debug("failed to unregister automatic demo dashboard source: %s", exc)
        writer.close()

    errors = validate_cast(out_path)
    if errors:
        print(f"[auto] validation FAILED for {out_path}:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"[auto] {out_path} — {writer.frame_count} frame(s), results={results}; validation: OK")
    return 0


def _make_progress_context() -> Any:
    """Stub :class:`ToolContext` for the inline orchestrator.

    The tool-context sink is not exercised in the demo path so a
    minimal placeholder is sufficient.
    """
    try:
        from src.tool_system.context import ToolContext

        return ToolContext(workspace_root=tempfile.gettempdir())
    except Exception:
        # Defensive — the auto demo must not fail on partial checkouts.
        return None


# ---------------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clawcodex record --auto",
        description=(
            "Record a real orchestrator batch + dashboard ticks into one "
            "asciicast v2 .cast. Mutually exclusive with --sources."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("auto-demo.cast"),
        help="Output .cast file path (default: %(default)s).",
    )
    p.add_argument(
        "--auto-duration-s",
        type=float,
        default=30.0,
        help="Total wall-clock budget in seconds (default: %(default)s).",
    )
    p.add_argument(
        "--auto-issue-count",
        type=int,
        default=3,
        help="Number of issues to dispatch, 1..3 (default: %(default)s).",
    )
    p.add_argument(
        "--auto-frame-delay-s",
        type=float,
        default=1.0,
        help="Wall-clock seconds between dashboard ticks (default: %(default)s).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(
        run(
            args.out,
            duration_s=args.auto_duration_s,
            issue_count=args.auto_issue_count,
            frame_delay_s=args.auto_frame_delay_s,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Helpers used by cli.py --auto integration
# ---------------------------------------------------------------------------


def supports_auto() -> bool:
    """Cheap capability probe — returns True when ever the demo can run."""
    # Importing this module already resolves the dashboard source used by
    # ``run``; reaching this probe therefore means the capability is present.
    return OrchestratorDashboardSource is not None
