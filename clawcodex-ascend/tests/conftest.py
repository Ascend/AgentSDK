"""Project-wide pytest fixtures.

Provides two cross-test isolation guarantees and one opt-in helper:

1. ``src`` submodule re-binding — ``_ensure_src_submodules_loaded`` keeps
   ``src.config`` / ``src.permissions`` / ``src.permissions.modes`` bound
   on the ``src`` module object so ``monkeypatch.setattr('src.config.X', ...)``
   and ``'src.permissions.modes.X', ...`` resolve even after a previous
   test pops ``src`` from ``sys.modules``.

2. In-memory keyring backend — ``_isolate_mcp_keyring`` swaps
   ``keyring.get_keyring()`` and the module-level free functions to a
   per-test fake so MCP token-storage tests cannot leak entries into
   the developer's real OS keychain (macOS Keychain / Linux
   secret-service / Windows DPAPI).

3. ``isolated_tmp_repo`` (opt-in) — initializes a deterministic git repo
   on ``tmp_path`` for orchestrator snapshot tests and stability-gate
   transcript tests.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


# Submodules that tests reach via ``monkeypatch.setattr('src.<name>.X', ...)``.
# Extend this tuple when adding new dotted-path patch sites; the autouse
# ``_ensure_src_submodules_loaded`` fixture will pick them up automatically.
# Order: parents before children (so ``src.permissions`` binds before
# ``setattr(src.permissions, 'modes', ...)`` resolves).
_PATCHED_SUBMODULES: tuple[str, ...] = (
    "src.config",
    "src.permissions",
    "src.permissions.modes",
)


@pytest.fixture(autouse=True)
def _ensure_src_submodules_loaded():
    """Pre-import patched-path submodules and force-bind them on the
    current ``src`` module object so ``monkeypatch.setattr`` resolves.

    Background: pytest's ``monkeypatch.setattr`` walks a dotted path
    via successive ``getattr`` calls. ``getattr(src, 'config')`` only
    succeeds if ``src.config`` is bound as an attribute of the ``src``
    package. ``src/__init__.py`` does ``from .config import ...`` which
    normally binds it, but earlier tests in the same pytest session can
    wipe the binding in two distinct ways:

    1. Aggressive ``monkeypatch.setattr`` / ``del sys.modules['src.X']``
       chains that remove the submodule.
    2. ``sys.modules.pop('src', None)`` — seen in
       ``tests/test_downstream_cli_dispatch.py::test_run_cli_version_short_circuit``.
       After this, a plain ``import src.permissions`` does NOT bind
       ``permissions`` onto the freshly-imported ``src`` module because
       Python's import machinery sees the submodule already cached in
       ``sys.modules`` and skips the parent-binding step. The attribute
       then does not exist on the new ``src`` module, and any subsequent
       ``monkeypatch.setattr('src.permissions.X', ...)`` fails with
       ``AttributeError: module 'src' has no attribute 'permissions'``.

    The same applies one level deeper: ``monkeypatch.setattr(
    'src.permissions.modes.X', ...)`` reaches ``src.permissions`` via
    the attribute the fixture just re-bound, then triggers
    ``__getattr__('modes')`` on that module — which only succeeds if
    ``src.permissions.modes`` is already populated in ``sys.modules``.
    Pre-importing the chain is the only robust fix.

    The robust shape is data-driven via ``_PATCHED_SUBMODULES``: import
    each entry, then ``setattr(parent_module, child_name, mod)`` on
    whichever module object is currently in ``sys.modules``. This is
    cheap (one dict lookup + one attribute write per entry per test)
    and is robust against any test-pollution pattern.

    Residual effect (NOT auto-reverted): the bindings persist across the
    session. This is intentional — pytest's module cache means tests
    cannot observe the prior unbound state anyway, and every subsequent
    test that uses ``monkeypatch.setattr`` with these dotted paths
    needs them. If a future test needs to verify the *unbound* state,
    it must explicitly
    ``monkeypatch.delattr(src, 'config', raising=False)``.
    """
    for dotted in _PATCHED_SUBMODULES:
        mod = importlib.import_module(dotted)
        parent_name, _, child_name = dotted.rpartition(".")
        setattr(sys.modules[parent_name], child_name, mod)
    yield


class _InMemoryKeyringBackend:
    """Minimal ``keyring.backend.KeyringBackend`` that holds tokens in a
    process-local dict. Used by ``_isolate_mcp_keyring`` to isolate
    token storage from the real macOS Keychain / Linux secret-service
    / Windows DPAPI.

    Implements only ``get_password`` / ``set_password`` / ``delete_password``
    plus ``priority`` — sufficient for ``src/.../mcp_token_storage`` as of
    2026-07. Extend with ``get_credential`` / ``set_credential`` if the
    storage layer migrates to keyring ≥ 0.16's Credential-based API.
    """

    priority = 1.0  # higher than FailKeyring

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self._store[(service, username)]
        except KeyError:
            # Lazy import so the class loads even when keyring is absent.
            from keyring.errors import PasswordDeleteError  # type: ignore

            raise PasswordDeleteError(f"No such entry: {service}/{username}")


def _git(*args: str, cwd: Path) -> None:
    """Run a git subcommand in ``cwd`` and surface stderr on failure.

    Plain ``subprocess.run(..., check=True, capture_output=True)`` swallows
    stderr into ``CalledProcessError.stderr``, which is rarely read by
    callers — making CI failures cryptic. This helper re-raises with the
    captured stderr/stdout inlined so the diagnostic lands in the
    pytest traceback.
    """
    try:
        subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd} (rc={e.returncode}):\n"
            f"  stderr: {e.stderr.strip() or '(empty)'}\n"
            f"  stdout: {e.stdout.strip() or '(empty)'}"
        ) from e


@pytest.fixture
def isolated_tmp_repo(tmp_path):
    """Initialize a minimal git repo on ``tmp_path`` with deterministic identity.

    Sets ``user.email=test@test`` and ``user.name=Test`` (local scope only)
    so commits are reproducible. Runs ``git init`` with ``-b main`` so the
    default branch is named ``main`` (matching GitHub/GitCode convention).
    Returns ``tmp_path`` ready to commit into.

    Requires Git ≥ 2.28 (released 2020-07-27) for ``-b`` initial-branch
    support. Older Git falls back to the build's default branch name
    (``master`` on most builds).

    Project-wide so both ``tests/stability_gate/`` (future transcript
    tests) and ``tests/orchestrator/`` (P0-3 ``_status_snapshot``
    snapshot tests) can reuse the same minimal-repo boilerplate.
    """
    _git("init", "-b", "main", cwd=tmp_path)
    _git("config", "user.email", "test@test", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _isolate_mcp_keyring(request, monkeypatch):
    """Swap ``keyring.get_keyring()`` and the module-level free functions
    to a per-test in-memory backend so MCP token-storage tests don't leak
    into the real OS keychain.

    Autouse — applies to every test, not just MCP token-storage. Reasons:

    1. **Defense in depth**: any test that incidentally touches keyring
       (including a misconfigured import chain) cannot leak tokens to
       the developer's real keychain.
    2. **Cheap**: zero I/O, zero external state; ``monkeypatch`` reverts
       the patch at teardown.
    3. **Safe opt-out**: tests that explicitly want the real keyring can
       mark themselves with ``@pytest.mark.real_keyring``. The marker is
       currently unused; kept as a forward-compat escape hatch for
       future integration tests that intentionally exercise real
       keyring backends (e.g., on a CI runner with a configured
       secret-service stub).
    """
    if "real_keyring" in request.keywords:
        yield
        return
    try:
        import keyring
    except ImportError:
        yield
        return
    fake = _InMemoryKeyringBackend()
    monkeypatch.setattr(keyring, "get_keyring", lambda: fake)
    monkeypatch.setattr(keyring, "get_password", fake.get_password)
    monkeypatch.setattr(keyring, "set_password", fake.set_password)
    monkeypatch.setattr(keyring, "delete_password", fake.delete_password)
    yield
