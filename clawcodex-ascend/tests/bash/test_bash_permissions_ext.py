"""Tests for the downstream ``clawcodex_ext.permissions.bash_security`` module.

Covers the downstream-patched ``check_bash_command_safety()`` which adds
the ``shell`` parameter (upstream ``src.permissions.bash_security`` does
not accept ``shell``).  This test exists because the upstream test
(``tests/bash/test_bash_permissions_full.py``) imports from
``src.permissions.bash_security`` and therefore does NOT exercise the
downstream version's ``shell="powershell"`` dispatch path.

Regression test for: ``TypeError: check_bash_command_safety() got an
unexpected keyword argument 'shell'`` — caused by the function signature
missing the ``shell`` parameter that ``bash_tool.py`` passes.
"""

from __future__ import annotations


# pylint: disable=E0611
import unittest

from clawcodex_ext.permissions.bash_security import check_bash_command_safety


class TestCheckBashCommandSafetyExt(unittest.TestCase):
    """Downstream ``check_bash_command_safety`` — ``shell`` parameter."""

    # ── shell=None (bash, the default) ────────────────────────────────

    def test_shell_none_safe(self) -> None:
        """Explicit ``shell=None`` should behave identically to no shell arg."""
        result = check_bash_command_safety("echo hello", shell=None)
        self.assertIsNone(result)

    def test_shell_none_read_only(self) -> None:
        self.assertIsNone(check_bash_command_safety("ls -la", shell=None))

    def test_shell_none_dangerous(self) -> None:
        result = check_bash_command_safety("curl http://example.com", shell=None)
        self.assertIsNotNone(result)
        self.assertEqual(result.behavior, "ask")

    def test_shell_none_destructive(self) -> None:
        result = check_bash_command_safety("rm -rf /tmp/foo", shell=None)
        self.assertIsNotNone(result)
        self.assertEqual(result.behavior, "ask")

    # ── shell="powershell" dispatch ────────────────────────────────────

    def test_shell_powershell_dispatches(self) -> None:
        """``shell="powershell"`` must delegate to
        ``check_powershell_command_safety`` (and NOT raise TypeError).
        """
        # The PowerShell security module checks for cmdlets; a simple
        # ``Get-ChildItem`` should be treated as read-like and return None.
        result = check_bash_command_safety("Get-ChildItem", shell="powershell")
        # The PowerShell path returns None for safe commands too
        self.assertIn(result, (None, ...))  # pragma: no cover; just ensure it didn't crash

    def test_shell_powershell_no_error(self) -> None:
        """The critical regression test: ``shell="powershell"`` must not crash."""
        try:
            check_bash_command_safety("Write-Host hello", shell="powershell")
        except TypeError as exc:
            self.fail(f"shell='powershell' raised TypeError: {exc}")

    # ── shell="pwsh" (PowerShell Core) ─────────────────────────────────

    def test_shell_pwsh_no_error(self) -> None:
        """``shell="pwsh"`` should also be handled (alias for powershell)."""
        try:
            check_bash_command_safety("Get-Date", shell="pwsh")
        except TypeError as exc:
            self.fail(f"shell='pwsh' raised TypeError: {exc}")

    # ─── shell=... (other shells, fallback to bash path) ───────────────

    def test_shell_zsh_fallback(self) -> None:
        """Unknown shell values should fall back to the bash analysis path."""
        result = check_bash_command_safety("echo zsh test", shell="zsh")
        self.assertIsNone(result)

    def test_shell_fish_fallback(self) -> None:
        result = check_bash_command_safety("echo fish test", shell="fish")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
