#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Fine-grained tests for PowerShell support.

Covers:
- search_classification
- read_only_validation
- command_semantics
- powershell_security
- destructive_warnings PowerShell patterns
"""

from __future__ import annotations


# pylint: disable=E0611
import unittest

from clawcodex_ext.permissions.powershell_security import (
    analyze_powershell_safety,
    check_powershell_command_safety,
)
from clawcodex_ext.tool_system.tools.bash.command_semantics import (
    CommandInterpretation,
    interpret_command_result,
)
from clawcodex_ext.tool_system.tools.bash.destructive_warnings import (
    get_destructive_command_warning,
)
from clawcodex_ext.tool_system.tools.bash.read_only_validation import (
    is_command_read_only,
)
from clawcodex_ext.tool_system.tools.bash.search_classification import (
    SearchOrReadResult,
    is_search_or_read_command,
)


# ---------------------------------------------------------------------------
# Search / read / list classification
# ---------------------------------------------------------------------------


class TestPowerShellSearchClassification(unittest.TestCase):
    def test_select_string_is_search(self) -> None:
        result = is_search_or_read_command("Select-String 'foo' *.txt", shell="powershell")
        self.assertEqual(result, SearchOrReadResult(is_search=True))

    def test_sls_alias_is_search(self) -> None:
        result = is_search_or_read_command("sls 'foo' *.txt", shell="powershell")
        self.assertEqual(result, SearchOrReadResult(is_search=True))

    def test_where_object_is_search(self) -> None:
        result = is_search_or_read_command("Get-Process | Where-Object {$_.CPU -gt 10}", shell="powershell")
        self.assertTrue(result.is_search)
        self.assertFalse(result.is_list)

    def test_get_content_is_read(self) -> None:
        result = is_search_or_read_command("Get-Content file.txt", shell="powershell")
        self.assertEqual(result, SearchOrReadResult(is_read=True))

    def test_get_childitem_is_list(self) -> None:
        result = is_search_or_read_command("Get-ChildItem .", shell="powershell")
        self.assertEqual(result, SearchOrReadResult(is_list=True))

    def test_write_host_is_neutral(self) -> None:
        result = is_search_or_read_command("Write-Host hello", shell="powershell")
        self.assertEqual(result, SearchOrReadResult())

    def test_set_content_not_search_or_read(self) -> None:
        result = is_search_or_read_command("Set-Content file.txt 'x'", shell="powershell")
        self.assertEqual(result, SearchOrReadResult())

    def test_pipeline_all_search_or_read(self) -> None:
        result = is_search_or_read_command(
            "Get-Content file.txt | Select-String 'foo' | Measure-Object",
            shell="powershell",
        )
        self.assertTrue(result.is_read)
        self.assertTrue(result.is_search)


# ---------------------------------------------------------------------------
# Read-only validation
# ---------------------------------------------------------------------------


class TestPowerShellReadOnlyValidation(unittest.TestCase):
    def test_get_content_read_only(self) -> None:
        self.assertTrue(is_command_read_only("Get-Content file.txt", shell="powershell"))

    def test_select_string_read_only(self) -> None:
        self.assertTrue(is_command_read_only("Select-String foo *.txt", shell="powershell"))

    def test_set_content_not_read_only(self) -> None:
        self.assertFalse(is_command_read_only("Set-Content file.txt 'x'", shell="powershell"))

    def test_pipeline_read_only(self) -> None:
        self.assertTrue(
            is_command_read_only(
                "Get-Content file.txt | Select-String foo | Measure-Object",
                shell="powershell",
            )
        )

    def test_pipeline_with_write_not_read_only(self) -> None:
        self.assertFalse(
            is_command_read_only(
                "Get-Content file.txt | Set-Content out.txt",
                shell="powershell",
            )
        )

    def test_shell_metachar_rejected(self) -> None:
        self.assertFalse(is_command_read_only("Get-Content file.txt; rm x", shell="powershell"))


# ---------------------------------------------------------------------------
# Command semantics / exit-code interpretation
# ---------------------------------------------------------------------------


class TestPowerShellCommandSemantics(unittest.TestCase):
    def test_select_string_no_match_exit_1_not_error(self) -> None:
        result = interpret_command_result("Select-String foo *.txt", 1, "", "", shell="powershell")
        self.assertEqual(result, CommandInterpretation(is_error=False, message="No matches found"))

    def test_select_string_exit_2_is_error(self) -> None:
        result = interpret_command_result("Select-String foo *.txt", 2, "", "bad", shell="powershell")
        self.assertTrue(result.is_error)

    def test_sls_alias_no_match_exit_1_not_error(self) -> None:
        result = interpret_command_result("sls foo *.txt", 1, "", "", shell="powershell")
        self.assertEqual(result, CommandInterpretation(is_error=False, message="No matches found"))

    def test_generic_cmdlet_non_zero_is_error(self) -> None:
        result = interpret_command_result("Get-Content missing.txt", 1, "", "not found", shell="powershell")
        self.assertTrue(result.is_error)
        self.assertIn("exit code 1", result.message or "")


# ---------------------------------------------------------------------------
# PowerShell safety analysis
# ---------------------------------------------------------------------------


class TestPowerShellSafetyAnalysis(unittest.TestCase):
    def test_get_content_is_read_only(self) -> None:
        result = analyze_powershell_safety("Get-Content file.txt")
        self.assertEqual(result.safety, "read_only")

    def test_write_host_is_safe(self) -> None:
        result = analyze_powershell_safety("Write-Host hello")
        self.assertEqual(result.safety, "safe")

    def test_set_content_is_write(self) -> None:
        result = analyze_powershell_safety("Set-Content file.txt 'x'")
        self.assertEqual(result.safety, "write")

    def test_remove_item_force_is_destructive(self) -> None:
        result = analyze_powershell_safety("Remove-Item file.txt -Force")
        self.assertEqual(result.safety, "destructive")

    def test_remove_item_recurse_force_is_destructive(self) -> None:
        result = analyze_powershell_safety("Remove-Item -Recurse -Force ./dir")
        self.assertEqual(result.safety, "destructive")

    def test_invoke_expression_is_dangerous(self) -> None:
        result = analyze_powershell_safety("Invoke-Expression 'foo'")
        self.assertEqual(result.safety, "dangerous")

    def test_iex_alias_is_dangerous(self) -> None:
        result = analyze_powershell_safety("iex 'foo'")
        self.assertEqual(result.safety, "dangerous")

    def test_clear_disk_is_write(self) -> None:
        result = analyze_powershell_safety("Clear-Disk 0")
        self.assertEqual(result.safety, "write")

    def test_format_volume_is_write(self) -> None:
        result = analyze_powershell_safety("Format-Volume -DriveLetter C")
        self.assertEqual(result.safety, "write")

    def test_unknown_command_is_unknown(self) -> None:
        result = analyze_powershell_safety("some-unknown-command")
        self.assertEqual(result.safety, "unknown")

    def test_pipeline_uses_most_severe(self) -> None:
        result = analyze_powershell_safety("Get-ChildItem | Remove-Item -Recurse -Force")
        self.assertEqual(result.safety, "destructive")


class TestPowerShellCommandSafetyCheck(unittest.TestCase):
    def test_dangerous_returns_ask(self) -> None:
        result = check_powershell_command_safety("Invoke-Expression 'rm -rf /'")
        assert result is not None
        self.assertEqual(result.behavior, "ask")
        self.assertIn("Dangerous", result.message)

    def test_destructive_returns_ask(self) -> None:
        result = check_powershell_command_safety("Remove-Item -Recurse -Force ./dir")
        assert result is not None
        self.assertEqual(result.behavior, "ask")
        self.assertIn("Destructive", result.message)

    def test_safe_returns_none(self) -> None:
        result = check_powershell_command_safety("Get-Content file.txt")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Destructive warnings
# ---------------------------------------------------------------------------


class TestPowerShellDestructiveWarnings(unittest.TestCase):
    def test_remove_item_recurse_force(self) -> None:
        warning = get_destructive_command_warning("Remove-Item -Recurse -Force ./dir")
        self.assertEqual(warning, "Note: may recursively force-remove files")

    def test_remove_item_force_recurse(self) -> None:
        warning = get_destructive_command_warning("Remove-Item -Force -Recurse ./dir")
        self.assertEqual(warning, "Note: may recursively force-remove files")

    def test_iex_warning(self) -> None:
        warning = get_destructive_command_warning("iex 'foo'")
        self.assertEqual(warning, "Note: may execute arbitrary code")

    def test_invoke_expression_warning(self) -> None:
        warning = get_destructive_command_warning("Invoke-Expression 'foo'")
        self.assertEqual(warning, "Note: may execute arbitrary code")

    def test_start_process_runas_warning(self) -> None:
        warning = get_destructive_command_warning("Start-Process notepad -Verb runas")
        self.assertEqual(warning, "Note: may run with elevated privileges")

    def test_clear_disk_warning(self) -> None:
        warning = get_destructive_command_warning("Clear-Disk 0")
        self.assertEqual(warning, "Note: may erase disk contents")

    def test_format_volume_warning(self) -> None:
        warning = get_destructive_command_warning("Format-Volume -DriveLetter C")
        self.assertEqual(warning, "Note: may format a volume")

    def test_set_executionpolicy_bypass_warning(self) -> None:
        warning = get_destructive_command_warning("Set-ExecutionPolicy -ExecutionPolicy Bypass")
        self.assertEqual(warning, "Note: may weaken script execution policy")

    def test_safe_command_no_warning(self) -> None:
        warning = get_destructive_command_warning("Get-Content file.txt")
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main()
