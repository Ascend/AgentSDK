"""Tests for Auto Mode LLM Classifier and Danger Detector.

Tests the extended LLM-based classification system that works alongside
the existing rule-based auto_mode_classify. Covers:
- ClassificationCache: TTL expiry, max entries, cache hit/miss
- Danger detection: Bash/Write/Edit dangerous patterns
- LLM classifier: JSON parsing, fallback handling, provider integration
- Cycle validation: can_cycle_to_auto, protected directories, danger history
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from clawcodex_ext.permissions.classifier import (
    ClassificationCache,
    LLMClassificationResult,
    auto_mode_classify_with_llm,
    get_cache,
    llm_classify_tool_call,
)
from clawcodex_ext.permissions.cycle import (
    can_cycle_to_auto,
    get_auto_mode_availability_reason,
)
from clawcodex_ext.permissions.danger_detector import (
    detect_dangerous_bash_command,
    detect_dangerous_edit_path,
    detect_dangerous_tool_call,
    detect_dangerous_write_path,
)
from clawcodex_ext.permissions.types import ToolPermissionContext


class TestClassificationCache(unittest.TestCase):
    def test_cache_set_and_get(self) -> None:
        cache = ClassificationCache()
        result = LLMClassificationResult(
            decision="AUTO_ALLOW",
            reasoning="test",
            confidence=0.9,
            cache_key="key1",
        )
        cache.set("key1", result)
        retrieved = cache.get("key1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.decision, "AUTO_ALLOW")

    def test_cache_miss(self) -> None:
        cache = ClassificationCache()
        retrieved = cache.get("nonexistent")
        self.assertIsNone(retrieved)

    def test_cache_ttl_expiry(self) -> None:
        cache = ClassificationCache(ttl_seconds=0.1)
        result = LLMClassificationResult(
            decision="AUTO_ALLOW",
            reasoning="test",
            confidence=0.9,
            cache_key="key1",
        )
        cache.set("key1", result)

        import time

        time.sleep(0.15)

        retrieved = cache.get("key1")
        self.assertIsNone(retrieved)

    def test_cache_max_entries(self) -> None:
        cache = ClassificationCache(max_entries=2)
        for i in range(3):
            cache.set(
                f"key{i}",
                LLMClassificationResult(
                    decision="AUTO_ALLOW",
                    reasoning="test",
                    confidence=0.9,
                    cache_key=f"key{i}",
                ),
            )
        self.assertEqual(len(cache.entries), 2)
        self.assertIsNone(cache.get("key0"))


class TestDangerousBashCommand(unittest.TestCase):
    def test_rm_rf_detected(self) -> None:
        is_danger, reason = detect_dangerous_bash_command("rm -rf /tmp/test")
        self.assertTrue(is_danger)
        self.assertIn("dangerous pattern", reason)

    def test_sudo_detected(self) -> None:
        is_danger, reason = detect_dangerous_bash_command("sudo apt update")
        self.assertTrue(is_danger)
        self.assertIn("sudo", reason)

    def test_safe_command_not_detected(self) -> None:
        is_danger, _reason = detect_dangerous_bash_command("ls -la")
        self.assertFalse(is_danger)

    def test_empty_command_detected(self) -> None:
        is_danger, _reason = detect_dangerous_bash_command("")
        self.assertTrue(is_danger)


class TestDangerousWritePath(unittest.TestCase):
    def test_git_directory_detected(self) -> None:
        is_danger, _reason = detect_dangerous_write_path(".git/config")
        self.assertTrue(is_danger)

    def test_etc_detected(self) -> None:
        is_danger, _reason = detect_dangerous_write_path("/etc/passwd")
        self.assertTrue(is_danger)

    def test_safe_path_not_detected(self) -> None:
        is_danger, _reason = detect_dangerous_write_path("/tmp/test.py")
        self.assertFalse(is_danger)

    def test_empty_path_detected(self) -> None:
        is_danger, _reason = detect_dangerous_write_path("")
        self.assertTrue(is_danger)


class TestDangerousEditPath(unittest.TestCase):
    def test_git_head_detected(self) -> None:
        is_danger, _reason = detect_dangerous_edit_path(".git/HEAD")
        self.assertTrue(is_danger)

    def test_safe_path_not_detected(self) -> None:
        is_danger, _reason = detect_dangerous_edit_path("src/main.py")
        self.assertFalse(is_danger)


class TestDangerousToolCall(unittest.TestCase):
    def test_bash_dangerous(self) -> None:
        is_danger, _reason = detect_dangerous_tool_call("Bash", {"command": "rm -rf /tmp"})
        self.assertTrue(is_danger)

    def test_bash_safe(self) -> None:
        is_danger, _reason = detect_dangerous_tool_call("Bash", {"command": "ls -la"})
        self.assertFalse(is_danger)

    def test_write_dangerous(self) -> None:
        is_danger, _reason = detect_dangerous_tool_call("Write", {"file_path": ".git/config"})
        self.assertTrue(is_danger)

    def test_write_safe(self) -> None:
        is_danger, _reason = detect_dangerous_tool_call("Write", {"file_path": "/tmp/test.txt"})
        self.assertFalse(is_danger)

    def test_mcp_tool_dangerous(self) -> None:
        is_danger, _reason = detect_dangerous_tool_call("mcp__server__tool", {})
        self.assertTrue(is_danger)

    def test_read_safe(self) -> None:
        is_danger, _reason = detect_dangerous_tool_call("Read", {"file_path": "/etc/passwd"})
        self.assertFalse(is_danger)


class TestLLMClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ToolPermissionContext(mode="auto")

    def test_cache_hit_returns_cached_result(self) -> None:
        get_cache().clear()
        mock_provider = MagicMock()
        mock_provider.chat.return_value = MagicMock(
            content='{"decision": "AUTO_ALLOW", "reasoning": "test", "confidence": 0.9}'
        )

        result1 = llm_classify_tool_call("Bash", {"command": "echo hello"}, self.context, provider=mock_provider)

        result2 = llm_classify_tool_call("Bash", {"command": "echo hello"}, self.context, provider=mock_provider)

        self.assertEqual(result1.decision, result2.decision)
        self.assertEqual(mock_provider.chat.call_count, 1)

    def test_no_provider_returns_ask_user(self) -> None:
        get_cache().clear()
        with patch(
            "clawcodex_ext.providers.runtime.build_provider_from_config",
            side_effect=ImportError("No provider"),
        ):
            result = llm_classify_tool_call("Bash", {"command": "ls"}, self.context, provider=None)
            self.assertEqual(result.decision, "ASK_USER")

    def test_json_response_parsed_correctly(self) -> None:
        get_cache().clear()
        mock_provider = MagicMock()
        mock_provider.chat.return_value = MagicMock(
            content='{"decision": "AUTO_DENY", "reasoning": "dangerous", "confidence": 0.95}'
        )

        result = llm_classify_tool_call("Bash", {"command": "rm -rf /"}, self.context, provider=mock_provider)

        self.assertEqual(result.decision, "AUTO_DENY")
        self.assertEqual(result.confidence, 0.95)

    def test_non_json_response_fallback(self) -> None:
        get_cache().clear()
        mock_provider = MagicMock()
        mock_provider.chat.return_value = MagicMock(content="AUTO_ALLOW: This is safe.")

        result = llm_classify_tool_call("Bash", {"command": "ls"}, self.context, provider=mock_provider)

        self.assertEqual(result.decision, "AUTO_ALLOW")


class TestAutoModeClassifyWithLLM(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ToolPermissionContext(mode="auto")

    def test_read_tools_allowed_directly(self) -> None:
        result = auto_mode_classify_with_llm("Read", {}, self.context)
        self.assertTrue(result.allow)

    def test_glob_allowed_directly(self) -> None:
        result = auto_mode_classify_with_llm("Glob", {}, self.context)
        self.assertTrue(result.allow)

    def test_safe_bash_allowed(self) -> None:
        result = auto_mode_classify_with_llm("Bash", {"command": "ls -la"}, self.context)
        self.assertTrue(result.allow)

    def test_llm_override_for_uncertain(self) -> None:
        get_cache().clear()
        mock_provider = MagicMock()
        mock_provider.chat.return_value = MagicMock(
            content='{"decision": "AUTO_ALLOW", "reasoning": "safe operation", "confidence": 0.85}'
        )

        result = auto_mode_classify_with_llm(
            "Bash",
            {"command": "npm run build"},
            self.context,
            provider=mock_provider,
            use_llm_for_uncertain=True,
        )

        self.assertTrue(result.allow)


class TestCanCycleToAuto(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ToolPermissionContext(mode="default")

    def test_default_mode_can_cycle(self) -> None:
        result = can_cycle_to_auto(self.context)
        self.assertTrue(result)

    def test_already_in_auto_mode(self) -> None:
        context = ToolPermissionContext(mode="auto")
        result = can_cycle_to_auto(context)
        self.assertTrue(result)

    def test_protected_directory_blocks_auto(self) -> None:
        context = ToolPermissionContext(mode="default")
        context.cwd = Path("/home/user/project/.git")
        result = can_cycle_to_auto(context, check_protected_directory=True)
        self.assertFalse(result)

    def test_protected_directory_check_disabled(self) -> None:
        context = ToolPermissionContext(mode="default")
        context.cwd = Path("/home/user/project/.git")
        result = can_cycle_to_auto(
            context,
            check_protected_directory=False,
        )
        self.assertTrue(result)

    def test_normal_directory_allowed(self) -> None:
        context = ToolPermissionContext(mode="default")
        context.cwd = Path("/home/user/project/src")
        result = can_cycle_to_auto(context)
        self.assertTrue(result)

    def test_danger_history_blocks_auto(self) -> None:
        from src.permissions.check import get_denial_tracker, reset_denial_tracker

        reset_denial_tracker()
        tracker = get_denial_tracker()
        for _ in range(3):
            tracker.record_denial("Bash")
        result = can_cycle_to_auto(
            self.context,
            check_danger_history=True,
        )
        self.assertFalse(result)
        reset_denial_tracker()

    def test_danger_history_check_disabled(self) -> None:
        from src.permissions.check import get_denial_tracker, reset_denial_tracker

        reset_denial_tracker()
        tracker = get_denial_tracker()
        for _ in range(5):
            tracker.record_denial("Bash")
        result = can_cycle_to_auto(
            self.context,
            check_danger_history=False,
        )
        self.assertTrue(result)
        reset_denial_tracker()


class TestAutoModeAvailabilityReason(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ToolPermissionContext(mode="default")

    def test_available_returns_none(self) -> None:
        reason = get_auto_mode_availability_reason(self.context)
        self.assertIsNone(reason)

    def test_protected_directory_returns_reason(self) -> None:
        context = ToolPermissionContext(mode="default")
        context.cwd = Path("/home/user/project/.git")
        reason = get_auto_mode_availability_reason(context)
        self.assertIsNotNone(reason)
        self.assertIn("protected", reason.lower())

    def test_danger_history_returns_reason(self) -> None:
        from src.permissions.check import get_denial_tracker, reset_denial_tracker

        reset_denial_tracker()
        tracker = get_denial_tracker()
        for _ in range(3):
            tracker.record_denial("Bash")
        reason = get_auto_mode_availability_reason(self.context)
        self.assertIsNotNone(reason)
        self.assertIn("dangerous", reason.lower())
        reset_denial_tracker()


if __name__ == "__main__":
    unittest.main()
