"""
Comprehensive tests for the command system.

Tests cover:
- Command type system
- Argument substitution
- Command registry
- Command execution engine
- Built-in commands
- Skills integration
"""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from src.command_system import (
    CommandContext,
    CommandEngine,
    CommandRegistry,
    CommandType,
    LocalCommand,
    LocalCommandResult,
    PromptCommand,
    create_command_context,
    execute_command_async,
    execute_command_sync,
    get_command_registry,
    is_command_enabled,
    parse_argument_names,
    register_builtin_commands,
    substitute_arguments,
)
from src.bootstrap.state import reset_state_for_tests
from src.cost_tracker import CostTracker
from src.history import HistoryLog


@dataclass
class MockConversation:
    """Mock conversation for testing."""

    messages: list = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []

    def clear(self):
        self.messages.clear()


class TestArgumentSubstitution(unittest.TestCase):
    """Tests for argument substitution."""

    def test_simple_positional_args(self):
        """Test simple positional argument substitution."""
        content = "Hello $0 and $1!"
        result = substitute_arguments(content, "Alice Bob")
        self.assertEqual(result, "Hello Alice and Bob!")

    def test_named_args(self):
        """Test named argument substitution."""
        content = "Hello $name, you are $age years old"
        result = substitute_arguments(content, "Alice 30", ["name", "age"])
        self.assertEqual(result, "Hello Alice, you are 30 years old")

    def test_all_args_placeholder(self):
        """Test $ARGUMENTS placeholder."""
        content = "Args: $ARGUMENTS"
        result = substitute_arguments(content, 'foo bar "baz qux"')
        self.assertEqual(result, 'Args: foo bar "baz qux"')

    def test_parse_argument_names_string(self):
        """Test parsing argument names from string."""
        self.assertEqual(
            parse_argument_names("name, age, location"),
            ["name", "age", "location"],
        )

    def test_parse_argument_names_list(self):
        """Test parsing argument names from list."""
        self.assertEqual(
            parse_argument_names(["name", "age"]),
            ["name", "age"],
        )


class TestCommandTypes(unittest.TestCase):
    """Tests for the command type system."""

    def test_prompt_command_creation(self):
        """Test creating a PromptCommand."""
        cmd = PromptCommand(
            name="test-prompt",
            description="Test prompt command",
            progress_message="Testing...",
            content_length=100,
            markdown_content="# Test\n\nHello world",
        )
        self.assertEqual(cmd.command_type, CommandType.PROMPT)
        self.assertEqual(cmd.name, "test-prompt")
        self.assertEqual(cmd.progress_message, "Testing...")

    def test_local_command_creation(self):
        """Test creating a LocalCommand."""

        def mock_call(args: str, context: CommandContext) -> LocalCommandResult:
            return LocalCommandResult(type="text", value=f"Called with: {args}")

        cmd = LocalCommand(
            name="test-local",
            description="Test local command",
            supports_non_interactive=True,
        )
        cmd.set_call(mock_call)

        self.assertEqual(cmd.command_type, CommandType.LOCAL)
        self.assertEqual(cmd.name, "test-local")

    def test_command_enabled_check(self):
        """Test command enabled check."""
        enabled = True

        def check_enabled() -> bool:
            nonlocal enabled
            return enabled

        cmd = PromptCommand(
            name="test",
            description="Test",
            is_enabled=check_enabled,
        )

        self.assertTrue(is_command_enabled(cmd))
        enabled = False
        self.assertFalse(is_command_enabled(cmd))


class TestCommandRegistry(unittest.TestCase):
    """Tests for the command registry."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = CommandRegistry()

    def tearDown(self):
        """Clean up test fixtures."""
        get_command_registry().clear()

    def test_register_and_get_command(self):
        """Test registering and getting a command."""
        cmd = PromptCommand(
            name="test",
            description="Test command",
        )
        self.registry.register(cmd)

        retrieved = self.registry.get("test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test")

    def test_register_with_alias(self):
        """Test registering a command with aliases."""
        cmd = PromptCommand(
            name="test",
            description="Test command",
            aliases=["t", "testing"],
        )
        self.registry.register(cmd)

        self.assertIsNotNone(self.registry.get("t"))
        self.assertIsNotNone(self.registry.get("testing"))
        self.assertEqual(self.registry.get("t").name, "test")

    def test_list_commands(self):
        """Test listing commands."""
        cmd1 = PromptCommand(name="test1", description="Test 1")
        cmd2 = PromptCommand(name="test2", description="Test 2", is_hidden=True)
        self.registry.register(cmd1)
        self.registry.register(cmd2)

        commands = self.registry.list_commands()
        self.assertEqual(len(commands), 1)

    def test_find_commands(self):
        """Test finding commands by search."""
        cmd1 = PromptCommand(name="help", description="Show help")
        cmd2 = PromptCommand(name="hello", description="Say hello")
        self.registry.register(cmd1)
        self.registry.register(cmd2)

        matches = self.registry.find_commands("he")
        self.assertEqual(len(matches), 2)


class TestBuiltinCommands(unittest.TestCase):
    """Tests for built-in commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.tmpdir.name).resolve()
        self.conversation = MockConversation()
        self.cost_tracker = CostTracker()
        self.history = HistoryLog()
        self.runtimes = []

        self.context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        for runtime in self.runtimes:
            scheduler = getattr(runtime.tool_context, "cron_scheduler", None)
            if scheduler is not None:
                scheduler.stop()
        self.tmpdir.cleanup()

    def _build_runtime(self):
        from clawcodex_ext.runtime.context import RuntimeContext, RuntimeOptions

        runtime = RuntimeContext.build(
            RuntimeOptions(workspace_root=self.workspace_root, skip_permissions=True),
        )
        self.runtimes.append(runtime)
        return runtime

    def test_register_builtin_commands(self):
        """Test registering built-in commands."""
        registry = CommandRegistry()
        register_builtin_commands(registry)

        self.assertTrue(registry.has("help"))
        self.assertTrue(registry.has("clear"))
        self.assertTrue(registry.has("exit"))
        self.assertTrue(registry.has("skills"))
        self.assertTrue(registry.has("cost"))
        self.assertTrue(registry.has("context"))
        self.assertTrue(registry.has("compact"))
        self.assertTrue(registry.has("cron-list"))
        self.assertTrue(registry.has("cron-delete"))
        self.assertTrue(registry.has("cron-status"))
        self.assertTrue(registry.has("cron-runs"))
        self.assertTrue(registry.has("cron-run"))
        self.assertTrue(registry.has("init"))

    def test_cost_command_shows_non_zero_after_record_usage(self):
        """Regression: /cost must show non-zero tokens after record_usage()."""
        reset_state_for_tests()

        # Record non-zero usage via the real tracker.
        self.cost_tracker.record_usage(
            "claude-sonnet-4-20250514",
            {"input_tokens": 1000, "output_tokens": 500},
        )

        success, result, _ = execute_command_sync("cost", "", self.context)
        self.assertTrue(success)
        self.assertIn("1,000", result)
        self.assertIn("500", result)
        self.assertIn("claude-sonnet-4-20250514", result)

    def test_cost_command_shows_no_usage_message_when_empty(self):
        """Regression: /cost must show a clear 'no usage' message when empty."""
        reset_state_for_tests()

        success, result, _ = execute_command_sync("cost", "", self.context)
        self.assertTrue(success)
        self.assertIn("No API usage recorded yet", result)

    def test_cron_list_uses_injected_tool_runtime(self):
        """Test that /cron-list reads jobs from the injected cron runtime."""
        runtime = self._build_runtime()
        create_tool = runtime.tool_registry.get("CronCreate")
        self.assertIsNotNone(create_tool)
        created = create_tool.call(
            {"cron": "*/5 * * * *", "prompt": "session ping"},
            runtime.tool_context,
        ).output
        context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
            tool_registry=runtime.tool_registry,
            tool_context=runtime.tool_context,
        )

        success, result, error = execute_command_sync("cron-list", "", context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn(created["id"], result)
        self.assertIn("session ping", result)

    def test_cron_delete_uses_injected_tool_runtime(self):
        """Test that /cron-delete deletes through the injected cron runtime."""
        runtime = self._build_runtime()
        create_tool = runtime.tool_registry.get("CronCreate")
        list_tool = runtime.tool_registry.get("CronList")
        self.assertIsNotNone(create_tool)
        self.assertIsNotNone(list_tool)
        created = create_tool.call(
            {"cron": "*/5 * * * *", "prompt": "session ping"},
            runtime.tool_context,
        ).output
        context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
            tool_registry=runtime.tool_registry,
            tool_context=runtime.tool_context,
        )

        success, result, error = execute_command_sync("cron-delete", created["id"], context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn(created["id"], result)
        listed = list_tool.call({}, runtime.tool_context).output
        self.assertEqual(listed["jobs"], [])

    def test_cron_delete_honors_tool_permission_context(self):
        """Test that /cron-delete dispatches through permission checks."""
        from src.permissions.types import ToolPermissionContext

        runtime = self._build_runtime()
        create_tool = runtime.tool_registry.get("CronCreate")
        self.assertIsNotNone(create_tool)
        created = create_tool.call(
            {"cron": "*/5 * * * *", "prompt": "session ping"},
            runtime.tool_context,
        ).output
        runtime.tool_context.permission_context = ToolPermissionContext.from_iterables(
            deny_names=["CronDelete"],
        )
        context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
            tool_registry=runtime.tool_registry,
            tool_context=runtime.tool_context,
        )

        success, result, error = execute_command_sync("cron-delete", created["id"], context)

        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertIn("CronDelete", error or "")

    def test_cron_list_without_runtime_reads_persistent_tasks(self):
        """Test that /cron-list falls back to persistent cron storage."""
        from clawcodex_ext.cron_system.tasks import add_cron_task

        task = add_cron_task(
            self.workspace_root,
            cron="*/5 * * * *",
            prompt="persistent ping",
            durable=True,
        )

        success, result, error = execute_command_sync("cron-list", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn(task.id, result or "")
        self.assertIn("persistent ping", result or "")

    def test_cron_list_without_runtime_reports_empty_storage(self):
        """Test that /cron-list works without injected runtime handles."""
        success, result, error = execute_command_sync("cron-list", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertEqual("No scheduled cron jobs.", result)

    def test_cron_delete_without_runtime_fails_closed(self):
        """Test that /cron-delete does not mutate persistent tasks without runtime permissions."""
        from clawcodex_ext.cron_system.tasks import add_cron_task, read_all_cron_tasks

        task = add_cron_task(
            self.workspace_root,
            cron="*/5 * * * *",
            prompt="persistent ping",
            durable=True,
        )

        success, result, error = execute_command_sync("cron-delete", task.id, self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("Cron runtime is required", result or "")
        self.assertEqual([task.id], [stored.id for stored in read_all_cron_tasks(self.workspace_root)])

    def test_cron_status_and_runs_work_without_runtime(self):
        """Test status views read the persistent cron ledger directly."""
        success, status, status_error = execute_command_sync("cron-status", "--deep", self.context)
        runs_success, runs, runs_error = execute_command_sync("cron-runs", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(status_error)
        self.assertIn("Autonomy status", status or "")
        self.assertIn("Cron jobs", status or "")
        self.assertIn("Scheduled-task runs", status or "")
        self.assertTrue(runs_success)
        self.assertIsNone(runs_error)
        self.assertEqual("No scheduled-task runs.", runs)

    def test_cron_run_queues_manual_fire_in_outbox(self):
        """Test that /cron-run creates a run and queues it for REPL execution."""
        from clawcodex_ext.cron_system.runs import read_cron_runs
        from clawcodex_ext.cron_system.tasks import add_cron_task

        runtime = self._build_runtime()
        runtime.tool_context.outbox = []
        task = add_cron_task(
            self.workspace_root,
            cron="*/5 * * * *",
            prompt="manual ping",
            durable=True,
        )
        context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
            tool_registry=runtime.tool_registry,
            tool_context=runtime.tool_context,
        )

        success, result, error = execute_command_sync("cron-run", task.id, context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("Queued for execution in this session", result or "")
        runs = read_cron_runs(self.workspace_root)
        self.assertEqual(1, len(runs))
        self.assertEqual("queued", runs[0].status)
        self.assertEqual(
            [
                {
                    "type": "cron_prompt",
                    "prompt": "manual ping",
                    "task_id": task.id,
                    "run_id": runs[0].id,
                }
            ],
            context.tool_context.outbox,
        )

    def test_cron_run_tolerates_outbox_append_failure(self):
        """Test manual fire still succeeds when the active outbox cannot be used."""
        from clawcodex_ext.cron_system.runs import read_cron_runs
        from clawcodex_ext.cron_system.tasks import add_cron_task

        class BrokenOutbox:
            def append(self, item):
                raise RuntimeError("outbox full")

        runtime = self._build_runtime()
        runtime.tool_context.outbox = BrokenOutbox()
        task = add_cron_task(
            self.workspace_root,
            cron="*/5 * * * *",
            prompt="manual ping",
            durable=True,
        )
        context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
            tool_registry=runtime.tool_registry,
            tool_context=runtime.tool_context,
        )

        success, result, error = execute_command_sync("cron-run", task.id, context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("Queued, but no active cron outbox", result or "")
        runs = read_cron_runs(self.workspace_root)
        self.assertEqual(1, len(runs))
        self.assertEqual("queued", runs[0].status)

    def test_cron_run_alias_and_duplicate_active_run(self):
        """Test /cron-fire alias and duplicate active-run message."""
        from clawcodex_ext.cron_system.tasks import add_cron_task

        runtime = self._build_runtime()
        context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
            tool_registry=runtime.tool_registry,
            tool_context=runtime.tool_context,
        )
        task = add_cron_task(
            self.workspace_root,
            cron="*/5 * * * *",
            prompt="manual ping",
            durable=True,
        )

        first_success, _, first_error = execute_command_sync("cron-fire", task.id, context)
        second_success, second_result, second_error = execute_command_sync("cron-run", task.id, context)

        self.assertTrue(first_success)
        self.assertIsNone(first_error)
        self.assertTrue(second_success)
        self.assertIsNone(second_error)
        self.assertIn("previous run is still queued or running", second_result or "")

    def test_cron_run_reports_missing_task(self):
        """Test manual fire reports unknown task ids clearly."""
        runtime = self._build_runtime()
        context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
            tool_registry=runtime.tool_registry,
            tool_context=runtime.tool_context,
        )

        success, result, error = execute_command_sync("cron-run", "missing", context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("No scheduled cron job found", result or "")
        self.assertIn("missing", result or "")

    def test_cron_run_without_runtime_fails_closed(self):
        """Test manual fire does not queue runs without runtime permissions."""
        from clawcodex_ext.cron_system.runs import read_cron_runs
        from clawcodex_ext.cron_system.tasks import add_cron_task

        task = add_cron_task(
            self.workspace_root,
            cron="*/5 * * * *",
            prompt="manual ping",
            durable=True,
        )

        success, result, error = execute_command_sync("cron-run", task.id, self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("Cron runtime is required", result or "")
        self.assertEqual([], read_cron_runs(self.workspace_root))

    def test_cron_run_honors_tool_permission_context(self):
        """Test /cron-run dispatches through CronRun permission checks."""
        from clawcodex_ext.cron_system.runs import read_cron_runs
        from clawcodex_ext.cron_system.tasks import add_cron_task
        from src.permissions.types import ToolPermissionContext

        runtime = self._build_runtime()
        runtime.tool_context.permission_context = ToolPermissionContext.from_iterables(
            deny_names=["CronRun"],
        )
        task = add_cron_task(
            self.workspace_root,
            cron="*/5 * * * *",
            prompt="manual ping",
            durable=True,
        )
        context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
            tool_registry=runtime.tool_registry,
            tool_context=runtime.tool_context,
        )

        success, result, error = execute_command_sync("cron-run", task.id, context)

        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertIn("CronRun", error or "")
        self.assertEqual([], read_cron_runs(self.workspace_root))

    def test_cron_run_respects_cron_kill_switch(self):
        """Test manual fire follows CLAWCODEX_DISABLE_CRON."""

        from clawcodex_ext.cron_system.runs import read_cron_runs
        from clawcodex_ext.cron_system.tasks import add_cron_task

        runtime = self._build_runtime()
        context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
            tool_registry=runtime.tool_registry,
            tool_context=runtime.tool_context,
        )
        task = add_cron_task(
            self.workspace_root,
            cron="*/5 * * * *",
            prompt="manual ping",
            durable=True,
        )

        with patch.dict(os.environ, {"CLAWCODEX_DISABLE_CRON": "1"}):
            success, result, error = execute_command_sync("cron-run", task.id, context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("Cron is disabled", result or "")
        self.assertEqual([], read_cron_runs(self.workspace_root))

    def test_skills_command_with_project_root(self):
        """Test that /skills command can find project skills."""
        from src.skills.create import create_skill

        # Create a skill in the temp directory
        project_skills_dir = Path(self.tmpdir.name) / ".clawcodex" / "skills"
        project_skills_dir.mkdir(parents=True)

        create_skill(
            directory=project_skills_dir,
            name="test-project-skill",
            description="A test skill in project",
            body="Hello from project skill",
        )

        # Create command context with the project root
        context = create_command_context(
            workspace_root=self.tmpdir.name,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
        )

        # Execute /skills command
        success, result, error = execute_command_sync("skills", "", context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("test-project-skill", result)


class TestCommandEngine(unittest.TestCase):
    """Tests for the command engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.tmpdir.name).resolve()
        self.registry = CommandRegistry()
        register_builtin_commands(self.registry)

        self.conversation = MockConversation()
        self.cost_tracker = CostTracker()
        self.history = HistoryLog()

        self.context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
        )

        self.engine = CommandEngine(
            registry=self.registry,
            workspace_root=self.workspace_root,
            context=self.context,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.tmpdir.cleanup()

    async def test_execute_help_command(self):
        """Test executing /help command."""
        result = await self.engine.execute("/help")

        self.assertTrue(result.success)
        self.assertEqual(result.command_name, "help")

    async def test_execute_unknown_command(self):
        """Test executing unknown command."""
        result = await self.engine.execute("/unknown-command")

        self.assertFalse(result.success)
        self.assertIn("Unknown command", result.error or "")


class TestSkillsIntegration(unittest.TestCase):
    """Tests for skills system integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.skills_dir = Path(self.tmpdir.name) / "skills"
        self.skills_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        self.tmpdir.cleanup()

    def test_skill_to_prompt_command(self):
        """Test converting a skill to a prompt command."""
        from src.command_system.skills_integration import skill_to_prompt_command
        from src.skills.create import create_skill

        create_skill(
            directory=self.skills_dir,
            name="test-skill",
            description="Test skill",
            when_to_use="When testing",
            allowed_tools=["Read", "Grep"],
            arguments=["name"],
            body="Hello $name",
        )

        from src.skills.loader import load_skills_from_dir

        skills = load_skills_from_dir(self.skills_dir)
        self.assertEqual(len(skills), 1)

        cmd = skill_to_prompt_command(skills[0])

        self.assertEqual(cmd.name, "test-skill")
        self.assertEqual(cmd.description, "Test skill")
        self.assertEqual(cmd.markdown_content, "Hello $name")


class TestInitCommand(unittest.TestCase):
    """Tests for the /init command implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.tmpdir.name).resolve()
        self.registry = CommandRegistry()
        register_builtin_commands(self.registry)

        self.conversation = MockConversation()
        self.cost_tracker = CostTracker()
        self.history = HistoryLog()

        self.context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
        )

        self.engine = CommandEngine(
            registry=self.registry,
            workspace_root=self.workspace_root,
            context=self.context,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.tmpdir.cleanup()

    def _get_init_command(self):
        """Get the /init command from the local registry."""
        return self.registry.get("init")

    def test_init_command_is_prompt_command(self):
        """Test that /init is a PromptCommand, not LocalCommand."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        self.assertEqual(init_cmd.command_type, CommandType.PROMPT)
        self.assertIsInstance(init_cmd, PromptCommand)

    def test_init_command_has_correct_description(self):
        """Test that /init has the correct description."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        self.assertIn("CLAUDE.md", init_cmd.description)
        self.assertIn("skills", init_cmd.description)
        self.assertIn("hooks", init_cmd.description)

    def test_init_command_has_progress_message(self):
        """Test that /init has a progress message."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        self.assertEqual(init_cmd.progress_message, "analyzing your codebase")

    def test_init_command_has_prompt_content(self):
        """Test that /init has the 7-step prompt content."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        self.assertIsInstance(init_cmd, PromptCommand)
        # Verify it contains the key steps
        content = init_cmd.markdown_content
        self.assertIn("Step 1", content)
        self.assertIn("Step 2", content)
        self.assertIn("Step 3", content)
        self.assertIn("Step 4", content)
        self.assertIn("Step 5", content)
        self.assertIn("Step 6", content)
        self.assertIn("Step 7", content)

    def test_init_command_includes_claude_md_instructions(self):
        """Test that /init prompt includes CLAUDE.md creation instructions."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        content = init_cmd.markdown_content
        # Verify it includes key CLAUDE.md content requirements
        self.assertIn("CLAUDE.md", content)
        self.assertIn("build/test/lint", content.lower())
        self.assertIn("code style", content.lower())

    def test_init_command_includes_user_interaction_phases(self):
        """Test that /init prompt includes user interaction phases."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        content = init_cmd.markdown_content
        # Verify it includes AskUserQuestion references
        self.assertIn("AskUserQuestion", content)

    async def test_execute_init_command_via_engine(self):
        """Test executing /init via the async engine."""
        result = await self.engine.execute("/init")

        self.assertTrue(result.success)
        self.assertEqual(result.command_name, "init")
        self.assertEqual(result.result_type, "prompt")
        self.assertTrue(result.should_query)
        self.assertEqual(result.display, "user")
        # Verify prompt content was returned
        self.assertTrue(len(result.prompt_content) > 0)
        self.assertEqual(result.prompt_content[0]["type"], "text")

    async def test_execute_init_command_async(self):
        """Test executing /init via execute_command_async."""
        # Register commands to global registry for this test
        from src.command_system import get_command_registry

        registry = get_command_registry()
        register_builtin_commands(registry)

        result = await execute_command_async("init", "", self.context)

        self.assertTrue(result.success)
        self.assertEqual(result.command_name, "init")
        self.assertEqual(result.result_type, "prompt")
        self.assertTrue(len(result.prompt_content) > 0)

    def test_sync_execute_does_not_handle_init(self):
        """Test that sync execution returns error for /init (it's a PromptCommand)."""
        # Register commands to global registry for this test
        from src.command_system import get_command_registry

        registry = get_command_registry()
        register_builtin_commands(registry)

        success, result, error = execute_command_sync("init", "", self.context)
        # Sync execution doesn't handle PromptCommand
        # It will either return False or an error
        if not success:
            # Expected: sync can't handle PromptCommand
            pass
        else:
            # If sync succeeds, it means LocalCommand handled it (shouldn't happen for /init)
            self.fail("/init should not be a LocalCommand")


if __name__ == "__main__":
    unittest.main()
