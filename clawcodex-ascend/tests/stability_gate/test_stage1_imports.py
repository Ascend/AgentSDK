#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Stage 1 — 核心模块导入验证（< 2 秒）。

确保所有核心模块可导入且关键类/函数可调用。这是最基本的门禁，
防止因 import 链断裂、AttributeError 或依赖缺失导致的启动崩溃。
"""

from __future__ import annotations


class TestStage1CoreImports:
    """验证 src/ 下所有核心模块的导入性。"""

    def test_src_cli_import(self):
        import src.cli

        assert callable(src.cli.main)
        assert callable(src.cli._build_parser)

    def test_src_entrypoints_tui_import(self):
        from src.entrypoints import tui

        assert hasattr(tui, "run_tui")
        assert hasattr(tui, "TUIOptions")
        assert hasattr(tui, "should_use_tui")

    def test_src_entrypoints_headless_import(self):
        import src.entrypoints.headless as hl

        assert hasattr(hl, "run_headless")
        assert hasattr(hl, "HeadlessOptions")

    def test_repl_core_import(self):
        import src.repl.core as repl_core

        assert hasattr(repl_core, "ClawcodexREPL")

    def test_agent_session_import(self):
        import src.agent.session

        assert hasattr(src.agent.session, "Session")
        assert callable(src.agent.session.Session.create)

    def test_agent_conversation_import(self):
        import src.agent.conversation

        assert hasattr(src.agent.conversation, "Conversation")
        assert callable(src.agent.conversation.Conversation)

    def test_providers_base_import(self):
        import src.providers.base

        assert hasattr(src.providers.base, "ChatResponse")

    def test_tool_system_context_import(self):
        import src.tool_system.context

        assert hasattr(src.tool_system.context, "ToolContext")

    def test_tool_system_defaults_import(self):
        import src.tool_system.defaults

        assert callable(src.tool_system.defaults.build_default_registry)

    def test_permissions_types_import(self):
        import src.permissions.types

        assert hasattr(src.permissions.types, "ToolPermissionContext")

    def test_bootstrap_state_import(self):
        import src.bootstrap.state

        assert callable(src.bootstrap.state.get_session_id)

    def test_types_messages_import(self):
        from src.types.messages import AssistantMessage, UserMessage

        assert UserMessage is not None
        assert AssistantMessage is not None

    def test_types_content_blocks_import(self):
        from src.types.content_blocks import TextBlock, ToolUseBlock

        assert TextBlock is not None
        assert ToolUseBlock is not None

    def test_types_stream_events_import(self):
        from src.types.stream_events import (
            ContentBlockDelta,
            ContentBlockStart,
            ContentBlockStop,
            MessageDelta,
            MessageStart,
            MessageStop,
        )

        assert MessageStart is not None
        assert ContentBlockStart is not None
        assert ContentBlockDelta is not None
        assert ContentBlockStop is not None
        assert MessageDelta is not None
        assert MessageStop is not None

    def test_api_query_events_import(self):
        """Query loop 事件（TextDelta / ToolCallEvent / PhaseComplete）可导入。"""
        from extensions.api.query import (
            PhaseComplete,
            SessionComplete,
            TextDelta,
            ToolCallEvent,
            ToolResultEvent,
        )

        assert TextDelta is not None
        assert ToolCallEvent is not None
        assert ToolResultEvent is not None
        assert PhaseComplete is not None
        assert SessionComplete is not None

    def test_config_manager_import(self):
        import src.config

        assert hasattr(src.config, "ConfigManager")
        assert callable(src.config.get_default_provider)

    def test_init_module_import(self):
        import src.init

        assert callable(src.init.run_pre_action)

    def test_cli_core_exit_import(self):
        from src.cli_core.exit import cli_error

        assert callable(cli_error)

    def test_command_system_import(self):
        import src.command_system

        assert hasattr(src.command_system, "get_command_registry")
        assert hasattr(src.command_system, "CommandEngine")
        assert hasattr(src.command_system, "list_commands")

    def test_skills_frontmatter_import(self):
        from src.skills.frontmatter import parse_frontmatter

        assert callable(parse_frontmatter)

    def test_prefetch_import(self):
        import src.prefetch

        assert callable(src.prefetch.get_or_start_keychain_prefetch)

    def test_telemetry_import(self):
        """F-97: telemetry package importable; default-off path zero-cost."""
        from telemetry import recorder

        assert callable(recorder.get_recorder)
        # _NullRecorder is the default; its public methods must be no-ops
        # and must NOT raise. Validates the zero-cost on-by-default design.
        null = recorder._NullRecorder()
        assert null.enabled is False
        null.record_session_start(session_id="x", entrypoint="cli")
        null.record_session_end(session_id="x", duration_s=0.0, exit_status=0)
        null.record_command_run(session_id="x", command_name="repl")
        null.record_error(session_id="x", exc=RuntimeError("noop"))
        null.record_tool_summary(session_id="x", tool_name="bash")
        null.flush()
        null.close()

    # ------------------------------------------------------------------
    # Phase 2-A decoupling: 8 new src/ facades preserve import paths
    # ------------------------------------------------------------------

    def test_phase2a_bridge_repl_bridge_facade(self):
        import src.bridge.repl_bridge as m

        assert hasattr(m, "ReplBridgeHandle")
        assert hasattr(m, "BridgeState")
        assert hasattr(m, "BridgeCoreParams")
        assert callable(m.init_bridge_core)

    def test_phase2a_bridge_bridge_main_facade(self):
        import src.bridge.bridge_main as m

        assert hasattr(m, "BackoffConfig")
        assert hasattr(m, "ParsedArgs")
        assert callable(m.parse_args)
        assert callable(m.run_bridge_loop)
        assert callable(m.bridge_main)

    def test_phase2a_bridge_session_runner_facade(self):
        import src.bridge.session_runner as m

        assert hasattr(m, "SessionSpawnerDeps")
        assert hasattr(m, "CHILD_ENV_ALLOWLIST")
        assert callable(m.create_session_spawner)
        assert callable(m.build_child_env)

    def test_phase2a_bridge_remote_bridge_core_facade(self):
        import src.bridge.remote_bridge_core as m

        assert hasattr(m, "RemoteBridgeHandle")
        assert hasattr(m, "EnvLessBridgeParams")
        assert callable(m.init_env_less_bridge_core)

    def test_phase2a_transports_websocket_facade(self):
        import src.transports.websocket_transport as m

        assert hasattr(m, "WebSocketTransport")
        assert hasattr(m, "WebSocketTransportOptions")
        assert hasattr(m, "WebSocketTransportState")

    def test_phase2a_transports_serial_uploader_facade(self):
        import src.transports.serial_batch_event_uploader as m

        assert hasattr(m, "SerialBatchEventUploader")
        assert hasattr(m, "SerialBatchEventUploaderConfig")
        assert hasattr(m, "RetryableError")

    def test_phase2a_transports_hybrid_facade(self):
        import src.transports.hybrid_transport as m

        assert hasattr(m, "HybridTransport")
        assert hasattr(m, "BATCH_FLUSH_INTERVAL_S")
        assert hasattr(m, "POST_TIMEOUT_S")

    def test_phase2a_services_context_collapse_facade(self):
        import src.services.context_collapse as m

        assert hasattr(m, "CollapseEngine")
        assert hasattr(m, "BoundaryDetector")
        assert hasattr(m, "CollapseStateFile")
        assert hasattr(m, "TokenCounter")
        assert hasattr(m, "SummaryGenerator")
        assert hasattr(m, "Trigger")
        # Submodule facades also work
        from src.services.context_collapse import exceptions, persistence

        assert hasattr(exceptions, "ContextCollapseError")
        assert hasattr(persistence, "load_store")
