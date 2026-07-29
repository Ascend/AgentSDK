"""Tool system tools package.

Re-exports all built-in tool classes and ``ALL_STATIC_TOOLS``.
"""

from __future__ import annotations

from ..build_tool import Tool

from .advisor import AdvisorTool
from .agent import make_agent_tool
from .ask_user_question import AskUserQuestionTool
from .bash import BashTool
from .brief import BriefTool
from .config import ConfigTool
from .cron import CronCreateTool, CronDeleteTool, CronListTool

# F-120 Agent Dashboard — read-only observability tools.
from extensions.agent_dashboard.tools import (
    DashboardGetTool,
    DashboardListTool,
)
from lkb.clawcodex_tool import LkbTool
from .edit import EditTool
from .execute import ExecuteTool
from .glob import GlobTool
from .grep import GrepTool
from .lsp import LSPTool
from .mcp import MCPTool
from .memory import MemoryTool
from .notebook_edit import NotebookEditTool
from .mcp_resources import ListMcpResourcesTool, ReadMcpResourceTool
from .misc import ClipboardReadTool, ClipboardWriteTool, StatusTool
from .monitor import MonitorTool
from .plan_mode import EnterPlanModeTool, ExitPlanModeTool
from .read import ReadTool
from .remote_trigger import RemoteTriggerTool
from .schedule_wakeup import ScheduleWakeupTool
from .send_message import SendMessageTool
from .send_user_message import SendUserMessageTool
from .skill import SkillTool
from .skill_search import SkillSearchTool
from .snip import SnipTool
from .sleep import SleepTool
from .structured_output import StructuredOutputTool
from .task_stop import TaskStopTool
from .tasks_v2 import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskUpdateTool,
)
from .team import TeamCreateTool, TeamDeleteTool
from .team_memory import TeamMemoryTool
from .todo_write import TodoWriteTool
from .tool_search import make_tool_search_tool
from .web_browser import WebBrowserTool
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool
from .worktree import EnterWorktreeTool, ExitWorktreeTool
from .write import WriteTool

ALL_STATIC_TOOLS: list[Tool] = [
    AdvisorTool,
    AskUserQuestionTool,
    BashTool,
    BriefTool,
    ClipboardReadTool,
    ClipboardWriteTool,
    ConfigTool,
    CronCreateTool,
    CronDeleteTool,
    CronListTool,
    DashboardGetTool,
    DashboardListTool,
    EditTool,
    EnterPlanModeTool,
    EnterWorktreeTool,
    ExecuteTool,
    ExitPlanModeTool,
    ExitWorktreeTool,
    GlobTool,
    GrepTool,
    LSPTool,
    LkbTool,
    ListMcpResourcesTool,
    MCPTool,
    MonitorTool,
    MemoryTool,
    NotebookEditTool,
    ReadMcpResourceTool,
    ReadTool,
    RemoteTriggerTool,
    ScheduleWakeupTool,
    SendMessageTool,
    SendUserMessageTool,
    SkillTool,
    SkillSearchTool,
    SleepTool,
    SnipTool,
    StatusTool,
    # Restored for SOP Overview / domain allowlists (early clawcodex_ext
    # behavior). Workflow schema runs may still inject a validating
    # per-call instance via make_structured_output_tool that shadows this.
    StructuredOutputTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskStopTool,
    TaskUpdateTool,
    TeamCreateTool,
    TeamDeleteTool,
    TeamMemoryTool,
    TodoWriteTool,
    WebBrowserTool,
    WebFetchTool,
    WebSearchTool,
    WriteTool,
]

__all__ = [
    "ALL_STATIC_TOOLS",
    "AdvisorTool",
    "AskUserQuestionTool",
    "BashTool",
    "BriefTool",
    "ClipboardReadTool",
    "ClipboardWriteTool",
    "ConfigTool",
    "CronCreateTool",
    "CronDeleteTool",
    "CronListTool",
    "DashboardGetTool",
    "DashboardListTool",
    "EditTool",
    "EnterPlanModeTool",
    "EnterWorktreeTool",
    "ExecuteTool",
    "ExitPlanModeTool",
    "ExitWorktreeTool",
    "GlobTool",
    "GrepTool",
    "LSPTool",
    "LkbTool",
    "ListMcpResourcesTool",
    "MCPTool",
    "MonitorTool",
    "MemoryTool",
    "NotebookEditTool",
    "ReadMcpResourceTool",
    "ReadTool",
    "RemoteTriggerTool",
    "ScheduleWakeupTool",
    "SendMessageTool",
    "SendUserMessageTool",
    "SkillTool",
    "SkillSearchTool",
    "SleepTool",
    "SnipTool",
    "StatusTool",
    "MonitorTool",
    "StructuredOutputTool",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskOutputTool",
    "TaskStopTool",
    "TaskUpdateTool",
    "TeamCreateTool",
    "TeamDeleteTool",
    "TeamMemoryTool",
    "TodoWriteTool",
    "WebBrowserTool",
    "WebFetchTool",
    "WebSearchTool",
    "WriteTool",
    "make_agent_tool",
    "make_tool_search_tool",
]
