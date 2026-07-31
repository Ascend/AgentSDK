"""类型化的 outbox 事件（P102-C）。

将 ``ToolContext.outbox`` 从 ``list[dict[str, Any]]`` 改为 ``list[OutboxEvent]``
Union dataclass，使 mypy --strict 通过，并为未来事件类型提供扩展点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class CronPromptEvent:
    """Scheduler 触发的 cron 任务执行提示。"""

    prompt: str = ""
    task_id: str = ""
    run_id: str = ""

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        if key == "type":
            return "cron_prompt"
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        if key == "type":
            return "cron_prompt"
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key == "type" or hasattr(self, key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return other == {
                "type": "cron_prompt",
                "prompt": self.prompt,
                "task_id": self.task_id,
                "run_id": self.run_id,
            }
        if isinstance(other, CronPromptEvent):
            return self.prompt == other.prompt and self.task_id == other.task_id and self.run_id == other.run_id
        return NotImplemented


@dataclass
class CronMissedEvent:
    """Cron missed one-shot 通知。"""

    tasks: list[str] = field(default_factory=list)
    notification: str = ""

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        if key == "type":
            return "cron_missed"
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        if key == "type":
            return "cron_missed"
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key == "type" or hasattr(self, key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return other == {
                "type": "cron_missed",
                "tasks": self.tasks,
                "notification": self.notification,
            }
        if isinstance(other, CronMissedEvent):
            return self.tasks == other.tasks and self.notification == other.notification
        return NotImplemented


@dataclass
class ProactivePromptEvent:
    """Prompt injected by proactive tick/sleep wake-up."""

    prompt: str = ""
    source: str = "tick"

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        if key == "type":
            return "proactive_prompt"
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        if key == "type":
            return "proactive_prompt"
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key == "type" or hasattr(self, key)


@dataclass
class GenericOutboxEvent:
    """通用 outbox 事件，兼容任意工具写入的键值对。

    保留 ``payload`` dict 以容纳 ``tool``、``message``、``questions`` 等
    任意字段，同时提供类型标注。
    """

    payload: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        return self.payload.get(key, default)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        return self.payload[key]

    def __contains__(self, key: str) -> bool:
        return key in self.payload

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenericOutboxEvent:
        return cls(payload=dict(d))


OutboxEvent = Union[CronPromptEvent, CronMissedEvent, ProactivePromptEvent, GenericOutboxEvent]


def outbox_event_from_dict(d: dict[str, Any]) -> OutboxEvent:
    """从原始 dict 反序列化为类型化的 OutboxEvent。

    根据 ``type`` 字段自动分发到具体子类；未知类型统一归到
    ``GenericOutboxEvent``。
    """
    etype = d.get("type", "")
    if etype == "cron_prompt":
        return CronPromptEvent(
            prompt=d.get("prompt", ""),
            task_id=d.get("task_id", ""),
            run_id=d.get("run_id", ""),
        )
    if etype == "cron_missed":
        return CronMissedEvent(
            tasks=d.get("tasks", []),
            notification=d.get("notification", ""),
        )
    if etype == "proactive_prompt":
        return ProactivePromptEvent(
            prompt=d.get("prompt", ""),
            source=d.get("source", "tick"),
        )
    return GenericOutboxEvent.from_dict(d)
