"""Agent Loop Hook 注册表（P102-D）。

提供 ``register_loop_hook`` / ``unregister_loop_hook`` / ``call_hooks`` API，
统一管理 pre_llm / post_llm / pre_tool / post_tool / on_turn_end / on_turn_start
等阶段的钩子注册与去注册。

用法::

    from clawcodex_ext.query.hook_registry import register_loop_hook, call_hooks

    def my_pre_llm_hook(messages, system_prompt, state, params):
        # 修改 messages 或 system_prompt
        return (messages, system_prompt)

    register_loop_hook("budget_mode", my_pre_llm_hook, "pre_llm", priority=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

LoopHookPhase = Literal[
    "pre_llm",
    "post_llm",
    "pre_tool",
    "post_tool",
    "on_turn_start",
    "on_turn_end",
]


@dataclass
class LoopHook:
    """单个 loop hook 的元数据。"""

    name: str
    fn: Callable[..., Any]
    phase: LoopHookPhase
    priority: int = 0


# 全局注册表：phase -> 按优先级排序的 hook 列表
_REGISTRY: dict[LoopHookPhase, list[LoopHook]] = {
    "pre_llm": [],
    "post_llm": [],
    "pre_tool": [],
    "post_tool": [],
    "on_turn_start": [],
    "on_turn_end": [],
}


# ── 公共 API ─────────────────────────────────────────────────────────


def register_loop_hook(
    name: str,
    fn: Callable[..., Any],
    phase: LoopHookPhase,
    priority: int = 0,
) -> None:
    """注册一个 loop hook。

    同名同 phase 的 hook 会先被注销，再重新注册，避免重复。
    """
    unregister_loop_hook(name, phase)
    hook = LoopHook(name=name, fn=fn, phase=phase, priority=priority)
    _REGISTRY[phase].append(hook)
    _REGISTRY[phase].sort(key=lambda h: h.priority)
    logger.debug("Registered loop hook %r at phase %r (priority=%d)", name, phase, priority)


def unregister_loop_hook(name: str, phase: LoopHookPhase) -> None:
    """注销指定 phase 中 name 匹配的 hook。"""
    before = len(_REGISTRY[phase])
    _REGISTRY[phase] = [h for h in _REGISTRY[phase] if h.name != name]
    after = len(_REGISTRY[phase])
    if after < before:
        logger.debug("Unregistered loop hook %r from phase %r", name, phase)


def call_hooks(phase: LoopHookPhase, *args: Any, **kwargs: Any) -> tuple[Any, ...]:  # noqa: ANN401
    """按优先级顺序调用指定 phase 的所有已注册 hook。

    每个 hook 的返回值（如果非 None）会替换传入的 *args，供下一个 hook 使用。
    这允许 pre_llm hook 通过返回 ``(messages, system_prompt)`` 来修改参数。

    返回最终经过所有 hook 处理后的 args tuple。
    """
    hooks = _REGISTRY.get(phase, [])
    current_args: tuple[Any, ...] = args
    for hook in hooks:
        try:
            result = hook.fn(*current_args, **kwargs)
        except Exception:
            logger.exception("Loop hook %r at phase %r raised an error", hook.name, phase)
            continue
        if result is not None:
            if isinstance(result, tuple):
                current_args = result
            else:
                current_args = (result,)
    return current_args


def list_hooks(phase: LoopHookPhase | None = None) -> list[LoopHook]:
    """返回已注册 hook 的列表（只读副本）。用于调试和测试。"""
    if phase is not None:
        return list(_REGISTRY.get(phase, []))
    return [h for hooks in _REGISTRY.values() for h in hooks]


def clear_hooks(phase: LoopHookPhase | None = None) -> None:
    """清空注册表。主要用于测试隔离。"""
    if phase is not None:
        _REGISTRY[phase].clear()
    else:
        for p in _REGISTRY:  # pylint: disable=consider-using-dict-items
            _REGISTRY[p].clear()
