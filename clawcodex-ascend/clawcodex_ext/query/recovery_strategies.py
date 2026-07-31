# pylint: disable=relative-beyond-top-level
"""Post-LLM 恢复策略注册表（P102-B）。

将 ``query()`` 中 max_tokens/PTL/media_size 的硬编码恢复链改为注册式
``RecoveryStrategy`` 列表，使新错误恢复策略无需修改 ``query.py`` 即可注入。

用法::

    from clawcodex_ext.query.recovery_strategies import register_recovery_strategy, RecoveryContext

    def my_recovery(ctx: RecoveryContext) -> tuple[QueryState | None, list[Message]] | None:
        if ctx.error_type != "my_custom_error":
            return None
        # 构建新状态或返回终止消息
        return (new_state, [])

    register_recovery_strategy("my_recovery", my_recovery, priority=20)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from clawcodex_ext.types.messages import AssistantMessage, Message

if TYPE_CHECKING:
    from .config import FrozenQueryConfig, QueryConfig
    from .query import QueryParams
    from .transitions import QueryState

logger = logging.getLogger(__name__)


@dataclass
class RecoveryContext:
    """恢复策略执行时传入的上下文。

    包含当前 turn 的全部状态，策略可以读取任何字段并决定如何响应。
    """

    state: QueryState
    last_message: Message | None
    config: QueryConfig | FrozenQueryConfig
    params: QueryParams
    messages: list[Message]
    assistant_messages: list[AssistantMessage]
    error_type: str


RecoveryStrategyFn = Callable[[RecoveryContext], tuple["QueryState | None", list[Message]] | None]


@dataclass
class RecoveryStrategy:
    """单个恢复策略的元数据。"""

    name: str
    fn: RecoveryStrategyFn
    priority: int = 0


# 全局策略列表，按优先级排序
_STRATEGIES: list[RecoveryStrategy] = []

# 内置策略名称常量，方便测试和注销
MAX_OUTPUT_TOKENS_ESCALATE = "max_output_tokens_escalate"
MAX_OUTPUT_TOKENS_RECOVERY = "max_output_tokens_recovery"
MAX_OUTPUT_TOKENS_EXHAUSTED = "max_output_tokens_exhausted"
COLLAPSE_ENGINE_RECOVERY = "collapse_engine_recovery"
REACTIVE_COMPACT_RECOVERY = "reactive_compact_recovery"
MEDIA_SIZE_FALLBACK = "media_size_fallback"
PROMPT_TOO_LONG_FALLBACK = "prompt_too_long_fallback"


# ── 公共 API ─────────────────────────────────────────────────────────


def register_recovery_strategy(
    name: str,
    fn: RecoveryStrategyFn,
    priority: int = 0,
) -> None:
    """注册一个恢复策略。

    同名策略会先被注销，再重新注册，避免重复。
    """
    unregister_recovery_strategy(name)
    _STRATEGIES.append(RecoveryStrategy(name=name, fn=fn, priority=priority))
    _STRATEGIES.sort(key=lambda s: s.priority)
    logger.debug("Registered recovery strategy %r (priority=%d)", name, priority)


def unregister_recovery_strategy(name: str) -> None:
    """注销指定名称的恢复策略。"""
    before = len(_STRATEGIES)
    _STRATEGIES[:] = [s for s in _STRATEGIES if s.name != name]
    if len(_STRATEGIES) < before:
        logger.debug("Unregistered recovery strategy %r", name)


def find_recovery_strategies(
    error_type: str,
    state: QueryState,  # noqa: ARG001
) -> list[RecoveryStrategy]:
    """返回所有已注册策略（按优先级排序）。

    当前不根据 ``error_type`` 预过滤——每个策略的 ``fn`` 内部自行判断
    是否适用。这样策略可以基于更复杂的条件（如 ``state`` 字段、
    ``params`` 配置）做出决策。
    """
    return list(_STRATEGIES)


def clear_recovery_strategies() -> None:
    """清空所有策略。主要用于测试隔离。"""
    _STRATEGIES.clear()


# ── 内置策略实现 ─────────────────────────────────────────────────────


def _max_output_tokens_escalate(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """首次 max_output_tokens 错误：提升到 ESCALATED_MAX_TOKENS 并重试。"""
    if ctx.error_type != "max_output_tokens":
        return None
    from .transitions import QueryState, Transition

    s = ctx.state
    if s.max_output_tokens_override is not None or s.max_output_tokens_recovery_count != 0:
        return None

    from .query import ESCALATED_MAX_TOKENS

    new_state = QueryState(
        messages=s.messages,
        tool_use_context=s.tool_use_context,
        auto_compact_tracking=s.auto_compact_tracking,
        max_output_tokens_recovery_count=s.max_output_tokens_recovery_count,
        has_attempted_reactive_compact=s.has_attempted_reactive_compact,
        max_output_tokens_override=ESCALATED_MAX_TOKENS,
        stop_hook_active=None,
        turn_count=s.turn_count,
        pending_tool_use_summary=s.pending_tool_use_summary,
        continuation_nudge_count=s.continuation_nudge_count,
        transition=Transition(reason="max_output_tokens_escalate"),
    )
    return (new_state, [])


def _max_output_tokens_recovery(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """max_output_tokens 恢复：注入恢复提示并重试。"""
    if ctx.error_type != "max_output_tokens":
        return None
    from .transitions import QueryState, Transition

    s = ctx.state
    if s.max_output_tokens_recovery_count >= 3:  # MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
        return None

    from .query import _create_user_message

    recovery_message = _create_user_message(
        "Output token limit hit. Resume directly — no apology, no recap of what you were doing. "
        "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.",
        is_meta=True,
    )
    new_state = QueryState(
        messages=[*s.messages, *ctx.assistant_messages, recovery_message],
        tool_use_context=s.tool_use_context,
        auto_compact_tracking=s.auto_compact_tracking,
        max_output_tokens_recovery_count=s.max_output_tokens_recovery_count + 1,
        has_attempted_reactive_compact=s.has_attempted_reactive_compact,
        max_output_tokens_override=None,
        stop_hook_active=None,
        turn_count=s.turn_count,
        pending_tool_use_summary=s.pending_tool_use_summary,
        continuation_nudge_count=s.continuation_nudge_count,
        transition=Transition(
            reason="max_output_tokens_recovery",
            attempt=s.max_output_tokens_recovery_count + 1,
        ),
    )
    return (new_state, [])


def _max_output_tokens_exhausted(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """max_output_tokens 恢复次数用尽：yield 错误消息并终止。"""
    if ctx.error_type != "max_output_tokens":
        return None
    s = ctx.state
    if s.max_output_tokens_recovery_count < 3:  # MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
        return None
    return (None, [ctx.last_message] if ctx.last_message is not None else [])


def _collapse_engine_recovery(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """PTL 错误 + CollapseEngine 配置时，优先走引擎恢复路径。"""
    if ctx.error_type != "prompt_too_long":
        return None
    s = ctx.state
    if s.has_attempted_reactive_compact or ctx.params.collapse_engine is None:
        return None

    from clawcodex_ext.services.api.errors import PromptTooLongError

    synthetic_err = PromptTooLongError("synthetic 413: prompt is too long, recovering via CollapseEngine")
    try:
        recovery = ctx.params.collapse_engine.recover_from_413(messages=s.messages, error=synthetic_err)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "CollapseEngine.recover_from_413 raised %r; falling back to reactive_compact",
            exc,
        )
        return None
    if recovery is None or not getattr(recovery, "applied", False):
        return None

    projected = ctx.params.collapse_engine.store.project_view(list(s.messages))
    yield_msgs: list[Message] = list(projected)

    if ctx.params.pipeline_config is not None and ctx.params.pipeline_config.autocompact_tracking is not None:
        ctx.params.pipeline_config.autocompact_tracking.consecutive_failures = 0

    from .transitions import QueryState, Transition

    new_state = QueryState(
        messages=projected,
        tool_use_context=s.tool_use_context,
        auto_compact_tracking=(
            ctx.params.pipeline_config.autocompact_tracking if ctx.params.pipeline_config is not None else None
        ),
        max_output_tokens_recovery_count=s.max_output_tokens_recovery_count,
        has_attempted_reactive_compact=True,
        max_output_tokens_override=None,
        stop_hook_active=s.stop_hook_active,
        turn_count=s.turn_count,
        pending_tool_use_summary=s.pending_tool_use_summary,
        continuation_nudge_count=s.continuation_nudge_count,
        transition=Transition(reason="collapse_engine_retry"),
    )
    return (new_state, yield_msgs)


async def _reactive_compact_recovery(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """PTL/media_size 错误 + reactive_compact 启用时，走 LLM 驱动的压缩恢复。"""
    if ctx.error_type not in ("prompt_too_long", "media_size"):
        return None
    s = ctx.state
    if s.has_attempted_reactive_compact:
        return None
    from .config import FrozenQueryConfig, QueryConfig

    assert isinstance(ctx.config, (QueryConfig, FrozenQueryConfig))
    if not ctx.config.reactive_compact_enabled:
        return None

    from clawcodex_ext.services.api.errors import PromptTooLongError
    from src.services.compact.reactive_compact import reactive_compact

    synthetic_err = PromptTooLongError("withheld during streaming, recovering")
    result = await reactive_compact(
        messages=s.messages,
        error=synthetic_err,
        provider=ctx.params.provider,
        model=ctx.config.model,
    )
    if not result.compacted:
        return (None, [ctx.last_message] if ctx.last_message is not None else [])

    post_compact_messages: list[Message] = result.messages
    yield_msgs: list[Message] = list(post_compact_messages)

    if ctx.params.pipeline_config is not None and ctx.params.pipeline_config.autocompact_tracking is not None:
        ctx.params.pipeline_config.autocompact_tracking.consecutive_failures = 0

    from .transitions import QueryState, Transition

    new_state = QueryState(
        messages=post_compact_messages,
        tool_use_context=s.tool_use_context,
        auto_compact_tracking=(
            ctx.params.pipeline_config.autocompact_tracking if ctx.params.pipeline_config is not None else None
        ),
        max_output_tokens_recovery_count=s.max_output_tokens_recovery_count,
        has_attempted_reactive_compact=True,
        max_output_tokens_override=None,
        stop_hook_active=s.stop_hook_active,
        turn_count=s.turn_count,
        pending_tool_use_summary=s.pending_tool_use_summary,
        continuation_nudge_count=s.continuation_nudge_count,
        transition=Transition(reason="reactive_compact_retry"),
    )
    return (new_state, yield_msgs)


def _media_size_fallback(ctx: RecoveryContext) -> tuple[QueryState | None, list[Message]] | None:
    """media_size 错误且所有恢复策略都已耗尽：终止并返回 image_error。"""
    if ctx.error_type != "media_size":
        return None
    if not ctx.state.has_attempted_reactive_compact:
        return None
    # 返回 None state + 非空 yield 表示终止
    return (None, [ctx.last_message] if ctx.last_message is not None else [])


def _prompt_too_long_fallback(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """PTL 错误且所有恢复策略都已耗尽：终止并返回 prompt_too_long。"""
    if ctx.error_type != "prompt_too_long":
        return None
    if not ctx.state.has_attempted_reactive_compact:
        return None
    return (None, [ctx.last_message] if ctx.last_message is not None else [])


# 注册内置策略（优先级数值越小越靠前）
def _register_builtin_strategies() -> None:
    register_recovery_strategy(MAX_OUTPUT_TOKENS_ESCALATE, _max_output_tokens_escalate, priority=10)
    register_recovery_strategy(MAX_OUTPUT_TOKENS_RECOVERY, _max_output_tokens_recovery, priority=20)
    register_recovery_strategy(MAX_OUTPUT_TOKENS_EXHAUSTED, _max_output_tokens_exhausted, priority=45)
    register_recovery_strategy(COLLAPSE_ENGINE_RECOVERY, _collapse_engine_recovery, priority=30)
    register_recovery_strategy(REACTIVE_COMPACT_RECOVERY, _reactive_compact_recovery, priority=40)
    register_recovery_strategy(MEDIA_SIZE_FALLBACK, _media_size_fallback, priority=100)
    register_recovery_strategy(PROMPT_TOO_LONG_FALLBACK, _prompt_too_long_fallback, priority=100)


_register_builtin_strategies()
