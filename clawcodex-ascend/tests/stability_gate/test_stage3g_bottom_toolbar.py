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

"""Stage 3g — REPL 底部状态条 (``_bottom_toolbar``) 测试（< 2 秒）。

回归守卫
--------
2026-07-04 commit ``0293f5e1`` 在重构 ``_bottom_toolbar`` 时:

- 删除了 ``goal_part`` 的整个定义块(初始化 + try/except +
  ``clawcodex_ext.goal.registry`` 引用),但 ``return`` 内的
  ``f"{goal_part}"`` 引用忘了删 —— 每次 redraw 抛 ``NameError``,
  被顶层 ``except Exception: return ''`` 兜底吞掉,状态条整个消失

此门禁做两层防护:

1. **静态源扫描**: 抓 ``_bottom_toolbar`` 函数体内出现的
   ``f"{NAME}"`` 插值,要求每个 ``NAME`` 都在函数内被赋值。孤儿
   引用一出现就 fail,根本不等到运行时。
2. **运行时烟雾**: 用 stub ``self`` + ``_load_heavy_runtime()`` 跑
   一次 ``_bottom_toolbar``,断言不抛 ``NameError`` 且返回非空。

覆盖:

- 静态源扫描: ``goal_part`` 引用检测、``f"{NAME}"`` 完整性扫描
- 运行时: 正常渲染、零 advisor / cost 隐藏、非零 advisor 显示、
  ``tool_context`` 缺失兜底、未知名模型降级
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _heavy_runtime():
    """模块级 fixture — 一次性加载重型运行时。

    ``_load_heavy_runtime()`` 触发 21+ 个 clawcodex_ext 扩展和上游
    provider/tool/session 模块的延迟 import。多次调用收益为零,
    所以挂在 module scope 上让 9 个测试只触发一次。
    """
    from src.repl.core import ClawcodexREPL, _load_heavy_runtime

    _load_heavy_runtime()
    return ClawcodexREPL


class TestStage3gBottomToolbarSource:
    """源代码静态回归守卫。"""

    def test_no_orphan_goal_part_reference(self):
        """``_bottom_toolbar`` 不得保留孤儿 ``goal_part`` 引用。

        2026-07-04 commit ``0293f5e1`` 删除了 ``goal_part`` 的整个
        定义块但保留了 ``f"{goal_part}"`` 引用,触发 NameError 被
        except 吞掉的状态条消失事故。守卫的语义是:若 ``goal_part``
        在 f-string 中被引用,本地必须有对应的赋值。如果未来 commit
        再次引入同样的孤儿引用,在此处直接 fail。
        """
        import inspect
        import re

        from src.repl.core import ClawcodexREPL

        src = inspect.getsource(ClawcodexREPL._bottom_toolbar)
        # Only flag orphan when ``goal_part`` is referenced in an f-string
        # but has no local assignment. Using it as a regular variable name
        # outside f-strings (e.g. as a plain string concat) is allowed and
        # never raises NameError.
        goal_part_in_fstring = bool(re.search(r'f"\{goal_part\}"', src))
        goal_part_assigned = bool(re.search(r"\bgoal_part\s*(?::\s*\S+)?\s*=", src))
        assert not (goal_part_in_fstring and not goal_part_assigned), (
            "_bottom_toolbar references 'goal_part' in an f-string but "
            "the local assignment was removed (orphan variable regression, "
            "see commit 0293f5e1). Either restore the assignment or "
            "delete the f-string reference."
        )

    def test_all_interpolated_vars_are_assigned(self):
        """所有 ``f"{NAME}"`` 插值变量都必须在函数体内被赋值。

        这是 generic 版"孤儿引用"扫描,不只防 ``goal_part``,也防
        未来任何重构时只删定义不删引用的回归。
        """
        import inspect
        import re

        from src.repl.core import ClawcodexREPL

        src = inspect.getsource(ClawcodexREPL._bottom_toolbar)
        refs = set(re.findall(r'f"\{([a-z_][a-z_0-9]*)\}"', src))
        # Skip ``f"{NAME:FORMAT}"`` style with explicit format spec.
        refs = {r for r in refs if ":" not in r}
        for ref in refs:
            # Accept any local assignment shape: NAME = ..., NAME=..., NAME: T = ...
            assigned = bool(re.search(rf"\b{re.escape(ref)}\s*(?::\s*\S+)?\s*=", src))
            assert assigned, (
                f"_bottom_toolbar interpolates f'{{{ref}}}' but never "
                f"assigns it locally — orphan variable reference. Either "
                f"restore the definition or delete the f-string reference."
            )


class TestStage3gBottomToolbarRuntime:
    """运行时烟雾测试 — stub self 调用 ``_bottom_toolbar``。"""

    @staticmethod
    def _make_stub(
        *,
        provider_name="anthropic",
        model="claude-sonnet-4-6",
        cwd="/tmp",
        turns=0,
        in_tokens=0,
        out_tokens=0,
        advisor_in=0,
        advisor_out=0,
    ):
        class _Stub:
            pass

        stub = _Stub()
        stub.provider = type(
            "P",
            (),
            {"provider_name": provider_name, "model": model},
        )()
        stub.provider_name = provider_name
        stub.tool_context = type(
            "T",
            (),
            {
                "cwd": cwd,
                "workspace_root": cwd,
                "advisor_input_tokens": advisor_in,
                "advisor_output_tokens": advisor_out,
                "tasks": {},
            },
        )()
        stub._permission_mode = "default"
        stub._stats_turns = turns
        stub._stats_input_tokens = in_tokens
        stub._stats_output_tokens = out_tokens
        stub._shorten_path_text = staticmethod(lambda p: p)
        # ``_bottom_toolbar`` calls ``self._goal_footer_status()``; without
        # a no-op binding the call raises AttributeError on the stub and
        # the outer ``except Exception`` swallows it, returning "".
        stub._goal_footer_status = lambda: None
        stub._goal_footer_id = None
        stub._goal_footer_started_at = None
        stub._task_toolbar_part = lambda: ""
        return stub

    def test_renders_non_empty_string(self, _heavy_runtime):
        """正常渲染: 返回非空字符串,包含 provider / model / cwd。"""
        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub()
        result = ClawcodexREPL._bottom_toolbar(stub)
        assert result, f"expected non-empty status bar, got {result!r}"
        assert "anthropic" in result
        assert "claude-sonnet-4-6" in result
        assert "/tmp" in result

    def test_no_name_error_on_render(self, _heavy_runtime):
        """各种权限模式下都不抛 NameError。

        这是 ``goal_part`` 事故的核心症状:任何孤儿引用都会在
        f-string 求值时抛 ``NameError``。把 ``except Exception``
        当作 catch-all 兜底时,这种故障对用户完全不可见 —— 所以
        我们在门禁里直接 assert 调用能正常返回非空字符串。
        """
        ClawcodexREPL = _heavy_runtime
        for perm_mode in ("default", "plan", "acceptEdits", "bypassPermissions"):
            stub = self._make_stub()
            stub._permission_mode = perm_mode
            try:
                result = ClawcodexREPL._bottom_toolbar(stub)
            except NameError as e:
                raise AssertionError(
                    f"_bottom_toolbar raised NameError({e}) under "
                    f"permission_mode={perm_mode!r} — orphan variable "
                    f"reference regression (see commit 0293f5e1)."
                ) from e
            assert result, (
                f"permission_mode={perm_mode!r}: got empty result, "
                f"likely a silent failure swallowed by except Exception"
            )

    def test_zero_advisor_hides_advisor_part(self, _heavy_runtime):
        """零 advisor token 时不渲染 advisor part。"""
        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub(advisor_in=0, advisor_out=0)
        result = ClawcodexREPL._bottom_toolbar(stub)
        assert "advisor" not in result, f"advisor tokens are 0 but result contains 'advisor': {result!r}"

    def test_nonzero_advisor_renders_advisor_part(self, _heavy_runtime):
        """非零 advisor token 时渲染 advisor: X in / Y out。"""
        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub(advisor_in=1234, advisor_out=567)
        result = ClawcodexREPL._bottom_toolbar(stub)
        assert "advisor" in result, f"advisor tokens are non-zero but result lacks 'advisor': {result!r}"

    def test_zero_cost_hides_cost_part(self, _heavy_runtime):
        """零 token 时不渲染 cost part。"""
        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub(in_tokens=0, out_tokens=0)
        result = ClawcodexREPL._bottom_toolbar(stub)
        assert "cost" not in result, f"tokens are 0 but result contains 'cost': {result!r}"

    def test_missing_tool_context_returns_empty(self, _heavy_runtime):
        """异常韧性: ``tool_context`` 缺失时不崩溃,返回空字符串。

        ``_bottom_toolbar`` 顶层 ``except Exception: return ''`` 保证
        即使依赖异常,REPL 输入框也不会被打断。此测试固定这个语义。
        """
        ClawcodexREPL = _heavy_runtime

        class _BareStub:
            provider = None
            provider_name = "anthropic"
            tool_context = None
            _permission_mode = "default"
            _stats_turns = 0
            _stats_input_tokens = 0
            _stats_output_tokens = 0
            _shorten_path_text = staticmethod(lambda p: p)

        result = ClawcodexREPL._bottom_toolbar(_BareStub())
        assert result == "", f"expected empty string when tool_context is None, got {result!r}"

    def test_unknown_model_renders_without_crash(self, _heavy_runtime):
        """未知名模型不崩,降级渲染(模型上下文窗口查询静默失败)。"""
        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub(
            provider_name="anthropic",
            model="some-future-unknown-model-2099",
        )
        result = ClawcodexREPL._bottom_toolbar(stub)
        assert result, f"expected non-empty result for unknown model, got {result!r}"
        assert "anthropic" in result

    def test_task_progress_is_persistent_in_toolbar(self, _heavy_runtime):
        """Task progress remains visible without requiring another TaskList call."""

        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub()
        stub.tool_context.tasks = {
            "T-1": {"id": "T-1", "status": "completed", "metadata": {}},
            "T-2": {"id": "T-2", "status": "in_progress", "metadata": {}},
            "T-3": {
                "id": "T-3",
                "status": "pending",
                "metadata": {},
                "lkb": {"derivedStatus": "blocked"},
            },
        }
        stub._task_toolbar_part = lambda: ClawcodexREPL._task_toolbar_part(stub)

        result = ClawcodexREPL._bottom_toolbar(stub)

        assert "tasks: 1/3" in result
        assert "1 running" in result
        assert "1 blocked" in result
