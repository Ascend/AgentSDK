# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
conftest.py — loaded by pytest before collecting test modules in this directory.

Registers lightweight stubs for the ``rllm`` package (and, when necessary,
``transformers``) in ``sys.modules`` so that the production source files can
be imported without the full third-party dependency tree being installed.
"""

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

# ── torch_npu / device backend guard ─────────────────────────────────────
# ``torch_npu`` is installed but its native extension requires ``libhccl.so``
# which is unavailable in CI/test environments.  Disable torch's automatic
# device-backend loading so that ``import torch`` does not attempt to
# initialise the broken ``torch_npu`` extension.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


# ── rllm stubs ────────────────────────────────────────────────────────────

class _ToolParserStub:
    """Stand-in base class for ``rllm.parser.tool_parser.tool_parser_base.ToolParser``."""
    pass


class _ToolCallStub:
    """Stand-in for ``rllm.tools.tool_base.ToolCall``."""
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _ToolAgentStub:
    """Stand-in for ``rllm.agents.tool_agent.ToolAgent``."""
    _format_observation_as_messages = None


class _ChatTemplateParserStub:
    """Stand-in for ``rllm.parser.chat_template.parser.ChatTemplateParser``."""
    pass


def _make_module(name, parent=None, **attrs):
    """Create a minimal ``ModuleType`` and wire it to its *parent*."""
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    if parent is not None:
        setattr(parent, name.rsplit(".", 1)[-1], mod)
    return mod


if "rllm" not in sys.modules:
    _rllm = _make_module("rllm")
    _rllm_parser = _make_module("rllm.parser", parent=_rllm)
    _rllm_parser_tp = _make_module("rllm.parser.tool_parser", parent=_rllm_parser)
    _ = _make_module("rllm.parser.tool_parser.tool_parser_base", parent=_rllm_parser_tp,
                 ToolParser=_ToolParserStub)
    _rllm_parser_ct = _make_module("rllm.parser.chat_template", parent=_rllm_parser)
    _ = _make_module("rllm.parser.chat_template.parser", parent=_rllm_parser_ct,
                 ChatTemplateParser=_ChatTemplateParserStub)
    _rllm_tools = _make_module("rllm.tools", parent=_rllm)
    _ = _make_module("rllm.tools.tool_base", parent=_rllm_tools, ToolCall=_ToolCallStub)
    _rllm_agents = _make_module("rllm.agents", parent=_rllm)
    _ = _make_module("rllm.agents.tool_agent", parent=_rllm_agents, ToolAgent=_ToolAgentStub)

    for _mod in (
        _rllm, _rllm_parser, _rllm_parser_tp, _rllm_parser_ct,
        _rllm_tools, _rllm_agents,
    ):
        sys.modules.setdefault(_mod.__name__, _mod)
    # Leaf modules
    for _leaf_name in (
        "rllm.parser.tool_parser.tool_parser_base",
        "rllm.parser.chat_template.parser",
        "rllm.tools.tool_base",
        "rllm.agents.tool_agent",
    ):
        sys.modules.setdefault(_leaf_name, getattr(
            sys.modules[_leaf_name.rsplit(".", 1)[0]],
            _leaf_name.rsplit(".", 1)[1],
        ))


# ── torch_npu stub ────────────────────────────────────────────────────────
# ``torch_npu`` is installed but its native extension (``_C``) requires
# ``libhccl.so`` which is unavailable in the CI/test environment.  Stubbing
# the package in ``sys.modules`` prevents the real import from reaching the
# missing shared library.  This must happen *before* ``transformers`` or
# ``sentence_transformers`` are imported, as they probe ``torch_npu``
# availability at module level.

if "torch_npu" not in sys.modules:
    import importlib.machinery

    _torch_npu = _make_module("torch_npu")
    _torch_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", None)
    _torch_npu._C = MagicMock()
    _torch_npu.npu = _make_module("torch_npu.npu", parent=_torch_npu)
    _torch_npu_utils = _make_module("torch_npu.utils", parent=_torch_npu)
    _torch_npu_utils_ec = _make_module("torch_npu.utils._error_code", parent=_torch_npu_utils,
                                       ErrCode=MagicMock(), pta_error=MagicMock())
    for _name, _mod in (
        ("torch_npu", _torch_npu),
        ("torch_npu._C", _torch_npu._C),
        ("torch_npu.npu", _torch_npu.npu),
        ("torch_npu.utils", _torch_npu_utils),
        ("torch_npu.utils._error_code", _torch_npu_utils_ec),
    ):
        sys.modules.setdefault(_name, _mod)


# ── transformers compatibility shim ───────────────────────────────────────
# The installed ``transformers`` may crash on import due to a version
# mismatch with ``torch`` (missing ``register_pytree_node``).  When that
# happens, provide a minimal mock so that type-annotation-only imports
# (``PreTrainedTokenizerBase``, ``AutoTokenizer``) succeed.

if "transformers" not in sys.modules:
    try:
        import transformers  # noqa: F401 – attempt real import
    except (ImportError, AttributeError):
        _transformers = _make_module(
            "transformers",
            PreTrainedTokenizerBase=type("PreTrainedTokenizerBase", (), {}),
            AutoTokenizer=MagicMock(),
        )
        sys.modules["transformers"] = _transformers
