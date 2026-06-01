#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test-local fixtures for config_cls UTs."""

import sys
import types
from pathlib import Path

import pytest


_CONFIG_CLS_PKG = "aura.trainer.train_adapter.mindspeed_rl.config_cls"


@pytest.fixture(autouse=True, scope="session")
def ensure_aura_src_on_sys_path():
    project_root = Path(__file__).resolve().parents[7]
    aura_src = project_root / "aura"
    aura_src_str = str(aura_src)
    if aura_src_str not in sys.path:
        sys.path.insert(0, aura_src_str)

    yield


@pytest.fixture(autouse=True)
def cleanup_config_cls_mock_pollution_before_test():
    """Remove config_cls-related mocked modules before each test."""
    polluted = [
        name
        for name, mod in list(sys.modules.items())
        if name == _CONFIG_CLS_PKG or name.startswith(f"{_CONFIG_CLS_PKG}.")
        if not isinstance(mod, types.ModuleType)
    ]

    for name in polluted:
        sys.modules.pop(name, None)

    yield
