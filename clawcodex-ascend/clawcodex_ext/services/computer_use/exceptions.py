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

"""Computer Use domain exceptions."""


class ComputerUseError(RuntimeError):
    """Base error for Computer Use failures."""


class BinaryNotFoundError(ComputerUseError):
    """Raised when a required system binary (e.g. xdotool, scrot) is missing."""


class SafetyViolationError(ComputerUseError):
    """Raised when an action is blocked by the safety policy / dry-run gate."""


class CoordinatesOutOfBoundsError(ComputerUseError):
    """Raised when a coordinate is outside the validated region."""


class WindowNotFoundError(ComputerUseError):
    """Raised when a window lookup cannot find a matching window."""
