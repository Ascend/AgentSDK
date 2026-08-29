#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Stable error codes for the Phase 4 macro convert."""

from __future__ import annotations


class MacroConvertError(ValueError):
    """Validation / convert failure with a stable machine-readable code."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        manifest: str = "",
        step_id: str = "",
        field: str = "",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.manifest = manifest
        self.step_id = step_id
        self.field = field

    def to_dict(self) -> dict[str, str]:
        payload = {
            "error_code": self.error_code,
            "message": str(self),
        }
        if self.manifest:
            payload["manifest"] = self.manifest
        if self.step_id:
            payload["step_id"] = self.step_id
        if self.field:
            payload["field"] = self.field
        return payload
