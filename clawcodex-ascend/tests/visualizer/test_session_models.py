#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Session model contracts after removal of downloadable exports."""

from extensions.visualizer import models
from extensions.visualizer.models.viz_models import ShareLink


def test_export_format_is_not_part_of_the_public_model_api() -> None:
    assert not hasattr(models, "ExportFormat")


def test_legacy_share_format_is_ignored_without_affecting_share_data() -> None:
    share = ShareLink(
        id="share-1",
        session_id="session-1",
        created_at=1.0,
        expires_at=2.0,
        format="pdf",
        payload={"view": "timeline"},
    )

    assert share.payload == {"view": "timeline"}
    assert "format" not in share.model_dump()
