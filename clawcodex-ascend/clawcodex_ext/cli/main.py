#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Downstream CLI entrypoint."""

from __future__ import annotations

import sys


def main():
    """Delegate to the downstream CLI dispatch.

    Triggers the lazy ``ensure_nested_transcript_initialized()`` from
    ``clawcodex_ext.__init__`` so the nested-session transcript path
    resolver is registered before any code that writes transcripts
    runs. See that function's docstring for the circular-import
    reason this lives here, not in the package ``__init__``.
    """
    from clawcodex_ext import ensure_nested_transcript_initialized

    ensure_nested_transcript_initialized()
    from clawcodex_ext.cli.dispatch import run_cli

    return run_cli()


if __name__ == "__main__":
    sys.exit(main())
