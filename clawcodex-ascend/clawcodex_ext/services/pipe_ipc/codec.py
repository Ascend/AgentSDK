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

"""JSON Lines codec for Pipe IPC messages."""

from __future__ import annotations

import json

from .models import PipeMessage


class PipeJsonCodec:
    @staticmethod
    def encode_message(message: PipeMessage) -> bytes:
        return (json.dumps(message.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")

    @staticmethod
    def decode_message(raw: bytes | str) -> PipeMessage:
        if isinstance(raw, bytes):
            raw_text = raw.decode("utf-8")
        else:
            raw_text = raw

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid Pipe IPC JSON") from exc

        if not isinstance(data, dict):
            raise TypeError("Pipe IPC message must be a JSON object")
        return PipeMessage.from_dict(data)


def encode_message(message: PipeMessage) -> bytes:
    return PipeJsonCodec.encode_message(message)


def decode_message(raw: bytes | str) -> PipeMessage:
    return PipeJsonCodec.decode_message(raw)
