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

"""Tests for minimax live."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("CLAWCODEX_RUN_EXTERNAL_TESTS") != "1",
    reason="set CLAWCODEX_RUN_EXTERNAL_TESTS=1 to run live MiniMax API tests",
)


def _check_key() -> str:
    """Test helper for check key."""
    key = os.environ.get("MINIMAX_API_KEY")
    if key:
        return key
    creds_path = Path("~/.clawcodex/tts/minimax/credentials.json").expanduser()
    if creds_path.is_file():
        try:
            data = json.loads(creds_path.read_text(encoding="utf-8"))
            return data.get("api_key", "")
        except (json.JSONDecodeError, OSError):
            pass  # Invalid stored credentials fall through to the setup error.
    sys.exit(
        "❌ MINIMAX_API_KEY 未设置。运行:\n"
        "  export MINIMAX_API_KEY='your-key-here'\n"
        "或创建 ~/.clawcodex/tts/minimax/credentials.json"
    )


def _get_group_id() -> str:
    gid = os.environ.get("MINIMAX_GROUP_ID", "")
    if gid:
        return gid
    creds_path = Path("~/.clawcodex/tts/minimax/credentials.json").expanduser()
    if creds_path.is_file():
        try:
            data = json.loads(creds_path.read_text(encoding="utf-8"))
            return data.get("group_id", "")
        except Exception:
            pass  # Missing optional credentials leave the group ID unset.
    return ""


AVAILABLE_VOICES = {
    "中文女声": "Chinese (Mandarin)_Warm_Girl",
    "中文男声": "Chinese (Mandarin)_Gentleman",
    "中文甜嗓": "Chinese (Mandarin)_Sweet_Lady",
    "中文新闻": "Chinese (Mandarin)_News_Anchor",
    "中文可爱": "Chinese (Mandarin)_Cute_Spirit",
    "英文叙述": "English_expressive_narrator",
    "英文阳光": "English_radiant_girl",
    "英文男声": "English_magnetic_voiced_man",
    "日文知性": "Japanese_IntellectualSenior",
    "粤语温柔": "Cantonese_GentleLady",
}

AVAILABLE_MODELS = [
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
]

TEST_TEXTS = {
    "中文": "今天天气真不错(sighs)，阳光明媚，微风拂面。我们一起去公园散步吧！",
    "英文": "Hello(sighs), this is a test of the MiniMax text-to-speech API. The weather is beautiful today!",
    "中英混合": "大家好，欢迎使用 MiniMax TTS API。This is a mixed language test.",
    "粤语": "各位早晨，今日天氣真好(sighs)，我哋一齊去飲茶啦！",
}


def test_tts_synthesize():
    """Verify tts synthesize."""
    print("\n" + "=" * 70)
    print("🧪 MiniMax TTS 测试 — HTTP 合成 (synthesize)")
    print("=" * 70)

    key = _check_key()
    api_key = key
    group_id = _get_group_id()
    endpoint = os.environ.get("MINIMAX_ENDPOINT") or "https://api.minimax.io/v1/t2a_v2"

    print(f"端点: {endpoint}")
    print(f"GroupId: {group_id or '(未设置)'}")
    print()

    print("--- 1/3: 测试模型切换 ---")
    for model in ("speech-2.8-turbo", "speech-2.8-hd"):
        print(f"  模型: {model} ... ", end="", flush=True)
        payload = json.dumps(
            {
                "model": model,
                "text": "你好，这是一个测试。",
                "stream": False,
                "language_boost": "Chinese",
                "voice_setting": {
                    "voice_id": "Chinese (Mandarin)_Warm_Girl",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {
                    "sample_rate": 24000,
                    "format": "pcm",
                    "channel": 1,
                },
            }
        ).encode("utf-8")

        import urllib.request

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            base = resp_data.get("base_resp", {})
            if base.get("status_code") == 0:
                audio_hex = resp_data.get("data", {}).get("audio", "")
                pcm_len = len(bytes.fromhex(audio_hex)) if audio_hex else 0
                extra = resp_data.get("extra_info", {})
                print(f"✅ {pcm_len / 1000:.0f}KB PCM, {extra.get('word_count', '?')}字符")
            else:
                print(f"❌ {base.get('status_msg')}")
        except Exception as e:
            print(f"❌ {e}")

    print("\n--- 2/3: 测试音色切换 ---")
    for name, voice_id in list(AVAILABLE_VOICES.items())[:6]:
        print(f"  音色 [{name}] ({voice_id[:30]}...) ... ", end="", flush=True)
        payload = json.dumps(
            {
                "model": "speech-2.8-turbo",
                "text": "今天天气真不错，阳光明媚，微风拂面。",
                "stream": False,
                "language_boost": "Chinese",
                "voice_setting": {
                    "voice_id": voice_id,
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {
                    "sample_rate": 24000,
                    "format": "pcm",
                    "channel": 1,
                },
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            base = resp_data.get("base_resp", {})
            if base.get("status_code") == 0:
                audio_hex = resp_data.get("data", {}).get("audio", "")
                pcm_len = len(bytes.fromhex(audio_hex)) if audio_hex else 0
                print(f"✅ {pcm_len / 1000:.0f}KB")
            else:
                print(f"❌ {base.get('status_msg')}")
        except Exception as e:
            print(f"❌ {e}")

    print("\n--- 3/3: 测试多语言 ---")
    texts_to_test = [
        ("中文", "Chinese", "Chinese (Mandarin)_Warm_Girl"),
        ("英文", "English", "English_expressive_narrator"),
        ("中英混合", "auto", "Chinese (Mandarin)_Warm_Girl"),
    ]
    for lang_name, lang_boost, voice_id in texts_to_test:
        text = TEST_TEXTS.get(lang_name, "你好。")
        print(f"  语言 [{lang_name}] language_boost={lang_boost} ... ", end="", flush=True)
        payload = json.dumps(
            {
                "model": "speech-2.8-turbo",
                "text": text,
                "stream": False,
                "language_boost": lang_boost,
                "voice_setting": {
                    "voice_id": voice_id,
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {
                    "sample_rate": 24000,
                    "format": "pcm",
                    "channel": 1,
                },
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            base = resp_data.get("base_resp", {})
            if base.get("status_code") == 0:
                audio_hex = resp_data.get("data", {}).get("audio", "")
                pcm_len = len(bytes.fromhex(audio_hex)) if audio_hex else 0
                extra = resp_data.get("extra_info", {})
                print(f"✅ {pcm_len / 1000:.0f}KB, {extra.get('word_count', '?')}字符")
            else:
                print(f"❌ {base.get('status_msg')}")
        except Exception as e:
            print(f"❌ {e}")

    print("\n✅ TTS 测试完成!")


def test_tts_streaming():
    """Verify tts streaming."""
    print("\n" + "=" * 70)
    print("🧪 MiniMax TTS 测试 — 流式输出 (streaming)")
    print("=" * 70)

    key = _check_key()
    group_id = _get_group_id()
    endpoint = os.environ.get("MINIMAX_ENDPOINT") or "https://api.minimax.io/v1/t2a_v2"

    print(f"端点: {endpoint}")
    print()

    import urllib.request

    text = (
        "大家好(sighs)，欢迎使用 MiniMax 语音合成技术。今天我们为大家带来一段流式输出的测试。(laughs)希望效果令人满意。"
    )

    payload = json.dumps(
        {
            "model": "speech-2.8-hd",
            "text": text,
            "stream": True,
            "language_boost": "Chinese",
            "voice_setting": {
                "voice_id": "Chinese (Mandarin)_Warm_Girl",
                "speed": 1.0,
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 24000,
                "format": "pcm",
                "channel": 1,
            },
        }
    ).encode("utf-8")

    url = endpoint
    if group_id:
        url = f"{url}?GroupId={group_id}" if "?" not in url else f"{url}&GroupId={group_id}"

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print("发送流式请求 ...")
    total_pcm = bytearray()
    chunk_count = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw_line in resp:
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = obj.get("data") or {}
                hex_audio = data.get("audio") if isinstance(data, dict) else None
                if hex_audio:
                    try:
                        pcm = bytes.fromhex(hex_audio)
                        total_pcm.extend(pcm)
                        chunk_count += 1
                        print(f"  chunk#{chunk_count}: {len(pcm)} bytes", flush=True)
                    except ValueError:
                        print(f"  chunk#{chunk_count}: ⚠️ non-hex frame")
                if obj.get("is_final"):
                    extra = obj.get("extra_info", {})
                    print(f"  ✅ 流式完成: {chunk_count} chunks, {len(total_pcm) / 1000:.0f}KB 总PCM")
                    print(f"     字符数: {extra.get('word_count', '?')}, 音频时长: {extra.get('audio_length', '?')}ms")
    except Exception as e:
        print(f"❌ 流式请求失败: {e}")

    print("✅ 流式测试完成!")


def test_stt_realtime():
    """Verify stt realtime."""
    print("\n" + "=" * 70)
    print("🧪 MiniMax STT 测试 — Realtime API (voice-in → text-out)")
    print("=" * 70)

    try:
        import asyncio
        import base64
        import websockets
    except ImportError:
        print("❌ 缺少依赖: pip install websockets")
        return

    key = _check_key()
    group_id = _get_group_id()
    region = os.environ.get("MINIMAX_REGION", "global")
    endpoints = {
        "global": "wss://api.minimax.io/ws/realtime",
        "cn": "wss://api.minimaxi.chat/ws/realtime",
    }
    ws_url = endpoints.get(region, endpoints["global"])
    if group_id:
        ws_url = f"{ws_url}?group_id={group_id}"

    async def _test():
        print(f"端点: {ws_url}")
        print(f"Region: {region}")
        print()

        print("--- 1/4: 建立 WebSocket 连接 ---")
        try:
            ws = await websockets.connect(
                ws_url,
                additional_headers={"Authorization": f"Bearer {key}"},
                ping_interval=None,
            )
            print("✅ WebSocket 连接成功")
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return

        print("\n--- 2/4: session.create → 文本模态 ---")
        session_msg = {
            "type": "session.create",
            "session": {
                "model": "speech-2.8-turbo",
                "modalities": ["text"],
                "input_audio_format": "pcm16",
                "sample_rate": 16000,
            },
        }
        await ws.send(json.dumps(session_msg))
        print("  发送 session.create")

        print("\n--- 3/4: 等待服务器事件 (5s 超时) ---")
        try:
            for i in range(5):
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                payload = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
                msg_type = payload.get("type") or payload.get("event", "?")
                print(f"  收到事件 [{i + 1}]: {msg_type}")
                if msg_type == "session.created" or "ready" in str(msg_type).lower():
                    break
        except asyncio.TimeoutError:
            print("⏱️ 超时（事件名可能与预期不同 — 已记录到 _handle_message 双轨兼容）")

        print("\n--- 4/4: 模拟音频输入 (合成静音帧) ---")
        import struct

        silent_frame = struct.pack(f"<{8000}h", *([0] * 8000))
        encoded = base64.b64encode(silent_frame).decode("ascii")

        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": encoded,
                }
            )
        )
        print("  ✅ 发送 input_audio_buffer.append (0.5s 静音帧)")

        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.commit",
                }
            )
        )
        print("  ✅ 发送 input_audio_buffer.commit")

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            payload = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
            print(f"  服务器响应: type={payload.get('type', '?')}")
        except asyncio.TimeoutError:
            print("⏱️ 无响应（预期：因无真实音频输入）")

        await ws.close()
        print("\n✅ STT 连接测试完成!")
        print("\n⚠️ 注意: 上述测试仅验证 WebSocket 握手 + session.create 协议流。")
        print("   完整语音转录需要真实麦克风输入 + push-to-talk 控制器。")
        print("   协议事件名差异在 `minimax_stt._handle_message` 中收敛。")

    asyncio.run(_test())


def list_voices():
    """Test helper for list voices."""
    print("\n" + "=" * 70)
    print("🎤 MiniMax 系统音色列表 (官方 332+ 音色)")
    print("=" * 70)
    print()
    print("中文 (普通话):")
    for v in [
        "Chinese (Mandarin)_Warm_Girl",
        "Chinese (Mandarin)_Gentleman",
        "Chinese (Mandarin)_News_Anchor",
        "Chinese (Mandarin)_Mature_Woman",
        "Chinese (Mandarin)_Sweet_Lady",
        "Chinese (Mandarin)_Crisp_Girl",
        "Chinese (Mandarin)_Reliable_Executive",
        "Chinese (Mandarin)_Male_Announcer",
        "Chinese (Mandarin)_Cute_Spirit",
        "Chinese (Mandarin)_Soft_Girl",
        "Chinese (Mandarin)_Warm_Bestie",
        "Chinese (Mandarin)_Lyrical_Voice",
        "Chinese (Mandarin)_Gentle_Youth",
        "Chinese (Mandarin)_Kind-hearted_Elder",
        "Chinese (Mandarin)_HK_Flight_Attendant",
    ]:
        print(f"  ✅ {v}")

    print()
    print("英文 (English):")
    for v in [
        "English_expressive_narrator",
        "English_radiant_girl",
        "English_magnetic_voiced_man",
        "English_captivating_female1",
        "English_Graceful_Lady",
        "English_Persuasive_Man",
        "English_CalmWoman",
        "English_FriendlyPerson",
        "English_Diligent_Man",
        "English_ManWithDeepVoice",
    ]:
        print(f"  ✅ {v}")

    print()
    print("日文 (Japanese):")
    for v in [
        "Japanese_IntellectualSenior",
        "Japanese_GentleButler",
        "Japanese_KindLady",
        "Japanese_CalmLady",
    ]:
        print(f"  ✅ {v}")

    print()
    print("粤语 (Cantonese):")
    for v in [
        "Cantonese_GentleLady",
        "Cantonese_ProfessionalHost (F)",
        "Cantonese_ProfessionalHost (M)",
        "Cantonese_CuteGirl",
    ]:
        print(f"  ✅ {v}")

    print()
    print("推荐模型:")
    print("  speech-2.8-hd    — 最新旗舰，含语气词标签 (laughs)(sighs)(coughs)")
    print("  speech-2.8-turbo — 最新高速度版")
    print("  speech-2.6-hd    — 超低延时版")
    print("  speech-2.6-turbo — 极速经济版")
    print()
    print("完整音色列表: https://platform.minimax.io/docs/faq/system-voice-id")


def test_clawcodex_tts():
    """Verify clawcodex tts."""
    print("\n" + "=" * 70)
    print("🧪 clawcodex MiniMaxTTSProvider 接口测试")
    print("=" * 70)

    os.environ["MINIMAX_API_KEY"] = _check_key()

    from clawcodex_ext.services.voice.minimax_tts import (
        MiniMaxTTSProvider,
    )

    provider = MiniMaxTTSProvider()

    print("\n--- 1/4: 凭证解析 ---")
    try:
        api_key, group_id, endpoint = provider._resolve()
        masked = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "(短key)"
        print(f"  API Key: {masked}")
        print(f"  GroupId: {group_id or '(未设置)'}")
        print(f"  Endpoint: {endpoint}")
        print("  ✅ 凭证解析成功")
    except Exception as e:
        print(f"  ❌ {e}")
        return

    print("\n--- 2/4: synthesize (batch 路径) ---")
    from clawcodex_ext.services.voice.tts import TTSConfig
    import asyncio

    cfg = TTSConfig(
        model="speech-2.8-turbo",
        voice="Chinese (Mandarin)_Warm_Girl",
        language="Chinese",
    )

    async def _batch_test():
        try:
            pcm = await provider.synthesize("你好，这是一个测试。今天天气真不错！", cfg)
            if pcm and len(pcm) > 0:
                print(f"  ✅ 合成成功: {len(pcm) / 1000:.0f}KB PCM ({len(pcm)} bytes)")
            else:
                print("  ❌ 合成返回空数据")
        except Exception as e:
            print(f"  ❌ 合成失败: {e}")

    asyncio.run(_batch_test())

    print("\n--- 3/4: synthesize_stream (流式路径) ---")

    async def _stream_test():
        audios: list[tuple[bytes, bool]] = []
        errors: list[str] = []
        done = []

        def on_audio(chunk):
            audios.append((chunk.pcm, chunk.is_final))
            print(f"    收到音频帧: {len(chunk.pcm)} bytes, final={chunk.is_final}", flush=True)

        def on_error(msg):
            errors.append(msg)

        def on_done():
            done.append(True)

        syn = provider.synthesize_stream(
            on_audio=on_audio,
            on_error=on_error,
            on_done=on_done,
            config=cfg,
        )
        await syn.feed_text("大家好(sighs)，欢迎使用 MiniMax 语音合成技术。")
        import asyncio

        await asyncio.sleep(2)
        if done:
            total = sum(len(c[0]) for c in audios)
            print(f"  ✅ 流式完成: {len(audios)} frames, {total / 1000:.0f}KB")
        elif errors:
            print(f"  ❌ 流式错误: {errors[0]}")
        else:
            print("  ⏱️ 异步任务未完成（标准行为 — provider 等待 feed_text 关闭后触发 POST）")

    asyncio.run(_stream_test())

    print("\n--- 4/4: 内置音色索引 ---")
    from clawcodex_ext.services.voice.minimax_tts import MINIMAX_SYSTEM_VOICES

    for lang, voices in MINIMAX_SYSTEM_VOICES.items():
        print(f"  {lang}: {len(voices)} 个音色")
    print("  ✅ 音色索引就绪")

    print("\n✅ clawcodex MiniMaxTTSProvider 接口测试完成!")


def test_with_real_api():
    """Verify with real api."""
    print("\n" + "=" * 70)
    print("🧪 MiniMax T2A API 端到端测试")
    print("=" * 70)

    key = _check_key()
    group_id = _get_group_id()
    endpoint = os.environ.get("MINIMAX_ENDPOINT") or "https://api.minimax.io/v1/t2a_v2"

    url = endpoint
    if group_id:
        url = f"{url}?GroupId={group_id}" if "?" not in url else f"{url}&GroupId={group_id}"

    print(f"端点: {url}")
    print()

    import urllib.request

    test_cases = [
        (
            "中文 Warm_Girl",
            {
                "model": "speech-2.8-turbo",
                "text": "你好，很高兴认识你。今天是个好天气。",
                "stream": False,
                "language_boost": "Chinese",
                "voice_setting": {
                    "voice_id": "Chinese (Mandarin)_Warm_Girl",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {"sample_rate": 24000, "format": "pcm", "channel": 1},
            },
        ),
        (
            "中文 Gentleman",
            {
                "model": "speech-2.8-hd",
                "text": "各位好(sighs)，欢迎参加今天的会议。让我们开始吧。",
                "stream": False,
                "language_boost": "Chinese",
                "voice_setting": {
                    "voice_id": "Chinese (Mandarin)_Gentleman",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {"sample_rate": 24000, "format": "pcm", "channel": 1},
            },
        ),
        (
            "英文 expressive_narrator",
            {
                "model": "speech-2.8-turbo",
                "text": "Hello(breath), welcome to the MiniMax TTS API. This is a quick test of the system voices.",
                "stream": False,
                "language_boost": "English",
                "voice_setting": {
                    "voice_id": "English_expressive_narrator",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {"sample_rate": 24000, "format": "pcm", "channel": 1},
            },
        ),
        (
            "日文 IntellectualSenior",
            {
                "model": "speech-2.8-turbo",
                "text": "こんにちは、今日はいい天気ですね。",
                "stream": False,
                "language_boost": "Japanese",
                "voice_setting": {
                    "voice_id": "Japanese_IntellectualSenior",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {"sample_rate": 24000, "format": "pcm", "channel": 1},
            },
        ),
        (
            "粤语 GentleLady",
            {
                "model": "speech-2.8-hd",
                "text": "早晨(sighs)，今日天氣幾好，我哋一齊去食早餐啦！",
                "stream": False,
                "language_boost": "Chinese,Yue",
                "voice_setting": {
                    "voice_id": "Cantonese_GentleLady",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {"sample_rate": 24000, "format": "pcm", "channel": 1},
            },
        ),
        (
            "语气词表情测试 HD",
            {
                "model": "speech-2.8-hd",
                "text": "哇(laughs)，这个真的太厉害了！(sighs) 不过我们还是小心一点比较好。",
                "stream": False,
                "language_boost": "Chinese",
                "voice_setting": {
                    "voice_id": "Chinese (Mandarin)_Warm_Girl",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {"sample_rate": 24000, "format": "pcm", "channel": 1},
            },
        ),
        (
            "高速模型 2.6-turbo",
            {
                "model": "speech-2.6-turbo",
                "text": "你好，这是一个快速测试。",
                "stream": False,
                "language_boost": "Chinese",
                "voice_setting": {
                    "voice_id": "Chinese (Mandarin)_Crisp_Girl",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {"sample_rate": 24000, "format": "pcm", "channel": 1},
            },
        ),
    ]

    passed = 0
    failed = 0
    for label, payload in test_cases:
        print(f"[{label}] ... ", end="", flush=True)
        t0 = time.time()
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            base = resp_data.get("base_resp", {})
            if base.get("status_code") == 0:
                audio_hex = resp_data.get("data", {}).get("audio", "")
                pcm_len = len(bytes.fromhex(audio_hex)) if audio_hex else 0
                elapsed = time.time() - t0
                extra = resp_data.get("extra_info", {})
                print(f"✅ {pcm_len / 1000:.0f}KB PCM, {elapsed:.1f}s (chars={extra.get('word_count', '?')})")
                passed += 1
            else:
                print(f"❌ {base.get('status_msg')}")
                failed += 1
        except Exception as e:
            print(f"❌ {e}")
            failed += 1

    print(f"\n结果: {passed} passed, {failed} failed (共 {len(test_cases)} 用例)")
    if passed == len(test_cases):
        print("🎉 全部通过!")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMax 语音模型测试")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tts", action="store_true", help="仅测试 TTS HTTP 合成")
    group.add_argument("--stt", action="store_true", help="仅测试 STT Realtime 连接")
    group.add_argument("--list", action="store_true", help="列出可用音色")
    group.add_argument("--api", action="store_true", help="端到端 API 测试")
    group.add_argument("--clawcodex", action="store_true", help="通过 clawcodex 接口测试")
    args = parser.parse_args()

    if args.list:
        list_voices()
    elif args.tts:
        test_tts_synthesize()
        test_tts_streaming()
    elif args.stt:
        test_stt_realtime()
    elif args.api:
        test_with_real_api()
    elif args.clawcodex:
        test_clawcodex_tts()
    else:
        print("🎤 MiniMax 语音模型测试套件")
        print("=" * 50)
        print("1) 端到端 API 测试 (T2A HTTP)")
        print("2) TTS 合成 + 流式测试")
        print("3) clawcodex MiniMaxTTSProvider 接口测试")
        print("4) STT Realtime 连接测试 (需 websockets)")
        print("5) 列出可用音色")
        print("q) 退出")
        print("=" * 50)
        choice = input("请选择 (1-5): ").strip()
        if choice == "1":
            test_with_real_api()
        elif choice == "2":
            test_tts_synthesize()
            test_tts_streaming()
        elif choice == "3":
            test_clawcodex_tts()
        elif choice == "4":
            test_stt_realtime()
        elif choice == "5":
            list_voices()
        else:
            print("退出.")
