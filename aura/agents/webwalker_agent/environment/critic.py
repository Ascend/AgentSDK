#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class WebWalkerCriticMixin:
    def _empty_critic_metadata(self) -> dict:
        return {
            "critic_useful_information": [],
            "critic_suggested_answer": "",
            "critic_early_stop_enabled": self.enable_critic_early_stop,
            "webwalker_memory_snapshot": [],
            "critic_traces": [],
        }

    def _build_critic_trace(
        self,
        *,
        stage: str,
        messages: list[dict],
        raw_response_text: str = "",
        parsed_response: dict | None = None,
        result: str | None = None,
        error: str = "",
        skipped: bool = False,
    ) -> dict:
        return {
            "stage": stage,
            "model": self.model,
            "messages": messages,
            "raw_response_text": raw_response_text,
            "parsed_response": parsed_response,
            "result": result,
            "error": error,
            "skipped": skipped,
        }

    def _count_text_tokens(self, text: str) -> int | None:
        if self.tokenizer is None:
            return None
        try:
            return len(self.tokenizer.encode(str(text or ""), add_special_tokens=False))
        except Exception:
            return None

    def _dump_critic_failure(
        self,
        *,
        stage: str,
        query: str,
        observation: str,
        messages: list[dict],
        error: Exception,
    ) -> str:
        if not self.critic_failure_dump_dir:
            return ""

        try:
            output_dir = Path(self.critic_failure_dump_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            user_prompt = ""
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    user_prompt = str(message.get("content") or "")
                    break
            system_prompt = ""
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "system":
                    system_prompt = str(message.get("content") or "")
                    break

            payload = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "stage": stage,
                "critic_url": self.api_url,
                "critic_model": self.model,
                "critic_temperature": self.chat_model_temperature,
                "critic_top_p": self.chat_model_top_p,
                "critic_max_tokens": self.chat_model_max_tokens,
                "root_url": self.root_url,
                "current_page_url": self.current_page_url,
                "question": query,
                "error_type": type(error).__name__,
                "error": str(error),
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "observation_chars": len(str(observation or "")),
                "messages_text_chars": len(system_prompt) + len(user_prompt),
                "system_prompt_tokens": self._count_text_tokens(system_prompt),
                "user_prompt_tokens": self._count_text_tokens(user_prompt),
                "observation_tokens": self._count_text_tokens(str(observation or "")),
                "messages": messages,
                "observation": observation,
            }
            digest_src = f"{stage}\n{self.root_url}\n{self.current_page_url}\n{query}\n{time.time()}"
            digest = hashlib.sha1(digest_src.encode("utf-8", errors="ignore")).hexdigest()[:12]
            output_path = output_dir / f"critic_failure_{time.strftime('%Y%m%d_%H%M%S')}_{digest}.json"
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            return str(output_path)
        except Exception as dump_error:
            logger.warning(f"Failed to write critic failure dump: {dump_error}")
            return ""

    def _critic_request_kwargs(self, messages: list[dict]) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.chat_model_temperature,
            "top_p": self.chat_model_top_p,
            "max_tokens": self.chat_model_max_tokens,
        }
        return request_kwargs

    def _parse_critic_json(self, raw_response_text: str) -> dict[str, Any]:
        text = str(raw_response_text or "").strip()
        if not text:
            raise ValueError("empty critic response")

        try:
            content = json.loads(text)
            if isinstance(content, dict):
                return content
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced_match:
            try:
                content = json.loads(fenced_match.group(1).strip())
                if isinstance(content, dict):
                    return content
            except json.JSONDecodeError:
                pass

        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                content, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(content, dict):
                return content

        raise ValueError(f"critic response is not a JSON object: {text[:200]}")


    # ---------------- Critic evaluation ----------------
    def _observation_information_extraction(self, query, observation):
        from agents.webwalker_agent.prompt.prompts import SYSTEM_CRITIC_INFORMATION

        user_prompt = f"- Query: {query}\n- Observation: {observation}"
        messages = [
            {'role': 'system', 'content': SYSTEM_CRITIC_INFORMATION},
            {'role': 'user', 'content': user_prompt}
        ]
        trace = self._build_critic_trace(stage="critic_information_extraction", messages=messages)
        try:
            response = self.client.chat.completions.create(**self._critic_request_kwargs(messages))
            raw_response_text = response.choices[0].message.content or ""
            trace["raw_response_text"] = raw_response_text
            content = self._parse_critic_json(raw_response_text)
            trace["parsed_response"] = content
            if content.get("usefulness") is True:
                trace["result"] = content.get("information")
                return content.get("information"), trace
        except Exception as e:
            trace["error"] = str(e)
            dump_path = self._dump_critic_failure(
                stage="critic_information_extraction",
                query=query,
                observation=observation,
                messages=messages,
                error=e,
            )
            trace["failure_dump_path"] = dump_path
            if self.fail_on_critic_error:
                raise RuntimeError(
                    f"Critic API failed during information extraction "
                    f"(url={self.api_url}, model={self.model}, dump={dump_path or 'n/a'}): {e}"
                ) from e
        return None, trace

    def _critic_information(self, query):
        if not self.webwalker_memory:
            return None, self._build_critic_trace(
                stage="critic_answer_generation",
                messages=[],
                error="skip: no accumulated memory",
                skipped=True,
            )
        from agents.webwalker_agent.prompt.prompts import SYSTEM_CRITIC_ANSWER

        memory_str = "-".join(self.webwalker_memory)
        user_prompt = f"- Query: {query}\n- Accumulated Information: {memory_str}"
        messages = [
            {'role': 'system', 'content': SYSTEM_CRITIC_ANSWER},
            {'role': 'user', 'content': user_prompt}
        ]
        trace = self._build_critic_trace(stage="critic_answer_generation", messages=messages)
        try:
            response = self.client.chat.completions.create(**self._critic_request_kwargs(messages))
            raw_response_text = response.choices[0].message.content or ""
            trace["raw_response_text"] = raw_response_text
            content = self._parse_critic_json(raw_response_text)
            trace["parsed_response"] = content
            if content.get("judge") is True:
                trace["result"] = content.get("answer")
                return content.get("answer"), trace
        except Exception as e:
            trace["error"] = str(e)
            dump_path = self._dump_critic_failure(
                stage="critic_answer_generation",
                query=query,
                observation=memory_str,
                messages=messages,
                error=e,
            )
            trace["failure_dump_path"] = dump_path
            if self.fail_on_critic_error:
                raise RuntimeError(
                    f"Critic API failed during answer generation "
                    f"(url={self.api_url}, model={self.model}, dump={dump_path or 'n/a'}): {e}"
                ) from e
        return None, trace
