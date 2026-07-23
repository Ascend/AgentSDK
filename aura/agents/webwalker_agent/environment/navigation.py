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
import json
import urllib.parse
from typing import Any

from agents.webwalker_agent.golden_path_utils import (
    extract_golden_click_paths_from_task,
    get_chain_click_horizon,
    get_target_golden_node,
    get_target_golden_node_from_progress,
    is_exact_golden_click_path,
    is_golden_click_path_prefix,
    is_golden_path_end_reached,
    match_golden_click,
    match_golden_progress,
    normalize_button_label,
)


class WebWalkerNavigationMixin:
    @staticmethod
    def _coerce_task(task: Any) -> dict:
        if task is None:
            return {}
        if isinstance(task, dict):
            normalized = dict(task)
            extra_args = normalized.pop("extra_args", None) or {}
            if isinstance(extra_args, dict):
                normalized.update(extra_args)
            if normalized.get("problem") and not normalized.get("question"):
                normalized["question"] = normalized["problem"]
            if normalized.get("task_id") and not normalized.get("id"):
                normalized["id"] = normalized["task_id"]
            return normalized

        if hasattr(task, "model_dump"):
            raw_task = task.model_dump()
        elif hasattr(task, "dict"):
            raw_task = task.dict()
        else:
            return {"question": str(task)}

        extra_args = raw_task.pop("extra_args", None) or {}
        normalized = {
            "id": raw_task.get("task_id"),
            "question": raw_task.get("problem", ""),
            "ground_truth": raw_task.get("ground_truth", ""),
            "prompt_id": raw_task.get("prompt_id", 0),
            "content": raw_task.get("content", ""),
        }
        normalized.update(extra_args)
        return normalized

    @staticmethod
    def _normalize_button_label(value: str) -> str:
        return normalize_button_label(value)

    def _decode_text_field(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        if not isinstance(value, list) or self.tokenizer is None:
            return ""

        pad_id = self.tokenizer.pad_token_id
        clean_ids = []
        for token_id in value:
            try:
                token_int = int(token_id)
            except (TypeError, ValueError):
                continue
            if token_int < 0:
                continue
            if pad_id is not None and token_int == pad_id:
                continue
            clean_ids.append(token_int)
        if not clean_ids:
            return ""
        return self.tokenizer.decode(clean_ids, skip_special_tokens=True).strip()

    def _parse_text_list_field(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [item.strip() for item in value if item.strip()]

        text = self._decode_text_field(value)
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item or "").strip()]
        if isinstance(parsed, str):
            return [parsed.strip()] if parsed.strip() else []

        separator = ";" if ";" in text else "\n"
        return [item.strip() for item in text.split(separator) if item.strip()]

    def _normalize_indexed_task_fields(self) -> None:
        if not isinstance(self.task, dict):
            return

        for key in ("root_url", "website", "question", "query", "answer", "ground_truth"):
            if key in self.task:
                decoded = self._decode_text_field(self.task.get(key))
                if decoded:
                    self.task[key] = decoded

        golden_path = self._parse_text_list_field(self.task.get("golden_path"))
        if golden_path:
            self.task["golden_path"] = golden_path

        source_website = self._parse_text_list_field(self.task.get("source_website"))
        if source_website:
            self.task["source_website"] = source_website

    def _extract_golden_path_from_task(self) -> list[list[str]]:
        return extract_golden_click_paths_from_task(self.task)

    def _get_current_click_index(self) -> int:
        return max(self.step_count - 1, 0)

    def _get_current_golden_node(self) -> str:
        target = get_target_golden_node_from_progress(self.golden_path, self.golden_progress_indices)
        return target or get_target_golden_node(self.golden_path, self._get_current_click_index())

    def _get_reward_mode(self) -> str:
        if isinstance(self.task, dict):
            reward_mode = str(self.task.get("reward_mode", "")).strip().lower()
            if reward_mode in ("step", "trajectory"):
                return reward_mode
        if isinstance(self.task, dict):
            generation_method = str(self.task.get("trajectory_generation_method", "")).strip().lower()
        else:
            generation_method = ""
        return "step" if generation_method == "tree" else "trajectory"

    def _should_stop_on_golden_path(self) -> bool:
        return self.stop_mode == "golden_path_horizon_or_finish"

    def _is_navigation_eval_mode(self) -> bool:
        return self.stop_mode == "navigation_eval"

    def _should_stop_on_click_failure(self) -> bool:
        """Hallucinated / failed clicks end the rollout in training and navigation eval."""
        return self.stop_mode in ("golden_path_horizon_or_finish", "navigation_eval")

    def _should_stop_on_non_golden_click(self) -> bool:
        """Only the training-style golden_path horizon stops on real but off-path clicks."""
        return self.stop_mode == "golden_path_horizon_or_finish"

    def _should_enforce_chain_prefix(self) -> bool:
        return self.stop_mode == "golden_path_horizon_or_finish"

    def _critic_may_early_stop(self) -> bool:
        """Navigation eval keeps Critic for logging only; it must not end the rollout."""
        return bool(self.enable_critic_early_stop) and not self._is_navigation_eval_mode()

    def _should_stop_rollout_on_click_failure(self, *, is_chain_mode: bool) -> bool:
        if not is_chain_mode:
            return self._should_stop_on_click_failure()
        return self._is_navigation_eval_mode()

    def _resolve_visit_target_url(self, actions: list) -> str:
        for tool_call in actions or []:
            if "function" in tool_call:
                tool_name = tool_call["function"].get("name", "")
                tool_args = tool_call["function"].get("arguments", {})
            else:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("arguments", {})

            if tool_name != "visit_page":
                continue
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            if not isinstance(tool_args, dict):
                continue

            for key in ("visit_url", "url", "target_url"):
                raw_url = tool_args.get(key)
                if raw_url:
                    return str(raw_url).strip()

            button_raw = tool_args.get("button", "")
            if isinstance(button_raw, str) and button_raw.strip():
                button_clean = self._normalize_button_label(button_raw)
                if button_clean in self.button_url_dict:
                    return str(self.button_url_dict[button_clean] or "").strip()
        return ""

    def _get_chain_click_horizon(self) -> int | None:
        if self._get_reward_mode() != "trajectory":
            return None
        if not self._should_stop_on_golden_path():
            return None
        return get_chain_click_horizon(self.golden_path)

    def _is_chain_click_prefix_valid(self) -> bool:
        if self._get_reward_mode() != "trajectory":
            return True
        if not self._should_stop_on_golden_path():
            return True
        if self.reached_answer_page:
            return True
        if any(progress > 0 for progress in self.golden_progress_indices):
            return True
        if not self.golden_path or not self.clicked_buttons:
            return True
        return is_golden_click_path_prefix(self.golden_path, self.clicked_buttons)

    def _is_chain_exact_golden_path(self) -> bool:
        if self._get_reward_mode() != "trajectory":
            return False
        return is_exact_golden_click_path(self.golden_path, self.clicked_buttons)

    def _extract_clicked_button(self, actions: list[dict]) -> str:
        for tool_call in actions:
            if "function" in tool_call:
                tool_name = tool_call["function"].get("name", "")
                tool_args = tool_call["function"].get("arguments", {})
            else:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("arguments", {})

            if tool_name != "visit_page":
                continue
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            button_raw = tool_args.get("button", "") if isinstance(tool_args, dict) else ""
            if isinstance(button_raw, str):
                return self._normalize_button_label(button_raw)
        return ""

    @staticmethod
    def _extract_finish_response(arguments) -> str:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return arguments.strip()

        if not isinstance(arguments, dict):
            return ""

        return str(
            arguments.get("response")
            or arguments.get("answer")
            or arguments.get("reason")
            or ""
        ).strip()

    @staticmethod
    def _is_successful_tool_output(result: Any) -> bool:
        result_text = result if isinstance(result, str) else str(result)
        if "can not be clicked" in result_text:
            return False
        if "information of the current page is not accessible" in result_text:
            return False
        if "Error accessing page" in result_text or "Error: " in result_text:
            return False
        return True

    def _is_golden_path_end_reached(self, clicked_button: str) -> bool:
        progress_match = match_golden_progress(
            self.golden_path,
            clicked_button,
            self.golden_progress_indices,
        )
        if progress_match.get("matched"):
            return bool(progress_match.get("is_final"))
        return is_golden_path_end_reached(
            self.golden_path,
            clicked_button,
            self._get_current_click_index(),
        )

    def _is_on_golden_path_step(self, clicked_button: str) -> bool:
        """Return whether the clicked button matches the parsed golden click sequence."""
        progress_match = match_golden_progress(
            self.golden_path,
            clicked_button,
            self.golden_progress_indices,
        )
        if progress_match.get("matched"):
            return True
        is_match, _ = match_golden_click(
            self.golden_path,
            clicked_button,
            self._get_current_click_index(),
        )
        return is_match

    def _source_urls(self) -> list[str]:
        if not isinstance(self.task, dict):
            return []
        info = self.task.get("info") if isinstance(self.task.get("info"), dict) else {}
        raw_urls = info.get("source_website") if isinstance(info, dict) else None
        if raw_urls is None:
            raw_urls = self.task.get("source_website")
        return self._parse_text_list_field(raw_urls)

    @staticmethod
    def _normalize_url_for_match(url: str) -> str:
        text = str(url or "").strip()
        if not text:
            return ""
        parsed = urllib.parse.urlsplit(text)
        path = parsed.path.rstrip("/")
        query = f"?{parsed.query}" if parsed.query else ""
        return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, query, ""))

    def _is_source_url_hit(self, url: str) -> bool:
        current = self._normalize_url_for_match(url)
        if not current:
            return False
        for source_url in self._source_urls():
            source = self._normalize_url_for_match(source_url)
            if current == source:
                return True
        return False

    def _apply_golden_progress(self, progress_match: dict) -> None:
        if not progress_match.get("matched"):
            return
        path_index = progress_match.get("path_index")
        advance_to = progress_match.get("advance_to")
        if not isinstance(path_index, int) or not isinstance(advance_to, int):
            return
        while len(self.golden_progress_indices) < len(self.golden_path):
            self.golden_progress_indices.append(0)
        self.golden_progress_indices[path_index] = max(
            self.golden_progress_indices[path_index],
            advance_to,
        )

    def _extract_ground_truth(self, task_info: dict) -> str:
        if not isinstance(task_info, dict):
            return ""

        for key in ("ground_truth", "Answer", "answer"):
            value = task_info.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()

        labels = task_info.get("labels", None)
        if labels is None:
            return ""

        if isinstance(labels, str):
            return labels.strip()

        if hasattr(labels, "tolist"):
            labels = labels.tolist()

        if isinstance(labels, list) and self.tokenizer is not None:
            pad_id = self.tokenizer.pad_token_id
            clean_ids = []
            for token_id in labels:
                if not isinstance(token_id, int):
                    continue
                if token_id < 0 or token_id == -100:
                    continue
                if pad_id is not None and token_id == pad_id:
                    continue
                clean_ids.append(token_id)
            if clean_ids:
                return self.tokenizer.decode(clean_ids, skip_special_tokens=True).strip()
        return ""
