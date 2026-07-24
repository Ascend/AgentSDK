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
import copy
import json
import os
import time
from pathlib import Path

from aura.runner.agent_engine_wrapper.base.environment.base_env import BaseEnv
from agents.webwalker_agent.environment.critic import WebWalkerCriticMixin
from agents.webwalker_agent.environment.navigation import WebWalkerNavigationMixin
from agents.webwalker_agent.environment.page_fetch import WebWalkerPageFetchMixin
from agents.webwalker_agent.environment.runtime import safe_asyncio_run
from agents.webwalker_agent.golden_path_utils import match_golden_progress
from agents.webwalker_agent.reward.reward_config import get_webwalker_reward_config
from agents.webwalker_agent.reward.reward_fn import WebWalkerRewardStage
import logging

logger = logging.getLogger(__name__)

__all__ = ["WebWalkerEnvironment", "safe_asyncio_run"]

class WebWalkerEnvironment(WebWalkerNavigationMixin, WebWalkerCriticMixin, WebWalkerPageFetchMixin, BaseEnv):
    BUTTON_LIST_HEADER = "clickable button"
    BUTTON_TAG_HINT = "Each button is wrapped in a <button> tag"

    def __init__(self, task: dict | None = None, reward_fn = None, **kwargs):
        self.task = self._coerce_task(task)
        if kwargs.get("reward_mode") and "reward_mode" not in self.task:
            self.task["reward_mode"] = kwargs["reward_mode"]
        self.env_config = kwargs.get("env_config")

        if self.env_config is not None:
            self.max_steps = self.env_config.get("max_steps", 15)
            self.stop_mode = self.env_config.get("stop_mode", kwargs.get("stop_mode", "golden_path_horizon_or_finish"))
        else:
            self.max_steps = kwargs.get("max_steps", 15)
            self.stop_mode = kwargs.get("stop_mode", "golden_path_horizon_or_finish")

        self.step_count = 0

        # Use a simple fallback reward function for local tests that omit reward_fn.
        class DummyReward:
            reward = 0.0
        self.reward_fn = reward_fn if reward_fn is not None else (lambda action, stage, task_info: DummyReward())

        self.root_url = ""
        self.current_page_url = ""
        self.button_url_dict = {}
        self.webwalker_memory = []
        self.clicked_buttons = []
        self.golden_progress_indices = []
        self.reached_answer_page = False
        self.is_website_unreachable = False
        self.env_terminal_state = None
        self.golden_path = []
        self._page_cache = {}
        # Persistent SQLite cache enables deterministic offline page replay.
        self.page_cache_path = str(
            kwargs.get("page_cache_path")
            or os.getenv("WEBWALKER_PAGE_CACHE_PATH", "").strip()
            or ""
        )
        self.cache_mode = self._resolve_cache_mode(kwargs.get("cache_mode"))
        self._page_store = self._init_page_store()
        critic_early_stop_arg = kwargs.get("enable_critic_early_stop", True)
        self.enable_critic_early_stop = (
            critic_early_stop_arg
            if isinstance(critic_early_stop_arg, bool)
            else str(critic_early_stop_arg).strip().lower() not in ("0", "false", "no", "off")
        )
        # Critic model API configuration.
        self.api_url = kwargs.get("chat_model_url", "http://127.0.0.1:8005/v1")
        self.api_key = kwargs.get("chat_model_key", "EMPTY")
        self.model = kwargs.get("chat_model_name", "Qwen3-4B")
        self.chat_model_timeout = float(kwargs.get("chat_model_timeout", 60.0))
        self.chat_model_max_retries = int(kwargs.get("chat_model_max_retries", 1))
        self.chat_model_temperature = kwargs.get("chat_model_temperature", 0.0)
        if self.chat_model_temperature is not None:
            self.chat_model_temperature = float(self.chat_model_temperature)
        self.chat_model_top_p = float(kwargs.get("chat_model_top_p", 1.0))
        self.chat_model_max_tokens = kwargs.get("chat_model_max_tokens", 2048)
        if self.chat_model_max_tokens is not None:
            self.chat_model_max_tokens = int(self.chat_model_max_tokens)
        self.fail_on_critic_error = bool(kwargs.get("fail_on_critic_error", False))
        debug_dir = os.getenv("WEBWALKER_DEBUG_DIR", "").strip()
        self.critic_failure_dump_dir = str(
            kwargs.get("critic_failure_dump_dir")
            or os.getenv("WEBWALKER_CRITIC_FAILURE_DUMP_DIR", "").strip()
            or (str(Path(debug_dir) / "critic_failures") if debug_dir else "")
        )
        from openai import OpenAI

        self.client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
            timeout=self.chat_model_timeout,
            max_retries=self.chat_model_max_retries,
        )

        self.reward_config = get_webwalker_reward_config()
        self.tokenizer_path = (kwargs.get("tokenizer_path") or "").strip()
        if self.tokenizer_path:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"WebWalker Env failed to load tokenizer from {self.tokenizer_path}: {e}")
                self.tokenizer = None
        else:
            logger.warning("WebWalker Env tokenizer_path not configured; token-level features disabled.")
            self.tokenizer = None

    # ---------------- Environment interaction ----------------
    def reset(self):
        """Initialize the environment and fetch the first page."""
        self.step_count = 0
        self.button_url_dict = {}
        self.webwalker_memory = []
        self.clicked_buttons = []
        self.golden_progress_indices = []
        self.reached_answer_page = False
        self.is_website_unreachable = False
        self.env_terminal_state = None
        self._page_cache = {}
        self.current_page_url = ""
        self._normalize_indexed_task_fields()
        reset_info = {
            "metadata": self._empty_critic_metadata()
        }

        # Extract and store golden paths.
        self.golden_path = self._extract_golden_path_from_task()
        self.golden_progress_indices = [0 for _ in self.golden_path]

        self.root_url = ""
        if self.task is not None:
            raw_url_data = self.task.get("root_url", None)

            if raw_url_data is not None:
                self.root_url = self._decode_text_field(raw_url_data)
            else:
                # Fall back to the legacy website field.
                self.root_url = self._decode_text_field(self.task.get("website", ""))

        initial_obs_text = "Failed to load initial website."
        if self.root_url:
            self.current_page_url = self.root_url
            try:
                html, markdown = self._fetch_page_success_only(self.root_url, screenshot=False)

                buttons = self.extract_links_with_text(html)
                if not buttons.strip():
                    self.is_website_unreachable = True
                    self.env_terminal_state = "website_unreachable"
                    initial_obs_text = (
                        "Environment status: the initial webpage has no clickable buttons.\n\n"
                        "This is treated as an environment failure state and will be handled by the environment."
                    )
                else:
                    initial_obs_text = (
                        f"website information:\n\n{markdown}\n\n"
                        f"{self.BUTTON_LIST_HEADER}:\n\n{buttons}\n\n{self.BUTTON_TAG_HINT}"
                    )
            except Exception as e:
                self.is_website_unreachable = True
                self.env_terminal_state = "website_unreachable"
                error_msg = f"Failed to load initial website: {str(e)}"
                initial_obs_text = (
                    f"{error_msg}\n\n"
                    "Note: This appears to be a network or access error (not due to your actions).\n"
                    "The environment will mark this episode as an environment failure state."
                )
        query = (
            self.task.get("question") or self.task.get("query") or self.task.get("problem", "")
        ) if self.task else ""
        if query and not self.is_website_unreachable and initial_obs_text:
            stage1_info, stage1_trace = self._observation_information_extraction(query, initial_obs_text)
            reset_info["metadata"]["critic_traces"].append(stage1_trace)
            if stage1_info:
                reset_info["metadata"]["critic_useful_information"].append(stage1_info)
                self.webwalker_memory.append(stage1_info + "\n")
                reset_info["metadata"]["webwalker_memory_snapshot"] = list(self.webwalker_memory)
                if self.enable_critic_early_stop:
                    suggested_answer, stage2_trace = self._critic_information(query)
                    reset_info["metadata"]["critic_traces"].append(stage2_trace)
                    if suggested_answer:
                        reset_info["metadata"]["critic_suggested_answer"] = suggested_answer
                        reset_info["metadata"]["env_terminal_state"] = "critic_answer_sufficient"
                        self.env_terminal_state = "critic_answer_sufficient"

        obs_dict = {}
        if self.task:
            obs_dict.update(self.task)
        obs_dict["initial_observation"] = initial_obs_text
        obs_dict["env_terminal_state"] = self.env_terminal_state

        return obs_dict, reset_info

    def snapshot(self):
        """Create a snapshot without copy.deepcopy to avoid pickle failures."""
        new_env = WebWalkerEnvironment(
            task=self.task,
            reward_fn=self.reward_fn,
            max_steps=self.max_steps,
            chat_model_url=self.api_url,
            chat_model_key=self.api_key,
            chat_model_name=self.model,
            chat_model_timeout=self.chat_model_timeout,
            chat_model_max_retries=self.chat_model_max_retries,
            chat_model_temperature=self.chat_model_temperature,
            chat_model_max_tokens=self.chat_model_max_tokens,
            critic_failure_dump_dir=self.critic_failure_dump_dir,
            fail_on_critic_error=self.fail_on_critic_error,
            stop_mode=self.stop_mode,
            enable_critic_early_stop=self.enable_critic_early_stop,
            page_cache_path=self.page_cache_path,
            cache_mode=self.cache_mode,
            tokenizer_path=self.tokenizer_path,
        )

        # Copy mutable state.
        new_env.step_count = self.step_count
        new_env.button_url_dict = self.button_url_dict.copy()
        new_env.webwalker_memory = self.webwalker_memory.copy()
        new_env.clicked_buttons = self.clicked_buttons.copy()
        new_env.golden_progress_indices = self.golden_progress_indices.copy()
        new_env.reached_answer_page = self.reached_answer_page
        new_env.is_website_unreachable = self.is_website_unreachable
        new_env.env_terminal_state = self.env_terminal_state
        new_env.golden_path = self.golden_path.copy()
        new_env.root_url = self.root_url
        new_env.current_page_url = self.current_page_url
        new_env._page_cache = dict(self._page_cache)
        if hasattr(self, 'idx'):
            new_env.idx = self.idx
        if hasattr(self, 'prompt_id'):
            new_env.prompt_id = self.prompt_id

        from openai import OpenAI

        new_env.client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
            timeout=self.chat_model_timeout,
            max_retries=self.chat_model_max_retries,
        )

        return new_env

    def step(self, actions: list[dict]):
        if isinstance(actions, dict):
            actions = [actions]

        if self.env_terminal_state in ("website_unreachable", "critic_answer_sufficient"):
            task_info = copy.deepcopy(self.task) if self.task is not None else {}
            task_info['is_website_unreachable'] = True
            task_info['env_terminal_state'] = self.env_terminal_state
            task_info['golden_path'] = self.golden_path
            task_info['step_count'] = self.step_count
            task_info['golden_step_index'] = max(self.step_count - 1, 0)
            task_info['reward_mode'] = self._get_reward_mode()
            reward_output = self.reward_fn(action="", stage=WebWalkerRewardStage.DONE, task_info=task_info)
            total_reward = reward_output.reward if hasattr(reward_output, 'reward') else reward_output
            info_dict = {
                "response": actions,
                "metadata": {
                    "env_terminal_state": self.env_terminal_state,
                    "website_unreachable": self.env_terminal_state == "website_unreachable",
                    "critic_answer_sufficient": self.env_terminal_state == "critic_answer_sufficient",
                    "golden_path_configured": bool(self.golden_path),
                    "on_golden_path": False,
                    "clicked_button": "",
                    "target_golden_node": self._get_current_golden_node(),
                    "golden_step_index": max(self.step_count - 1, 0),
                },
            }
            if hasattr(reward_output, "metadata") and isinstance(reward_output.metadata, dict):
                info_dict["metadata"].update(reward_output.metadata)
            return {}, total_reward, True, info_dict

        self.step_count += 1
        total_reward = 0.0
        done = self.step_count >= self.max_steps
        info_dict = {"response": actions, "metadata": {}}

        # 1. Explorer-only mode: only visit_page is valid. Any finish/final-answer
        # output must pass through TOOLS_FORMAT and become a format error.
        clicked_button = self._extract_clicked_button(actions)
        parent_page_url = self.current_page_url
        selected_target_url = self._resolve_visit_target_url(actions)
        source_url_selected = bool(
            selected_target_url and self._is_source_url_hit(selected_target_url)
        )

        # Always write golden-path guidance for explicit beam node filtering.
        golden_step_index = self._get_current_click_index()
        progress_match = match_golden_progress(
            self.golden_path,
            clicked_button,
            self.golden_progress_indices,
        )
        on_golden_path = bool(progress_match.get("matched"))
        target_golden_node = str(progress_match.get("node") or self._get_current_golden_node())
        info_dict["metadata"]["golden_path_configured"] = bool(self.golden_path)
        info_dict["metadata"]["on_golden_path"] = on_golden_path if self.golden_path else False
        info_dict["metadata"]["clicked_button"] = clicked_button
        info_dict["metadata"]["target_golden_node"] = target_golden_node
        info_dict["metadata"]["golden_step_index"] = golden_step_index
        info_dict["metadata"]["golden_progress_indices"] = list(self.golden_progress_indices)
        info_dict["metadata"]["golden_progress_match"] = progress_match
        info_dict["metadata"]["parent_page_url"] = parent_page_url
        info_dict["metadata"]["selected_target_url"] = selected_target_url
        info_dict["metadata"]["source_url_selected"] = source_url_selected
        info_dict["metadata"]["current_page_url"] = self.current_page_url
        info_dict["metadata"]["stop_mode"] = self.stop_mode
        info_dict["metadata"].update(self._empty_critic_metadata())

        if done:
            task_info = copy.deepcopy(self.task) if self.task is not None else {}
            # Pass website reachability and golden_path to the reward function.
            task_info['is_website_unreachable'] = self.is_website_unreachable
            task_info['env_terminal_state'] = self.env_terminal_state
            task_info['golden_path'] = self.golden_path
            task_info['step_count'] = self.step_count
            task_info['golden_step_index'] = golden_step_index
            task_info['reward_mode'] = self._get_reward_mode()
            task_info['clicked_button'] = clicked_button
            task_info['target_golden_node'] = target_golden_node
            task_info['golden_progress_match'] = progress_match
            reward_output = self.reward_fn(action="", stage=WebWalkerRewardStage.DONE, task_info=task_info)
            total_reward += reward_output.reward if hasattr(reward_output, 'reward') else reward_output
            if hasattr(reward_output, "metadata") and isinstance(reward_output.metadata, dict):
                info_dict["metadata"].update(reward_output.metadata)

            return {}, total_reward, done, info_dict

        # Basic reward: validate tool-call format.
        task_info = copy.deepcopy(self.task) if self.task is not None else {}
        task_info['golden_path'] = self.golden_path
        task_info['step_count'] = self.step_count
        task_info['golden_step_index'] = golden_step_index
        task_info['reward_mode'] = self._get_reward_mode()
        # Pass the attempted clicked button from action args so golden-path matching
        # does not need to infer it from response text.
        task_info['clicked_button'] = clicked_button
        task_info['target_golden_node'] = target_golden_node
        task_info['golden_progress_match'] = progress_match
        format_reward_out = self.reward_fn(actions, stage=WebWalkerRewardStage.TOOLS_FORMAT, task_info=task_info)
        format_reward = format_reward_out.reward if hasattr(format_reward_out, 'reward') else format_reward_out
        total_reward += format_reward
        format_metadata = (
            format_reward_out.metadata
            if hasattr(format_reward_out, "metadata") and isinstance(format_reward_out.metadata, dict)
            else {}
        )
        if format_reward < 0 or format_metadata.get("tool_format_error"):
            info_dict["metadata"]["env_terminal_state"] = "tool_format_error"
            info_dict["metadata"]["tool_format_error"] = True
            info_dict["metadata"].update(format_metadata)
            return {}, total_reward, True, info_dict

        # Execute tools synchronously to avoid asyncio conflicts.
        tool_outputs = self._execute_tool_calls(actions)
        click_success = bool(
            clicked_button and any(self._is_successful_tool_output(v) for v in tool_outputs.values())
        )
        source_url_hit = click_success and self._is_source_url_hit(self.current_page_url)
        source_url_loaded = source_url_hit
        task_info["selected_target_url"] = selected_target_url
        task_info["source_url_selected"] = source_url_selected
        task_info["source_url_loaded"] = source_url_loaded
        task_info['source_url_hit'] = source_url_hit
        task_info['click_success'] = click_success

        # Basic reward: validate environment return values.
        return_reward_out = self.reward_fn({"tool_outputs": tool_outputs}, stage=WebWalkerRewardStage.TOOLS_RETURN, task_info=task_info)
        total_reward += return_reward_out.reward if hasattr(return_reward_out, 'reward') else return_reward_out

        if click_success:
            self.clicked_buttons.append(clicked_button)
            if progress_match.get("matched"):
                self._apply_golden_progress(progress_match)
            self.reached_answer_page = bool(
                self.reached_answer_page
                or source_url_hit
                or (progress_match.get("matched") and progress_match.get("is_final"))
            )
        info_dict["metadata"]["clicked_buttons"] = list(self.clicked_buttons)
        info_dict["metadata"]["source_url_selected"] = source_url_selected
        info_dict["metadata"]["source_url_loaded"] = source_url_loaded
        info_dict["metadata"]["source_url_hit"] = source_url_hit
        info_dict["metadata"]["current_page_url"] = self.current_page_url
        info_dict["metadata"]["golden_progress_indices"] = list(self.golden_progress_indices)
        is_chain_mode = self._get_reward_mode() == "trajectory"
        if clicked_button and not click_success:
            output_text = "\n".join(str(value) for value in tool_outputs.values())
            if "can not be clicked" in output_text:
                info_dict["metadata"]["click_terminal_state"] = "hallucinated_button"
            elif source_url_selected:
                info_dict["metadata"]["reached_golden_path_end"] = True
                info_dict["metadata"]["env_terminal_state"] = "source_url_selected_load_failed"
                self.reached_answer_page = True
            elif "Error accessing page" in output_text or "Error: " in output_text:
                info_dict["metadata"]["click_terminal_state"] = "page_access_failed"
            else:
                info_dict["metadata"]["click_terminal_state"] = "click_failed"
            if self._should_stop_rollout_on_click_failure(is_chain_mode=is_chain_mode):
                done = True
                if not info_dict["metadata"].get("env_terminal_state"):
                    info_dict["metadata"]["env_terminal_state"] = info_dict["metadata"]["click_terminal_state"]

        # Keep critic-extracted metadata without converting it into dense reward.
        query = (
            self.task.get("question") or self.task.get("query") or self.task.get("problem", "")
        ) if self.task else ""

        for _, obs_str in tool_outputs.items():
            stage1_info, stage1_trace = self._observation_information_extraction(query, obs_str)
            info_dict["metadata"]["critic_traces"].append(stage1_trace)

            if stage1_info:
                info_dict["metadata"]["critic_useful_information"].append(stage1_info)
                self.webwalker_memory.append(stage1_info + "\n")
                info_dict["metadata"]["webwalker_memory_snapshot"] = list(self.webwalker_memory)
                if self.enable_critic_early_stop:
                    suggested_answer, stage2_trace = self._critic_information(query)
                    info_dict["metadata"]["critic_traces"].append(stage2_trace)
                    if suggested_answer and self._critic_may_early_stop():
                        info_dict["metadata"]["critic_suggested_answer"] = suggested_answer
                        done = True
                        info_dict["metadata"]["critic_answer_sufficient"] = True
                        info_dict["metadata"]["env_terminal_state"] = "critic_answer_sufficient"
                    elif suggested_answer:
                        info_dict["metadata"]["critic_suggested_answer"] = suggested_answer

        reached_answer = (
            source_url_hit
            or source_url_selected
            or (click_success and progress_match.get("matched") and progress_match.get("is_final"))
        )
        if not done and reached_answer:
            info_dict["metadata"]["reached_golden_path_end"] = True
            info_dict["metadata"]["clicked_button"] = clicked_button
            info_dict["metadata"]["source_url_hit"] = source_url_hit
            done = True
            info_dict["metadata"]["env_terminal_state"] = "answer_page_reached"
        elif (
            not done
            and self._should_stop_on_non_golden_click()
            and click_success
            and self.golden_path
            and not progress_match.get("matched")
        ):
            done = True
            info_dict["metadata"]["non_golden_click"] = True
            info_dict["metadata"]["env_terminal_state"] = "non_golden_click"
        elif not done and click_success and self.golden_path and not progress_match.get("matched"):
            info_dict["metadata"]["non_golden_click"] = True

        if self._get_reward_mode() == "trajectory":
            chain_prefix_valid = self._is_chain_click_prefix_valid()
            info_dict["metadata"]["chain_prefix_valid"] = chain_prefix_valid
            info_dict["metadata"]["chain_exact_golden_path"] = self._is_chain_exact_golden_path()
            if self._should_enforce_chain_prefix() and not chain_prefix_valid:
                done = True
                info_dict["metadata"]["chain_prefix_deviated"] = True
                info_dict["metadata"]["env_terminal_state"] = "chain_prefix_deviated"

            chain_click_horizon = self._get_chain_click_horizon()
            if chain_click_horizon is not None:
                info_dict["metadata"]["chain_click_horizon"] = chain_click_horizon
                info_dict["metadata"]["chain_horizon_reached"] = self.step_count >= chain_click_horizon
                if not done and self.step_count >= chain_click_horizon:
                    done = True
                    info_dict["metadata"]["env_terminal_state"] = "chain_horizon_reached"

        info_dict["metadata"]["current_page_url"] = self.current_page_url
        next_obs = {"tool_outputs": tool_outputs}
        return next_obs, total_reward, done, info_dict

    def _execute_tool_calls(self, tool_calls: list) -> dict:
        """Execute tool calls sequentially to avoid Crawl4AI event-loop conflicts."""
        tool_outputs = {}
        for tool_call in tool_calls:
            tool_call_id, result = self.execute_tool(tool_call)
            tool_outputs[tool_call_id] = result
        return tool_outputs

    def _visit_page_by_button(self, button_clean: str) -> str:
        if button_clean not in self.button_url_dict:
            return f"The button '{button_clean}' can not be clicked, please retry a new button!"

        url = self.button_url_dict[button_clean]
        last_error = None
        for attempt in range(3):
            try:
                # Use the thread-safe fetch helper to avoid Ray event-loop crashes.
                html, markdown = self._fetch_page_success_only(url, screenshot=False)
                buttons_str = self.extract_links_with_text(html)
                response_content = markdown if markdown else "The information of the current page is not accessible"
                self.current_page_url = url
                return (
                    f"The web information is:\n\n{response_content}\n\n"
                    f"Clickable buttons are wrapped in <button> tag\n{buttons_str}"
                )
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        return f"Error accessing page: {str(last_error)}"

    def execute_tool(self, tool_call):
        tool_call_id = tool_call.get("id", "unknown")

        try:
            # Support parser output and raw OpenAI-style tool-call formats.
            if "function" in tool_call:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
            else:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]

            # If the parser leaves arguments as a string, try to decode them.
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    pass

            if tool_name == "visit_page":
                button_raw = tool_args.get("button", "")
                button_clean = button_raw.replace("<button>", "").replace("</button>", "").strip()
                result = self._visit_page_by_button(button_clean)

            elif tool_name == "finish":
                result = f"Task finished with response: {tool_args.get('response', '')}"
            elif tool_name == "error_tool":
                result = f"Tool parsing error: {tool_args.get('response', 'Unknown error')}"
            else:
                result = f"Unknown tool: {tool_name}"

        except Exception as e:
            result = f"Error executing tool: {str(e)}"

        return tool_call_id, result

    @staticmethod
    def from_dict(env_args: dict) -> "WebWalkerEnvironment":
        return WebWalkerEnvironment(**env_args)
