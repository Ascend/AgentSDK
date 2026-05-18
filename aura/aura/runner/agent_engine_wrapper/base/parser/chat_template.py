#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import json
from typing import Dict, List, Any, Optional
from .utils import PARSER_TEST_MESSAGES
from aura.runner.agent_engine_wrapper.rllm.msg_handler import preprocess_messages_for_qwen35
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class ChatTemplateParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.assistant_token = ""

    def get_assistant_token(self):
        return self.assistant_token

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def verify_equivalence(self, messages, verbose=True):
        """Verify that parsing messages together is equivalent to parsing them individually.

        Args:
            messages (list): List of message dictionaries to test
            verbose (bool): Whether to print detailed information about the test

        Returns:
            bool: True if the equivalence check passes, False otherwise

        Raises:
            Exception: If the equivalence check fails and verbose is True
        """
        batch_result = self.parse(messages)

        individual_results = []
        for message in messages:
            individual_results.append(self.parse([message]))

        concatenated_result = "".join(individual_results)

        is_equivalent = batch_result == concatenated_result

        if verbose and not is_equivalent:
            logger.error("Equivalence check failed!")
            logger.info("Batch parsing result:")
            logger.info(batch_result)
            logger.info("\nConcatenated individual parsing result:")
            logger.info(concatenated_result)
            raise AssertionError("Parser failed equivalence check. See above for details.")

        return is_equivalent

    @classmethod
    def get_parser(cls, tokenizer, disable_thinking=False, toolcall_parser="") -> "ChatTemplateParser":
        """Factory method to get the appropriate parser based on a string identifier.

        Args:
            parser_type (str): String identifier for the parser type
            tokenizer: The tokenizer to use with the parser
            disable_thinking: Whether generation prompt will disable thinking.

        Returns:
            ChatTemplateParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Determine parser type based on tokenizer name or path
        if toolcall_parser == "qwen3_coder":
            logger.info(f"Using Qwen3ChatTemplateParser for {tokenizer.name_or_path}")
            return Qwen3ChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            logger.info(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")

            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                logger.info(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                return DeepseekQwenChatTemplateParser(tokenizer)
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                logger.info(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                return QwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
            elif "llama" in model_name:
                logger.info(f"Using LlamaChatTemplateParser for {tokenizer.name_or_path}")
                return LlamaChatTemplateParser(tokenizer)
            elif "sophontokenizer" in tokenizer_cls:
                logger.info(f"Using PanguVocabLlmV4ChatTemplateParser for {tokenizer.name_or_path}")
                return PanguVocabLlmV4ChatTemplateParser(tokenizer, disable_thinking=disable_thinking)

        parser = ChatTemplateParser(tokenizer)
        logger.warning(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        if not parser.verify_equivalence(PARSER_TEST_MESSAGES):
            raise Exception("Parser failed equivalence check")
        return parser


class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, no_thinking=True):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        if no_thinking:
            self.assistant_token += "</think>"
        self.generation_prompt = self.eos_token + self.assistant_token + "<think>\n"
        self.tool_response_start_token = "\n<｜tool_output_begin｜>\n"
        self.tool_response_end_token = "\n<｜tool_output_end｜>\n"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message) -> str:
        return self.system_token + message["content"]

    def parse_user(self, message) -> str:
        return self.user_token + message["content"]

    def parse_assistant(self, message) -> str:
        return self.assistant_token + message["content"] + self.eos_token

    def parse_tool(self, message) -> str:
        return self.tool_response_start_token + message["content"] + self.tool_response_end_token


class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking=True):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\\n\\n</think>\\n\\n"
        self.generation_prompt = self.assistant_token

        self.tool_start_token = "\n<tool_call>\n"
        self.tool_end_token = "\n</tool_call>"

        self.tool_response_start_token = "<tool_response>\n"
        self.tool_response_end_token = "\n</tool_response>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg and messages[0]["role"] != "system":
            result += (
                self.system_token
                + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                + self.eot_token
            )

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message) -> str:
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message) -> str:
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message) -> str:
        result = self.assistant_token + message["content"] + self.eot_token
        return result

    def parse_tool(self, message) -> str:
        return (
            self.user_token
            + self.tool_response_start_token
            + message["content"]
            + self.tool_response_end_token
            + self.eot_token
        )


class LlamaChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = "<|begin_of_text|>"
        self.system_token = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_token = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.eot_token = "<|eot_id|>"
        self.generation_prompt = self.assistant_token

        self.tool_start_token = "<|start_header_id|>tool<|end_header_id|>\n\n"
        self.tool_end_token = "<|eot_id|>"
        self.tool_response_start_token = "<|start_header_id|>tool_response<|end_header_id|>\n\n"
        self.tool_response_end_token = "<|eot_id|>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message) -> str:
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message) -> str:
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message) -> str:
        return self.assistant_token + message["content"] + self.eot_token

    def parse_tool(self, message) -> str:
        return (
            self.user_token
            + self.tool_response_start_token
            + message["content"]
            + self.tool_response_end_token
            + self.eot_token
        )


class Qwen3ChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking: bool = True):
        super().__init__(tokenizer)
        self.disable_thinking = disable_thinking

        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        self.tool_token = "<|im_start|>tool\n"
        self.generation_prompt = self.assistant_token
        self.tool_start_token = "\n"
        self.tool_end_token = "\n"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        tools = kwargs.get("tools", None)
        # logger.info(f">>> parse messages: {messages}")
        # logger.info(f">>> parse tools: {tools}")
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages2 = preprocess_messages_for_qwen35(messages)

                return self.tokenizer.apply_chat_template(
                    messages2,
                    tools=tools,
                    chat_template=None,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception as e:
                logger.exception(
                    f"parse message: {messages}, apply_chat_template failed: {e}, falling back to manual parse"
                )

        return self._manual_parse(messages, tools=tools, add_generation_prompt=add_generation_prompt)

    def _manual_parse(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        result = ""
        # 1. Preprocess: inject tools definition into system message (official behavior)
        processed_messages = copy.deepcopy(messages)
        if tools and processed_messages and processed_messages[0].get("role") == "system":
            tools_text = self._render_tools_definition(tools)
            processed_messages[0]["content"] = processed_messages[0].get("content", "") + tools_text

        # 2. Parse each message with correct role token
        for message in processed_messages:
            role = message.get("role", "")
            content = message.get("content", "") or ""

            # Use role directly for token: <|im_start|>{role}\n
            result += f"<|im_start|>{role}\n"

            if role == "assistant":
                # Handle assistant with potential tool_calls
                result += content
                if "tool_calls" in message:
                    result += self._render_tool_calls(message["tool_calls"])
            else:
                # Support list format content: [{"text": "...", "type": "text"}, ...]
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text" and "text" in item:
                                text_parts.append(item["text"])
                            elif "text" in item:
                                # Handle {"text": "...", ...} format without explicit type
                                text_parts.append(item["text"])
                    content = "".join(text_parts)
                result += content

            result += self.eot_token

        # 3. Add generation prompt if needed
        if add_generation_prompt:
            result += self.generation_prompt

        return result

    def _render_tools_definition(self, tools: List[Dict[str, Any]]) -> str:
        """Render tools definition as JSON Schema (injected into system message)"""
        tools_text = "\n\n# Tools\n\nYou may call one or more tools to assist with the user query.\n"
        tools_text += "You are provided with tool descriptions in the following format:\n"
        tools_text += json.dumps(tools, ensure_ascii=False, indent=2)
        return tools_text

    def parse_system(self, message: Dict[str, Any]) -> str:
        return self.system_token + message.get("content", "") + self.eot_token

    def parse_user(self, message: Dict[str, Any]) -> str:
        return self.user_token + message.get("content", "") + self.eot_token

    def parse_assistant(self, message: Dict[str, Any]) -> str:
        content = message.get("content", "") or ""
        result = self.assistant_token + content
        if "tool_calls" in message:
            result += self._render_tool_calls(message["tool_calls"])
        result += self.eot_token
        return result

    def parse_tool(self, message: Dict[str, Any]) -> str:
        return self.tool_token + message.get("content", "") + self.eot_token

    def _render_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        parts = []
        for tc in tool_calls:
            func_info = tc.get("function", {})
            args = func_info.get("arguments", "{}")
            # Ensure args is parsed to dict then re-stringified
            try:
                if isinstance(args, str):
                    args_dict = json.loads(args)
                else:
                    args_dict = args
            except Exception as e:
                logger.warning(f"_render_tool_calls json.loads toolcalls failed! e={e}")
                args_dict = args
            call_json = {"name": func_info.get("name", ""), "arguments": args_dict}
            # Official format: \n<tool_call>\n{json}\n</tool_call>\n
            parts.append(f"\n<tool_call>\n{json.dumps(call_json, ensure_ascii=False)}\n</tool_call>\n")
        return "".join(parts)


class PanguVocabLlmV4ChatTemplateParser(ChatTemplateParser):
    """
    Pangu_Vocab_LLM-4 parser using transformers Tokenizer's built-in chat template parsing function
    refer to the following link for details about Pangu_Vocab_LLM-4:
    https://codehub-g.huawei.com/Central-Research-Institute/NoahArk_Lab/Pangu_Sophon/pangu_tokenizer/files?ref=master&filePath=Pangu_Vocab_LLM-4
    """

    def __init__(self, tokenizer, disable_thinking=False):
        super().__init__(tokenizer)

        # Training scenario: no tool descriptions needed
        self.tools = None

        # Default control parameters (can be overridden when parse() is called)
        self.default_thinking = not disable_thinking
        self.default_interleaved_think = False
        self.default_reasoning_effort = "high"

        # Special token (for is_first_msg handling)
        self.bos_token = "<|pangu_text_start|>"

        logger.info(f"PanguVocabLlmV4ChatTemplateParser initialized for {tokenizer.name_or_path}")

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs):
        """
        Render messages using transformers Tokenizer's built-in function.

        Args:
            messages: List of messages (rollout trajectories)
            add_generation_prompt: Compatibility parameter, template already includes generation prompt
            is_first_msg: Whether this is the first message (determines whether to remove BOS token)
            **kwargs: Optional parameters to override defaults (tools, think, interleaved_think, reasoning_effort)

        Returns:
            Rendered string
        """
        # Build template variables
        template_vars = {
            "tools": kwargs.get("tools", self.tools),
            "think": kwargs.get("think", self.default_thinking),
            "interleaved_think": kwargs.get("interleaved_think", self.default_interleaved_think),
            "reasoning_effort": kwargs.get("reasoning_effort", self.default_reasoning_effort),
        }

        # Call transformers Tokenizer's built-in parsing function
        result = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **template_vars
        )

        # If not the first message, remove BOS token
        if not is_first_msg and result.startswith(self.bos_token):
            result = result[len(self.bos_token) :]

        return result
