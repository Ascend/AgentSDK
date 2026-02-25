#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import unittest

from agentic_rl.memory.memory_simple import MemorySimple
from agentic_rl.memory.token_counter import SimpleTokenCounter


class TestMemorySimple(unittest.TestCase):
    def setUp(self):
        self.memory = MemorySimple()
        self.memory.token_counter = SimpleTokenCounter(chars_per_token=1)

    def test_add_message_attributes(self):
        message = {"role": "user", "content": "Hello"}
        self.memory.add_message(message)

        saved_message = self.memory.get_messages()[0]
        self.assertIn("id", saved_message)
        self.assertIn("time", saved_message)
        self.assertEqual(saved_message["id"], 0)
        self.assertEqual(saved_message["role"], "user")

    def test_add_message_with_error(self):
        with self.assertRaises(ValueError):
            self.memory.add_message("invalid_message")

    def test_token_caching(self):
        message = {"role": "user", "content": "Hello"}

        self.memory.add_message(message)
        msg_id = self.memory.get_messages()[0]["id"]

        self.assertIn(msg_id, self.memory._token_cache)
        self.assertEqual(self.memory._token_cache[msg_id], 9)

        total_len = self.memory.get_total_length()
        self.assertEqual(total_len, 9)

    def test_simplify_thinking_list(self):
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Start <think>Deep thought process</think> End"},
        ]

        simplified = self.memory.simplify_or_remove_think(messages)
        content = simplified[1]["content"]

        self.assertIn("Thinking process omitted", content)
        self.assertNotIn("Deep thought process", content)
        self.assertTrue(content.startswith("Start <think>"))
        self.assertTrue(content.endswith("</think> End"))

    def test_simplify_thinking_list_with_idx(self):
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Start <think>Deep thought process</think> End"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Start <think>Deep thought process</think> End"},
        ]

        # Test with valid range
        simplified = self.memory.simplify_or_remove_think(messages, 0, 2)
        content1 = simplified[1]["content"]
        content2 = simplified[3]["content"]

        # First assistant message should be simplified
        self.assertIn("Thinking process omitted", content1)
        self.assertNotIn("Deep thought process", content1)
        self.assertTrue(content1.startswith("Start <think>"))
        self.assertTrue(content1.endswith("</think> End"))

        # Second assistant message should not be simplified (outside range)
        self.assertIn("Deep thought process", content2)

        # Test with end_id = -1
        simplified_all = self.memory.simplify_or_remove_think(messages, 0, -1)
        content1_all = simplified_all[1]["content"]
        content2_all = simplified_all[3]["content"]

        # Both assistant messages should be simplified
        self.assertIn("Thinking process omitted", content1_all)
        self.assertIn("Thinking process omitted", content2_all)

    def test_simplify_thinking_str(self):
        text = "Start <think>Deep thought process</think> End"

        simplified = self.memory.simplify_or_remove_think(text)

        self.assertIn("Thinking process omitted", simplified)
        self.assertNotIn("Deep thought process", simplified)
        self.assertTrue(simplified.startswith("Start <think>"))
        self.assertTrue(simplified.endswith("</think> End"))

    def test_get_window_messages(self):
        for i in range(10):
            self.memory.add_message({"role": "user", "content": f"msg_{i}"})

        window = self.memory.get_window_messages(limit_size=5)
        self.assertEqual(len(window), 5)
        self.assertEqual(window[0]["content"], "msg_5")
        self.assertEqual(window[-1]["content"], "msg_9")

        self.assertEqual(list(window[0].keys()), ["role", "content"])

    def test_clear_memory(self):
        self.memory.add_message({"role": "user", "content": "test"})
        self.memory.clear_memory()
        self.assertEqual(len(self.memory), 0)
        self.assertEqual(len(self.memory._token_cache), 0)

        self.memory.clear_memory(role="system", content="init")
        self.assertEqual(len(self.memory), 1)
        self.assertEqual(self.memory.get_messages()[0]["role"], "system")

    def test_remove_other_keys(self):
        messages = [
            {"role": "user", "content": "Hello", "extra_key": "should_be_removed"},
            {"role": "assistant", "content": "Hi", "timestamp": "now"},
        ]
        saved_keys = ["role", "content"]

        messages = self.memory._remove_message_other_key(messages, saved_keys)
        for msg in messages:
            self.assertEqual(list(msg.keys()), saved_keys)

    def test_get_message_length_with_token_counter(self):
        message = {"id": 3, "role": "user", "content": "Hello"}
        self.memory._token_cache = {3: 5}
        length = self.memory.get_message_length(message)
        self.assertEqual(length, 5)

    def test_get_message_length_no_id(self):
        message = {"role": "user", "content": "Hello"}
        self.memory.token_counter = None
        length = self.memory.get_message_length(message)
        self.assertEqual(length, 0)

    def test_get_total_length(self):
        messages = [
            {"id": 3, "role": "user", "content": "Hello"},
            {"id": 4, "role": "user", "content": "Hello"},
        ]
        self.memory._token_cache = {3: 5}
        length = self.memory.get_total_length(messages)
        self.assertEqual(length, 14)

    def test_get_total_length_with_non_message(self):
        self.memory._token_cache = {3: 5, 4: 5}
        length = self.memory.get_total_length(None)
        self.assertEqual(length, 10)


if __name__ == "__main__":
    unittest.main()
