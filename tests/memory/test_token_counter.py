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
from unittest.mock import MagicMock, patch

from agentic_rl.memory.token_counter import SimpleTokenCounter, HuggingFaceTokenCounter


class TestSimpleTokenCounter(unittest.TestCase):
    def setUp(self):
        self.token_counter = SimpleTokenCounter(chars_per_token=1)

    def test_counter_tokens(self):
        counter = SimpleTokenCounter(chars_per_token=4)
        text = "12345678"
        self.assertEqual(counter.count_tokens(text), 2)

        text = "123456789"
        self.assertEqual(counter.count_tokens(text), 2)

        self.assertEqual(counter.count_tokens("a"), 1)

    def test_count_message(self):
        message = {"role": "user", "content": "Hello"}
        length = self.token_counter.count_message(message)
        self.assertEqual(length, 9)

    def test_truncate(self):
        text = "hello"
        truncated = self.token_counter.truncate(text, 3)
        self.assertEqual(truncated, "hel")

        truncated = self.token_counter.truncate(text, 10)
        self.assertEqual(truncated, "hello")

    def test_split_text(self):
        text = "abcdef"

        chunks = self.token_counter.split_text(text, 2)
        self.assertEqual(chunks, ["ab", "cd", "ef"])

        text = "abcde"
        chunks = self.token_counter.split_text(text, 2)
        self.assertEqual(chunks, ["ab", "cd", "e"])

    def test_counter_tokens_with_errors(self):
        with self.assertRaises(ValueError):
            self.token_counter.count_tokens(123)

    def test_counter_message_with_errors(self):
        with self.assertRaises(ValueError):
            self.token_counter.count_message("invalid_message")

    def test_truncate_with_errors(self):
        text = "invalid_message"
        with self.assertRaises(ValueError):
            self.token_counter.truncate(text, -1)


class TestHuggingFaceTokenCounter(unittest.TestCase):
    @patch("agentic_rl.memory.token_counter.AutoTokenizer")
    @patch("agentic_rl.memory.token_counter.FileCheck")
    def test_hf_counter_loading(self, mock_file_check, mock_tokenizer):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        counter = HuggingFaceTokenCounter("fake/path")
        mock_file_check.check_path_is_exist_and_valid.assert_called_with("fake/path")
        mock_tokenizer.from_pretrained.assert_called()

        mock_tokenizer_instance.encode.return_value = [1, 2, 3]
        self.assertEqual(counter.count_tokens("text"), 3)


if __name__ == "__main__":
    unittest.main()
