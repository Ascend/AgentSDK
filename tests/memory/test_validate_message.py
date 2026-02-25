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

from agentic_rl.memory.utils import validate_message


class TestValidateMessage(unittest.TestCase):
    def test_valid_dict(self):
        msg = {"role": "user", "content": "hello"}
        self.assertTrue(validate_message(msg))

    def test_valid_dict_extra_keys(self):
        msg = {"role": "user", "content": "hello", "extra": "value"}
        self.assertTrue(validate_message(msg))

    def test_invalid_dict_missing_role(self):
        msg = {"content": "hello"}
        self.assertFalse(validate_message(msg))

    def test_invalid_dict_missing_content(self):
        msg = {"role": "user"}
        self.assertFalse(validate_message(msg))

    def test_invalid_dict_non_string_key(self):
        msg = {"role": "user", "content": "hello", 1: "value"}
        self.assertFalse(validate_message(msg))

    def test_valid_list(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        self.assertTrue(validate_message(msgs))

    def test_invalid_list_one_missing_role(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"content": "hi"},
        ]
        self.assertFalse(validate_message(msgs))

    def test_invalid_list_one_missing_content(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant"},
        ]
        self.assertFalse(validate_message(msgs))

    def test_invalid_type(self):
        self.assertFalse(validate_message("string"))
        self.assertFalse(validate_message(123))
        self.assertFalse(validate_message(None))


if __name__ == '__main__':
    unittest.main()
