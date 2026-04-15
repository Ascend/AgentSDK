#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


import unittest

from agentic_rl.controllers.utils.http_status import HTTP_OK_200, HTTP_ERROR_400, HTTP_ERROR_500


class TestHttpStatus(unittest.TestCase):
    """Tests for HTTP status code constants."""

    def test_http_ok_200(self):
        """Verify HTTP_OK_200 equals 200."""
        self.assertEqual(HTTP_OK_200, 200)
        self.assertIsInstance(HTTP_OK_200, int)

    def test_http_error_400(self):
        """Verify HTTP_ERROR_400 equals 400."""
        self.assertEqual(HTTP_ERROR_400, 400)
        self.assertIsInstance(HTTP_ERROR_400, int)

    def test_http_error_500(self):
        """Verify HTTP_ERROR_500 equals 500."""
        self.assertEqual(HTTP_ERROR_500, 500)
        self.assertIsInstance(HTTP_ERROR_500, int)


if __name__ == '__main__':
    unittest.main()
