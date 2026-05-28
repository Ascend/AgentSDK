#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch


class TestCopyFuncAndInsertVar:

    def test_copy_func_and_insert_var_function_exists(self):
        from aura.trainer.train_adapter.mindspeed_rl.cp_opt.copy_func_and_insert_var import copy_func_and_insert_var
        assert callable(copy_func_and_insert_var)

    def test_copy_func_and_insert_var_file_not_found(self):
        from aura.trainer.train_adapter.mindspeed_rl.cp_opt.copy_func_and_insert_var import copy_func_and_insert_var

        with pytest.raises(FileNotFoundError):
            copy_func_and_insert_var('/non/existent/file.py', '/another/non/existent.py', 'test_func')

    def test_copy_func_and_insert_var_success(self):
        from aura.trainer.train_adapter.mindspeed_rl.cp_opt.copy_func_and_insert_var import copy_func_and_insert_var

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as src_file:
            src_file.write('def test_func():\n    return 42\n')
            src_path = src_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as dst_file:
            dst_file.write('def test_func():\n    return 0\n')
            dst_path = dst_file.name

        try:
            result = copy_func_and_insert_var(src_path, dst_path, 'test_func')
            assert result is True

            with open(dst_path, 'r') as f:
                content = f.read()
                assert 'return 42' in content
        finally:
            os.unlink(src_path)
            os.unlink(dst_path)
