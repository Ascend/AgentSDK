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
from unittest.mock import patch, MagicMock

from agentic_rl.base.utils.file_utils import FileCheck


class TestFileCheck(unittest.TestCase):

    def setUp(self):
        self.valid_path = "/valid/path"
        self.long_path = "a" * 1025
        self.path_with_illegal = "/path/with/illegal@char"
        self.path_with_dotdot = "/path/../with/dotdot"
        self.symlink_path = "/path/to/symlink"
        self.valid_mode = "755"
        self.valid_user = "testuser"
        self.valid_group = "testgroup"

    def test_check_path_is_exist_and_valid_path_not_string_or_not_exists(self):
        with self.assertRaises(ValueError) as context:
            FileCheck.check_path_is_exist_and_valid(123)

        self.assertEqual(str(context.exception), "Path is not a string or path is not existed.")

    @patch("os.path.exists", return_value=False)
    def test_check_path_is_exist_and_valid_path_not_exists(self, mock_exists):
        with self.assertRaises(ValueError) as context:
            FileCheck.check_path_is_exist_and_valid(self.valid_path)

        self.assertEqual(str(context.exception), "Path is not a string or path is not existed.")

    @patch("os.path.exists", return_value=True)
    def test_check_path_is_exist_and_valid_path_too_long(self, mock_exists):
        with self.assertRaises(ValueError) as context:
            FileCheck.check_path_is_exist_and_valid(self.long_path)

        self.assertEqual(str(context.exception), "Input path is too long, it's length must be less than 1024.")

    @patch("os.path.exists", return_value=True)
    def test_check_path_is_exist_and_valid_path_has_illegal_char(self, mock_exists):
        with self.assertRaises(ValueError) as context:
            FileCheck.check_path_is_exist_and_valid(self.path_with_illegal)

        self.assertEqual(str(context.exception),
                         "There are illegal characters in path, it must be in [a-z A-Z 0-9 . _ -].")

    @patch("os.path.exists", return_value=True)
    def test_check_path_is_exist_and_valid_path_has_dotdot(self, mock_exists):
        with self.assertRaises(ValueError) as context:
            FileCheck.check_path_is_exist_and_valid(self.path_with_dotdot)

        self.assertEqual(str(context.exception), "There are '..' characters in path.")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.realpath", return_value="/real/path")
    @patch("os.path.normpath", return_value="/original/path")
    def test_check_path_is_exist_and_valid_path_is_symlink(self, mock_normpath, mock_realpath, mock_exists):
        with self.assertRaises(ValueError) as context:
            FileCheck.check_path_is_exist_and_valid(self.symlink_path)

        self.assertEqual(str(context.exception), "Path is link, it's not supported.")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.realpath", return_value="/real/path")
    @patch("os.path.normpath", return_value="/real/path")
    def test_check_path_is_exist_and_valid_valid_path(self, mock_normpath, mock_realpath, mock_exists):
        FileCheck.check_path_is_exist_and_valid(self.valid_path)

    @patch("os.stat", return_value=MagicMock(st_mode=0o755))
    def test_check_path_mode_owner_group_mode_none(self, mock_stat):
        FileCheck.check_path_mode_owner_group(self.valid_path)

    @patch("os.stat", return_value=MagicMock(st_mode=0o755))
    def test_check_path_mode_owner_group_mode_invalid_type(self, mock_stat):
        with self.assertRaises(TypeError) as context:
            FileCheck.check_path_mode_owner_group(self.valid_path, mode=755)

        self.assertEqual(str(context.exception), "The input's mode must be a string.")

    @patch("os.stat", return_value=MagicMock(st_mode=0o755))
    def test_check_path_mode_owner_group_mode_invalid_format(self, mock_stat):
        with self.assertRaises(ValueError) as context:
            FileCheck.check_path_mode_owner_group(self.valid_path, mode="888")

        self.assertEqual(str(context.exception), "The input's mode must be a 3-digit octal string, "
                                                 "e.g., '755', and it must be less than 777.")

    @patch("os.stat", return_value=MagicMock(st_mode=0o755))
    def test_check_path_mode_owner_group_user_invalid_type(self, mock_stat):
        with self.assertRaises(TypeError) as context:
            FileCheck.check_path_mode_owner_group(self.valid_path, user=123, group=self.valid_group)

        self.assertEqual(str(context.exception), "The input's user must be a string.")

    @patch("os.stat", return_value=MagicMock(st_mode=0o755))
    def test_check_path_mode_owner_group_group_invalid_type(self, mock_stat):
        with self.assertRaises(TypeError) as context:
            FileCheck.check_path_mode_owner_group(self.valid_path, user=self.valid_user, group=123)

        self.assertEqual(str(context.exception), "The input's group must be a string.")

    @patch("os.stat", return_value=MagicMock(st_mode=0o755))
    def test_check_path_mode_owner_group_user_group_not_both(self, mock_stat):
        with self.assertRaises(ValueError) as context:
            FileCheck.check_path_mode_owner_group(self.valid_path, user=self.valid_user)

        self.assertEqual(str(context.exception),
                         "The input's user and group must be specified at the same time.")

        with self.assertRaises(ValueError) as context:
            FileCheck.check_path_mode_owner_group(self.valid_path, group=self.valid_group)

        self.assertEqual(str(context.exception),
                         "The input's user and group must be specified at the same time.")

    @patch("os.stat", return_value=MagicMock(st_mode=0o744))
    def test_check_path_mode_owner_group_mode_mismatch(self, mock_stat):
        with self.assertRaises(ValueError) as context:
            FileCheck.check_path_mode_owner_group(self.valid_path, mode="755", user=self.valid_user,
                                                  group=self.valid_group)

        self.assertEqual(str(context.exception), f"The input's path mode is not right, it must be 755.")

    @patch("os.stat", return_value=MagicMock(st_mode=0o755, st_uid=1000, st_gid=1000))
    @patch("pwd.getpwuid", return_value=MagicMock(pw_name="testuser"))
    @patch("grp.getgrgid", return_value=MagicMock(gr_name="testgroup"))
    def test_check_path_mode_owner_group_all_match(self, mock_getgrgid, mock_getpwuid, mock_stat):
        FileCheck.check_path_mode_owner_group(self.valid_path, mode=self.valid_mode, user=self.valid_user,
                                              group=self.valid_group)

    @patch("os.path.exists", return_value=False)
    def test_check_data_path_is_valid_path_not_exists(self, mock_exists):
        with self.assertRaises(ValueError) as context:
            FileCheck.check_data_path_is_valid(self.valid_path)

        self.assertEqual(str(context.exception), "Path is not a string or path is not existed.")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    @patch("os.stat", return_value=MagicMock(st_mode=0o640))
    @patch("pwd.getpwuid", return_value=MagicMock(pw_name="testuser"))
    @patch("grp.getgrgid", return_value=MagicMock(gr_name="testgroup"))
    def test_check_data_path_is_valid_file_mode_640(self, mock_stat, mock_isfile, mock_exists,
                                                    mock_getgrgid, mock_getpwuid):
        FileCheck.check_data_path_is_valid(self.valid_path)

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isdir", return_value=True)
    @patch("os.walk", return_value=[("/valid/path", [], ["file1"])])
    @patch("os.stat")
    @patch("agentic_rl.base.utils.file_utils.FileCheck.check_path_mode_owner_group")
    def test_check_data_path_is_valid_files_mode_not_640(self, mock_check, mock_stat, mock_walk, mock_isdir,
                                                         mock_exists):
        def stat_side_effect(path):
            if "file1" in path:
                return MagicMock(st_mode=0o644)
            else:
                return MagicMock(st_mode=0o750)

        mock_stat.side_effect = stat_side_effect

        def check_side_effect(path, mode=None, user=None, group=None):
            if "file1" in path:
                raise ValueError("The input's path mode is not right, it must be 640.")

        mock_check.side_effect = check_side_effect

        with self.assertRaises(ValueError) as context:
            FileCheck.check_data_path_is_valid("/valid/path")

        self.assertEqual(str(context.exception), "The input's path mode is not right, it must be 640.")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isdir", return_value=True)
    @patch("os.walk", return_value=[("/valid/path", ["subdir"], ["file1"])])
    @patch("os.stat")
    @patch("agentic_rl.base.utils.file_utils.FileCheck.check_path_mode_owner_group")
    def test_check_data_path_is_valid_subdir_mode_mismatch(self, mock_check, mock_stat, mock_walk, mock_isdir,
                                                           mock_exists):
        def stat_side_effect(path):
            if "subdir" in path:
                return MagicMock(st_mode=0o744)
            else:
                return MagicMock(st_mode=0o750)

        mock_stat.side_effect = stat_side_effect

        def check_side_effect(path, mode=None, user=None, group=None):
            if "subdir" in path:
                raise ValueError("The input's path mode is not right, it must be 750.")

        mock_check.side_effect = check_side_effect

        with self.assertRaises(ValueError) as context:
            FileCheck.check_data_path_is_valid("/valid/path")

        self.assertEqual(str(context.exception), "The input's path mode is not right, it must be 750.")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isdir", return_value=True)
    @patch("os.walk", return_value=[("/valid/path", ["subdir"], ["file1"])])
    @patch("os.stat")
    @patch("agentic_rl.base.utils.file_utils.FileCheck.check_path_mode_owner_group")
    def test_check_data_path_is_valid_file_mode_mismatch(self, mock_check, mock_stat, mock_walk, mock_isdir,
                                                         mock_exists):
        def stat_side_effect(path):
            if "file1" in path:
                return MagicMock(st_mode=0o644)
            else:
                return MagicMock(st_mode=0o750)

        mock_stat.side_effect = stat_side_effect

        def check_side_effect(path, mode=None, user=None, group=None):
            if "file1" in path:
                raise ValueError("The input's path mode is not right, it must be 640.")

        mock_check.side_effect = check_side_effect

        with self.assertRaises(ValueError) as context:
            FileCheck.check_data_path_is_valid("/valid/path")

        self.assertEqual(str(context.exception), "The input's path mode is not right, it must be 640.")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isdir", return_value=True)
    @patch("os.walk", return_value=[("/valid/path", [], [])])
    @patch("os.stat", return_value=MagicMock(st_mode=0o750))
    @patch("agentic_rl.base.utils.file_utils.FileCheck.check_path_mode_owner_group", return_value=None)
    def test_check_data_path_is_valid_no_files_or_dirs(self, mock_check, mock_stat, mock_walk, mock_isdir, mock_exists):
        FileCheck.check_data_path_is_valid(self.valid_path)

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isdir", return_value=True)
    @patch("os.walk", return_value=[("/valid/path", ["subdir"], ["file1"])])
    @patch("os.stat", return_value=MagicMock(st_mode=0o750))
    @patch("agentic_rl.base.utils.file_utils.FileCheck.check_path_mode_owner_group", return_value=None)
    def test_check_data_path_is_valid_with_dirs_and_files(self, mock_check, mock_stat, mock_walk, mock_isdir,
                                                          mock_exists):
        FileCheck.check_data_path_is_valid(self.valid_path)

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isdir", return_value=True)
    @patch("os.walk", return_value=[("/valid/path", [], ["file1", "file2"])])
    @patch("os.stat")
    @patch("agentic_rl.base.utils.file_utils.FileCheck.check_path_mode_owner_group")
    def test_check_data_path_is_valid_all_files_mode_640(self, mock_check, mock_stat, mock_walk, mock_isdir,
                                                         mock_exists):
        def stat_side_effect(path):
            if "file1" in path or "file2" in path:
                return MagicMock(st_mode=0o640)
            else:
                return MagicMock(st_mode=0o750)

        mock_stat.side_effect = stat_side_effect

        def check_side_effect(path, mode=None, user=None, group=None):
            pass

        mock_check.side_effect = check_side_effect

        FileCheck.check_data_path_is_valid("/valid/path")


if __name__ == "__main__":
    unittest.main()