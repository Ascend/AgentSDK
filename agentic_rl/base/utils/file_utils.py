#!/usr/bin/env python3
# coding=utf-8
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

import grp
import os
import pwd
import re
from pathlib import Path


class FileCheck:
    @staticmethod
    def check_path_is_exist_and_valid(path: str):
        """
        Check if the path is a valid string and exists in the file system.

        This method verifies:
        - The input is a string.
        - The path exists.
        - The path length is less than 1024 characters.
        - The path contains only allowed characters: [a-z A-Z 0-9 . _ -].
        - The path does not contain '..'.
        - The path is not a symbolic link.
        """
        if not isinstance(path, str) or not os.path.exists(path):
            raise ValueError("Path is not a string or path is not existed.")

        if len(path) > 1024:
            raise ValueError("Input path is too long, it's length must be less than 1024.")

        pattern_name = re.compile(r"[^0-9a-zA-Z_./-]")
        match_name = pattern_name.findall(path)
        if match_name:
            raise ValueError("There are illegal characters in path, it must be in [a-z A-Z 0-9 . _ -].")

        if ".." in path:
            raise ValueError("There are '..' characters in path.")

        real_path = os.path.realpath(path)
        if real_path != os.path.normpath(path):
            raise ValueError("Path is link, it's not supported.")

    @staticmethod
    def check_path_mode_owner_group(path: str, mode=None, user=None, group=None):
        """
        Validate the file path's permission, owner, and group.

        This method checks:
        - The mode is a valid 3-digit octal string (e.g., '755').
        - The user and group are valid strings.
        - The user and group are both provided or both omitted.
        - The file's mode matches the expected mode.
        - The file's owner and group match the expected values.

        Args:
            path (str): The file path to check.
            mode (str, optional): The expected file mode in octal format.
            user (str, optional): The expected owner of the file.
            group (str, optional): The expected group of the file.
        """
        if mode is not None:
            if not isinstance(mode, str):
                raise TypeError("The input's mode must be a string.")
            if not re.fullmatch(r"^[0-7]{3}$", mode):
                raise ValueError("The input's mode must be a 3-digit octal string, e.g., '755', "
                                 "and it must be less than 777.")

        # Validate the types of user and group
        if user is not None and not isinstance(user, str):
            raise TypeError("The input's user must be a string.")
        if group is not None and not isinstance(group, str):
            raise TypeError("The input's group must be a string.")

        # Ensure that user and group are both provided or both omitted
        only_user_specified = user and (not group)
        only_group_specified = (not user) and group
        if only_user_specified or only_group_specified:
            raise ValueError("The input's user and group must be specified at the same time.")

        # Check if the file mode matches the expected mode
        if mode:
            if oct(os.stat(path).st_mode)[-3:] != mode:
                raise ValueError(f"The input's path mode is not right, it must be {mode}.")

        # Check if the owner and group match the expected values
        if user and group:
            if pwd.getpwuid(os.stat(path).st_uid).pw_name != user:
                raise ValueError("The input's path user not right, it must be same with current user.")
            if grp.getgrgid(os.stat(path).st_gid).gr_name != group:
                raise ValueError("The input's path group not right, it must be same with current group.")

    @staticmethod
    def check_data_path_is_valid(path: str):
        """
        Validate the data path and all its subdirectories and files.

        This method:
        - Checks if the path is valid and exists.
        - Ensures the directory has mode 750.
        - Ensures all subdirectories have mode 750.
        - Ensures all files have mode 640.
        - Ensures all files and directories are owned by the current user.

        Args:
            path (str): The data path to validate.
        """
        FileCheck.check_path_is_exist_and_valid(path)

        current_user = pwd.getpwuid(os.getuid()).pw_name
        current_group = grp.getgrgid(os.getgid()).gr_name

        if os.path.isfile(path):
            FileCheck.check_path_mode_owner_group(path, "640", current_user, current_group)
        elif os.path.isdir(path):
            FileCheck.check_path_mode_owner_group(path, "750", current_user, current_group)
            for root, dirs, files in os.walk(path):
                for name in dirs:
                    full_path = os.path.join(root, name)
                    FileCheck.check_path_mode_owner_group(full_path, "750", current_user, current_group)
                for name in files:
                    full_path = os.path.join(root, name)
                    FileCheck.check_path_mode_owner_group(full_path, "640", current_user, current_group)
        else:
            raise ValueError("The input path is not a file or directory.")

    @staticmethod
    def check_file_size(path: str, max_size: int):
        """
        Validate the file has limited size.

        This method:
        - Checks if the path is valid and exists.
        - Checks if the file at given path has size within the limit range.
        """
        FileCheck.check_path_is_exist_and_valid(path)

        path = Path(path)

        if not path.is_file():
            raise ValueError("The path is not a file to check file size")

        if path.stat().st_size > max_size:
            raise ValueError(f"The file at the path is exceed size {max_size}B")
