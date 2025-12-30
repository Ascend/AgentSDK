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
import logging
import re
from datetime import datetime


class _MicroSecondFormatter(logging.Formatter):
    """
    Support logging with microsecond precision.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str = None):
        """
        Format the log record's time, including microseconds.

        Parameters:
            record (logging.LogRecord): The log record object.
            datefmt (str): The date format string, ignored.

        Returns:
            str: The formatted time string.
        """

        ct = datetime.fromtimestamp(record.created)
        s = ct.strftime("%Y-%m-%d-%H:%M:%S")

        milli_sec = ct.microsecond // 1000
        micro_sec = ct.microsecond % 1000

        return f"{s}.{milli_sec:03d}.{micro_sec:03d}"


class Loggers(object):
    """
    Logger class for formatting and filtering log messages.
    """

    def __init__(self, name: str = 'root', logger_level=logging.INFO):
        """
        Initialize the logger.

        Parameters:
            name (str): The logger name.
            logger_level (int): The logging level.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logger_level)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logger_level)
            console_handler.setFormatter(_MicroSecondFormatter(
                fmt='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)s] %(funcName)s: %(message)s',
            ))
            self.logger.addHandler(console_handler)

        self.logger.propagate = False

    @staticmethod
    def _generate_iteration_msg(ori_msg, iteration: int, steps: int):
        formate_msg = f"iteration: {iteration} / {steps} | "

        if not isinstance(ori_msg, dict):
            formate_msg = f"{formate_msg} {str(ori_msg)}"
            return formate_msg

        msg_list = [formate_msg]
        for key, value in ori_msg.items():
            if isinstance(value, (int, float)):
                if key == "param/lr":
                    format_value = "{:e}".format(value)
                else:
                    format_value = format(value, ".4f")
            else:
                format_value = str(value)

            msg_list.append(f"{key} : {format_value} | ")

        formate_msg = "".join(msg_list)
        if len(formate_msg) > 2:
            formate_msg = formate_msg[:-2]

        return formate_msg

    @staticmethod
    def _handle_msg(msg, iteration: int = None, steps: int = None):
        if iteration is not None and steps is not None:
            fmt_msg = Loggers._generate_iteration_msg(msg, iteration, steps)
        else:
            fmt_msg = str(msg)

        return Loggers._filter_invalid_chars(fmt_msg)

    @staticmethod
    def _filter_invalid_chars(s: str) -> str:
        invalid_chars = [
            '\n', '\f', '\r', '\b', '\t', '\v',
            '\u000D', '\u000A', '\u000C', '\u000B',
            '\u0009', '\u0008', '\u0007'
        ]
        pattern = '[' + re.escape(''.join(invalid_chars)) + ']+'
        return re.sub(pattern, ' ', s)

    def info(self, msg, iteration: int = None, steps: int = None):
        """
        Log an informational message.

        Parameters:
            msg (str or dict): The log message.
            iteration (int): The current iteration count.
            steps (int): The total number of iterations.
        """
        format_msg = Loggers._handle_msg(msg, iteration, steps)
        self.logger.info(format_msg, stacklevel=2)

    def warning(self, msg, iteration: int = None, steps: int = None):
        """
        Log a warning message.

        Parameters:
            msg (str or dict): The log message.
            iteration (int): The current iteration count.
            steps (int): The total number of iterations.
        """
        format_msg = Loggers._handle_msg(msg, iteration, steps)
        self.logger.warning(format_msg, stacklevel=2)

    def debug(self, msg, iteration: int = None, steps: int = None):
        """
        Log a debug message.

        Parameters:
            msg (str or dict): The log message.
            iteration (int): The current iteration count.
            steps (int): The total number of iterations.
        """
        format_msg = Loggers._handle_msg(msg, iteration, steps)
        self.logger.debug(format_msg, stacklevel=2)

    def error(self, msg, iteration: int = None, steps: int = None):
        """
        Log an error message.

        Parameters:
            msg (str or dict): The log message.
            iteration (int): The current iteration count.
            steps (int): The total number of iterations.
        """
        format_msg = Loggers._handle_msg(msg, iteration, steps)
        self.logger.error(format_msg, stacklevel=2)
