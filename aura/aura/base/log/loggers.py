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
import logging
import os

import torch
import torch.distributed as dist


def generate_iteration_msg(ori_msg, iteration, steps):
    formate_msg = f"iteration: {iteration} / {steps} | "
    if not isinstance(ori_msg, dict):
        formate_msg = f"{formate_msg} {str(ori_msg)}"
    else:
        for key in ori_msg:
            if key == "param/lr":
                value = "{:e}".format(ori_msg[key])
                formate_msg += f"{key} : {value} | "
            else:
                formate_msg += f"{key} : {format(ori_msg[key], '.4f')} | "
        formate_msg = formate_msg[:-2]
    return formate_msg


def handle_msg(msg, iteration, steps):
    if iteration is not None and steps is not None:
        fmt_msg = generate_iteration_msg(msg, iteration, steps)
    else:
        fmt_msg = str(msg)
    return fmt_msg


class Loggers(object):
    def __init__(self, name='root', logger_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logger_level)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logger_level)
            console_formatter = (
                logging.Formatter('%(asctime)s|%(levelname)s|%(filename)s|%(funcName)s():%(lineno)s|%(message)s'))
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        self.logger.propagate = False

    def format_info(self, msg, iteration=None, steps=None):
        if torch.distributed.is_initialized():
            if int(os.getenv("RANK", dist.get_rank())) == int(os.getenv("WORLD_SIZE", dist.get_world_size())) - 1:
                format_msg = handle_msg(msg, iteration, steps)
                self.logger.info(format_msg)
        else:
            format_msg = handle_msg(msg, iteration, steps)
            self.logger.info(format_msg)

    def format_warning(self, msg, iteration=None, steps=None):
        format_msg = handle_msg(msg, iteration, steps)
        self.logger.warning(format_msg)

    def format_debug(self, msg, iteration=None, steps=None):
        format_msg = handle_msg(msg, iteration, steps)
        self.logger.debug(format_msg)

    def format_error(self, msg, iteration=None, steps=None):
        format_msg = handle_msg(msg, iteration, steps)
        self.logger.error(format_msg)

    def get_logger(self):
        return self.logger
