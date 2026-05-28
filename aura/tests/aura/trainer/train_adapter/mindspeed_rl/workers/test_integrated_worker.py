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
from unittest.mock import MagicMock, patch


class TestIntegratedWorker:

    def test_integrated_worker_class_exists(self):
        from aura.trainer.train_adapter.mindspeed_rl.workers.integrated_worker import IntegratedWorker
        assert IntegratedWorker is not None

    def test_temporary_micro_batch_size(self):
        from aura.trainer.train_adapter.mindspeed_rl.workers.integrated_worker import temporary_micro_batch_size

        mock_worker = MagicMock()
        mock_worker.micro_batch_size = 8
        mock_args = MagicMock()
        mock_args.micro_batch_size = 8

        with temporary_micro_batch_size(worker=mock_worker, args=mock_args, new_mbs=4):
            assert mock_worker.micro_batch_size == 4
            assert mock_args.micro_batch_size == 4

        assert mock_worker.micro_batch_size == 8
        assert mock_args.micro_batch_size == 8

    def test_load_checkpoint_with_path_without_path(self):
        from aura.trainer.train_adapter.mindspeed_rl.workers.integrated_worker import IntegratedWorker

        args_obj = MagicMock()
        args_obj.load = None

        worker = MagicMock()
        worker.get_args.return_value = args_obj

        mock_model = MagicMock()

        IntegratedWorker.load_checkpoint_with_path(worker, mock_model, None)

        assert args_obj.load is None

    def test_set_args(self):
        from aura.trainer.train_adapter.mindspeed_rl.workers.integrated_worker import IntegratedWorker

        args_obj = MagicMock()
        args_obj.existing_attr = "original"

        worker = MagicMock()
        worker.get_args.return_value = args_obj

        IntegratedWorker._set_args(worker, {"existing_attr": "updated"})

        args_obj.existing_attr = "updated"
