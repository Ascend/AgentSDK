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

import torch
import pytest
from unittest.mock import MagicMock, patch


class TestLogprobComputer:

    def test_compute_with_remove_padding(self):
        mock_self = MagicMock()
        mock_self._get_log_probs_remove_prompt_pad = MagicMock(return_value=torch.randn(2, 10))

        mock_output = MagicMock()
        mock_batch = {
            "labels": torch.randint(0, 100, (2, 10)),
            "responses": torch.randint(0, 100, (2, 10)),
            "prompt_length": torch.tensor([2, 3]),
        }

        with patch('mindspeed_rl.utils.compute.compute_log_probs', return_value=torch.randn(2, 10)):
            with patch('mindspeed_rl.utils.compute.get_parallel_state') as mock_get_ps:
                mock_ps = MagicMock()
                mock_ps.get_context_parallel_world_size.return_value = 1
                mock_get_ps.return_value = mock_ps

                with patch('mindspeed_rl.utils.context_parallel.get_tensor_allgather_cp_with_pack',
                          return_value=torch.randn(2, 10)):
                    with patch('mindspeed_rl.utils.remove_padding.postprocess_packed_seqs',
                              return_value=torch.randn(2, 10)):
                        with patch('mindspeed_rl.utils.compute.vocab_parallel_entropy',
                                  return_value=torch.randn(2, 10)):
                            from aura.trainer.train_adapter.mindspeed_rl.patch.logprob_computer import compute

                            kwargs = {
                                "use_remove_padding": True,
                                "index": None,
                                "seqlens_in_batch": torch.tensor([10, 10]),
                                "cu_seqlens_padded": torch.tensor([0, 10, 20]),
                            }

                            log_probs, entropy = compute(mock_self, mock_output, mock_batch, False, **kwargs)

                            assert log_probs is not None
                            assert entropy is not None

    def test_compute_with_remove_padding_skip_entropy(self):
        mock_self = MagicMock()

        mock_output = MagicMock()
        mock_batch = {
            "labels": torch.randint(0, 100, (2, 10)),
            "responses": torch.randint(0, 100, (2, 10)),
            "prompt_length": torch.tensor([2, 3]),
        }

        with patch('mindspeed_rl.utils.compute.compute_log_probs', return_value=torch.randn(2, 10)):
            with patch('mindspeed_rl.utils.compute.get_parallel_state') as mock_get_ps:
                mock_ps = MagicMock()
                mock_ps.get_context_parallel_world_size.return_value = 1
                mock_get_ps.return_value = mock_ps

                with patch('mindspeed_rl.utils.context_parallel.get_tensor_allgather_cp_with_pack',
                          return_value=torch.randn(2, 10)):
                    with patch('mindspeed_rl.utils.remove_padding.postprocess_packed_seqs',
                              return_value=torch.randn(2, 10)):
                        with patch('mindspeed_rl.utils.compute.vocab_parallel_entropy',
                                  return_value=torch.randn(2, 10)):
                            from aura.trainer.train_adapter.mindspeed_rl.patch.logprob_computer import compute

                            kwargs = {
                                "use_remove_padding": True,
                                "index": None,
                                "seqlens_in_batch": torch.tensor([10, 10]),
                                "cu_seqlens_padded": torch.tensor([0, 10, 20]),
                            }

                            log_probs, entropy = compute(mock_self, mock_output, mock_batch, True, **kwargs)

                            assert log_probs is not None
                            assert entropy is not None

    def test_compute_without_remove_padding(self):
        mock_self = MagicMock()
        mock_self._get_log_probs_remove_prompt_pad = MagicMock(return_value=torch.randn(2, 10))

        mock_output = MagicMock()
        mock_batch = {
            "labels": torch.randint(0, 100, (2, 10)),
            "responses": torch.randn(2, 10),
            "prompt_length": torch.tensor([2, 3]),
        }

        with patch('mindspeed_rl.utils.compute.compute_log_probs', return_value=torch.randn(2, 10)):
            with patch('mindspeed_rl.utils.compute.get_parallel_state') as mock_get_ps:
                mock_ps = MagicMock()
                mock_ps.get_context_parallel_world_size.return_value = 1
                mock_get_ps.return_value = mock_ps

                with patch('mindspeed_rl.utils.context_parallel.get_tensor_allgather_cp_without_pack',
                          return_value=torch.randn(2, 10)):
                    with patch('mindspeed_rl.utils.compute.vocab_parallel_entropy',
                              return_value=torch.randn(2, 10)):
                        from aura.trainer.train_adapter.mindspeed_rl.patch.logprob_computer import compute

                        kwargs = {
                            "use_remove_padding": False,
                            "index": None,
                        }

                        log_probs, entropy = compute(mock_self, mock_output, mock_batch, False, **kwargs)

                        assert log_probs is not None
                        assert entropy is not None

    def test_compute_without_remove_padding_skip_entropy(self):
        mock_self = MagicMock()
        mock_self._get_log_probs_remove_prompt_pad = MagicMock(return_value=torch.randn(2, 10))

        mock_output = MagicMock()
        mock_batch = {
            "labels": torch.randint(0, 100, (2, 10)),
            "responses": torch.randn(2, 10),
            "prompt_length": torch.tensor([2, 3]),
        }

        with patch('mindspeed_rl.utils.compute.compute_log_probs', return_value=torch.randn(2, 10)):
            with patch('mindspeed_rl.utils.compute.get_parallel_state') as mock_get_ps:
                mock_ps = MagicMock()
                mock_ps.get_context_parallel_world_size.return_value = 1
                mock_get_ps.return_value = mock_ps

                with patch('mindspeed_rl.utils.context_parallel.get_tensor_allgather_cp_without_pack',
                          return_value=torch.randn(2, 10)):
                    with patch('mindspeed_rl.utils.compute.vocab_parallel_entropy',
                              return_value=torch.randn(2, 10)):
                        from aura.trainer.train_adapter.mindspeed_rl.patch.logprob_computer import compute

                        kwargs = {
                            "use_remove_padding": False,
                            "index": None,
                        }

                        log_probs, entropy = compute(mock_self, mock_output, mock_batch, True, **kwargs)

                        assert log_probs is not None
                        assert entropy is not None
