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
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestBuildIndexMapping:

    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "mindspeed_rl": MagicMock(),
            "mindspeed_rl.datasets": MagicMock(),
            "mindspeed_rl.datasets.utils": MagicMock(),
        }):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        with patch("mindspeed_rl.datasets.utils._build_shuffle_idx") as mock_build_shuffle_idx:
            patches = {
                "mock_build_shuffle_idx": mock_build_shuffle_idx
            }

            def fake_build_shuffle_idx(*args, **kwargs):
                return np.arange(0, 10)

            mock_build_shuffle_idx.side_effect = fake_build_shuffle_idx

            yield patches

    def test_build_index_mapping_fail_with_params(self, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.dataset_mapping import safe_build_index_mappings
        start_index = np.int32(1)
        nb_documents = 10
        num_samples = 10
        seed = 42
        full_shuffle_instruction_dataset = False
        no_shuffle = False

        with pytest.raises(ValueError, match="seed should be int"):
            safe_build_index_mappings(None, None, start_index, nb_documents, num_samples, "seed",
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        with pytest.raises(ValueError, match="seed should be no negative"):
            safe_build_index_mappings(None, None, start_index, nb_documents, num_samples, -1,
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        with pytest.raises(ValueError, match="start_index should be np.int32"):
            safe_build_index_mappings(None, None, "start_index", nb_documents, num_samples, seed,
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        with pytest.raises(ValueError, match="start_index should be no negative"):
            safe_build_index_mappings(None, None, np.int32(-1), nb_documents, num_samples, seed,
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        with pytest.raises(ValueError, match="nb_documents should be int"):
            safe_build_index_mappings(None, None, start_index, "nb_documents", num_samples, seed,
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        with pytest.raises(ValueError, match="nb_documents should be positive"):
            safe_build_index_mappings(None, None, start_index, 0, num_samples, seed,
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        with pytest.raises(ValueError, match="num_samples should be int"):
            safe_build_index_mappings(None, None, start_index, nb_documents, "num_samples", seed,
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        with pytest.raises(ValueError, match="num_samples should be positive"):
            safe_build_index_mappings(None, None, start_index, nb_documents, 0, seed,
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        with pytest.raises(ValueError, match="full_shuffle_instruction_dataset should be bool"):
            safe_build_index_mappings(None, None, start_index, nb_documents, num_samples, seed,
                                      "full_shuffle_instruction_dataset", None, no_shuffle)

        with pytest.raises(ValueError, match="no_shuffle should be bool"):
            safe_build_index_mappings(None, None, start_index, nb_documents, num_samples, seed,
                                      full_shuffle_instruction_dataset, None, "no_shuffle")

    def test_build_index_mapping_fail_with_build_shuffle_idx(self, patch_target):
        mock_build_shuffle_idx = patch_target["mock_build_shuffle_idx"]

        start_index = np.int32(1)
        nb_documents = 10
        num_samples = 10
        seed = 42
        full_shuffle_instruction_dataset = False
        no_shuffle = False

        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.dataset_mapping import safe_build_index_mappings
        mock_build_shuffle_idx.side_effect = RuntimeError("test")
        with pytest.raises(RuntimeError, match="build shuffle idx failed"):
            safe_build_index_mappings(None, None, start_index, nb_documents, num_samples, seed,
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        mock_build_shuffle_idx.side_effect = TypeError("test")
        with pytest.raises(RuntimeError, match="Unexpected error occurred when build shuffle idx"):
            safe_build_index_mappings(None, None, start_index, nb_documents, num_samples, seed,
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        mock_build_shuffle_idx.side_effect = None
        mock_build_shuffle_idx.return_value = [1, 2, 3]
        with pytest.raises(ValueError, match="new_document_ids should be np.ndarray"):
            safe_build_index_mappings(None, None, start_index, nb_documents, num_samples, seed,
                                      full_shuffle_instruction_dataset, None, no_shuffle)

        mock_build_shuffle_idx.return_value = np.arange(0, 10)

    def test_build_index_mapping_success(self, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.dataset_mapping import safe_build_index_mappings

        start_index = np.int32(1)
        nb_documents = 10
        num_samples = 10

        seed = 42
        full_shuffle_instruction_dataset = True
        no_shuffle = True

        shuffle_idx = safe_build_index_mappings(None, None, start_index, nb_documents, num_samples, seed,
                                                full_shuffle_instruction_dataset, None, no_shuffle)

        assert len(shuffle_idx) == num_samples
