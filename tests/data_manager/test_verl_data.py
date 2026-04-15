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

import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module-level mocks  (BEFORE importing the code under test)
# ---------------------------------------------------------------------------
mock_torch = MagicMock()
mock_np = MagicMock()
mock_verl = MagicMock()

# Make isinstance(value, torch.Tensor) work with real Python objects
_FakeTensor = type('Tensor', (), {})
mock_torch.Tensor = _FakeTensor

# Make isinstance(value, np.ndarray) work with real Python objects
_FakeNdarray = type('ndarray', (), {})
mock_np.ndarray = _FakeNdarray

# torch.from_numpy should return a distinguishable sentinel
mock_torch.from_numpy = MagicMock(side_effect=lambda x: f"tensor_from_numpy({x})")

with patch.dict(sys.modules, {
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'numpy': mock_np,
    'verl': mock_verl,
}):
    from agentic_rl.data_manager.verl_data import VerlDataManager
    import agentic_rl.data_manager.verl_data as _verl_data_mod


class TestVerlDataManager(unittest.TestCase):
    """Tests for VerlDataManager covering all public and private methods."""

    def setUp(self):
        self.dm = VerlDataManager()

    def tearDown(self):
        mock_torch.reset_mock()
        mock_np.reset_mock()
        mock_verl.reset_mock()

    # ---- __init__ -----------------------------------------------------------

    def test_init_defaults(self):
        """Verify default attribute values after construction."""
        self.assertIsNone(self.dm.controller)
        self.assertIsNone(self.dm._current_batch)
        self.assertEqual(self.dm._metrics, {})
        self.assertEqual(self.dm._pad_token_id, 0)

    # ---- sync_init_data_manager --------------------------------------------

    def test_sync_init_with_controller(self):
        """When the argument has get_next_training_batch, it is stored as controller."""
        ctrl = MagicMock()
        ctrl.get_next_training_batch = MagicMock()
        self.dm.sync_init_data_manager(ctrl)
        self.assertIs(self.dm.controller, ctrl)

    def test_sync_init_with_config(self):
        """When the argument lacks get_next_training_batch, it is stored as _config."""
        cfg = {"lr": 0.01}
        self.dm.sync_init_data_manager(cfg)
        self.assertIsNone(self.dm.controller)
        self.assertEqual(self.dm._config, cfg)

    # ---- all_consumed -------------------------------------------------------

    def test_all_consumed_returns_zero(self):
        """all_consumed always returns 0."""
        result = self.dm.all_consumed("train")
        self.assertEqual(result, 0)

    # ---- get_data -----------------------------------------------------------

    def test_get_data_no_controller(self):
        """get_data returns empty when no controller is set."""
        batch, indices = self.dm.get_data("train", None, 16)
        self.assertEqual(batch, {})
        self.assertEqual(indices, [])

    def test_get_data_with_controller(self):
        """get_data returns raw_batch and [0] from the controller."""
        ctrl = MagicMock()
        ctrl.get_next_training_batch.return_value = ({"input_ids": [1, 2]}, None)
        self.dm.sync_init_data_manager(ctrl)

        batch, indices = self.dm.get_data("train", None, 16)
        ctrl.get_next_training_batch.assert_called_once_with(False)
        self.assertEqual(batch, {"input_ids": [1, 2]})
        self.assertEqual(indices, [0])

    def test_get_data_updates_metrics(self):
        """get_data stores returned metrics."""
        ctrl = MagicMock()
        ctrl.get_next_training_batch.return_value = ({}, {"loss": 0.5})
        self.dm.sync_init_data_manager(ctrl)

        self.dm.get_data("train", None, 16)
        self.assertEqual(self.dm._metrics, {"loss": 0.5})

    def test_get_data_skips_none_metric(self):
        """get_data does not update metrics when metric is None."""
        ctrl = MagicMock()
        ctrl.get_next_training_batch.return_value = ({}, None)
        self.dm.sync_init_data_manager(ctrl)

        self.dm.get_data("train", None, 16)
        self.assertEqual(self.dm._metrics, {})

    # ---- put_data -----------------------------------------------------------

    def test_put_data_first_batch(self):
        """First put_data sets _current_batch to the converted DataProto."""
        with patch.object(_verl_data_mod, 'np', mock_np), \
             patch.object(_verl_data_mod, 'torch', mock_torch), \
             patch.object(_verl_data_mod, 'DataProto', mock_verl.DataProto):
            mock_verl.DataProto.from_dict.return_value = "proto_1"
            self.dm.put_data({"a": _FakeTensor()}, [0])
            self.assertEqual(self.dm._current_batch, "proto_1")

    def test_put_data_union_subsequent(self):
        """Subsequent put_data calls union with existing batch."""
        with patch.object(_verl_data_mod, 'np', mock_np), \
             patch.object(_verl_data_mod, 'torch', mock_torch), \
             patch.object(_verl_data_mod, 'DataProto', mock_verl.DataProto):
            proto1 = MagicMock()
            proto2 = MagicMock()
            proto1.union.return_value = "merged"
            mock_verl.DataProto.from_dict.side_effect = [proto1, proto2]

            self.dm.put_data({"a": _FakeTensor()}, [0])
            self.dm.put_data({"b": _FakeTensor()}, [1])

            proto1.union.assert_called_once_with(proto2)
            self.assertEqual(self.dm._current_batch, "merged")

    def test_put_data_with_metric(self):
        """put_data stores passed metrics."""
        with patch.object(_verl_data_mod, 'np', mock_np), \
             patch.object(_verl_data_mod, 'torch', mock_torch), \
             patch.object(_verl_data_mod, 'DataProto', mock_verl.DataProto):
            mock_verl.DataProto.from_dict.side_effect = None
            mock_verl.DataProto.from_dict.return_value = MagicMock()
            self.dm.put_data({"a": _FakeTensor()}, [0], metric={"acc": 0.9})
            self.assertEqual(self.dm._metrics, {"acc": 0.9})

    def test_put_data_without_metric(self):
        """put_data does not alter metrics when metric is None."""
        with patch.object(_verl_data_mod, 'np', mock_np), \
             patch.object(_verl_data_mod, 'torch', mock_torch), \
             patch.object(_verl_data_mod, 'DataProto', mock_verl.DataProto):
            mock_verl.DataProto.from_dict.side_effect = None
            mock_verl.DataProto.from_dict.return_value = MagicMock()
            self.dm.put_data({"a": _FakeTensor()}, [0])
            self.assertEqual(self.dm._metrics, {})

    # ---- put_experience -----------------------------------------------------

    def test_put_experience_delegates_to_put_data(self):
        """put_experience delegates to put_data."""
        with patch.object(_verl_data_mod, 'np', mock_np), \
             patch.object(_verl_data_mod, 'torch', mock_torch), \
             patch.object(_verl_data_mod, 'DataProto', mock_verl.DataProto):
            mock_verl.DataProto.from_dict.side_effect = None
            mock_verl.DataProto.from_dict.return_value = MagicMock()
            self.dm.put_experience({"x": _FakeTensor()}, [0])
            self.assertIsNotNone(self.dm._current_batch)

    # ---- update_metrics -----------------------------------------------------

    def test_update_metrics_no_cumulate(self):
        """Non-cumulative update overwrites existing key."""
        self.dm._metrics["loss"] = 1.0
        self.dm.update_metrics("loss", 2.0, cumulate=False)
        self.assertEqual(self.dm._metrics["loss"], 2.0)

    def test_update_metrics_cumulate_numeric(self):
        """Cumulative update adds to existing numeric value."""
        self.dm._metrics["loss"] = 1.0
        self.dm.update_metrics("loss", 0.5, cumulate=True)
        self.assertAlmostEqual(self.dm._metrics["loss"], 1.5)

    def test_update_metrics_cumulate_list_with_list(self):
        """Cumulative update extends existing list with a list."""
        self.dm._metrics["scores"] = [1, 2]
        self.dm.update_metrics("scores", [3, 4], cumulate=True)
        self.assertEqual(self.dm._metrics["scores"], [1, 2, 3, 4])

    def test_update_metrics_cumulate_list_with_scalar(self):
        """Cumulative update extends existing list with a scalar wrapped in list."""
        self.dm._metrics["scores"] = [1, 2]
        self.dm.update_metrics("scores", 3, cumulate=True)
        self.assertEqual(self.dm._metrics["scores"], [1, 2, 3])

    def test_update_metrics_cumulate_new_key(self):
        """Cumulative update on a non-existing key sets it directly."""
        self.dm.update_metrics("new_key", 42, cumulate=True)
        self.assertEqual(self.dm._metrics["new_key"], 42)

    # ---- get_metrics --------------------------------------------------------

    def test_get_metrics_returns_copy(self):
        """get_metrics returns a copy, not the original dict."""
        self.dm._metrics["a"] = 1
        m = self.dm.get_metrics()
        m["a"] = 999
        self.assertEqual(self.dm._metrics["a"], 1)

    # ---- clear --------------------------------------------------------------

    def test_clear(self):
        """clear resets both _current_batch and _metrics."""
        self.dm._current_batch = "something"
        self.dm._metrics = {"a": 1}
        self.dm.clear()
        self.assertIsNone(self.dm._current_batch)
        self.assertEqual(self.dm._metrics, {})

    # ---- get/set_current_batch ----------------------------------------------

    def test_get_set_current_batch(self):
        """set_current_batch / get_current_batch round-trip."""
        sentinel = MagicMock()
        self.dm.set_current_batch(sentinel)
        self.assertIs(self.dm.get_current_batch(), sentinel)

    # ---- set_pad_token_id ---------------------------------------------------

    def test_set_pad_token_id(self):
        """set_pad_token_id updates _pad_token_id."""
        self.dm.set_pad_token_id(42)
        self.assertEqual(self.dm._pad_token_id, 42)

    # ---- _dict_to_dataproto -------------------------------------------------

    def test_dict_to_dataproto_tensor_value(self):
        """Tensor values go into the tensors dict."""
        with patch.object(_verl_data_mod, 'np', mock_np), \
             patch.object(_verl_data_mod, 'torch', mock_torch), \
             patch.object(_verl_data_mod, 'DataProto', mock_verl.DataProto):
            tensor_val = _FakeTensor()
            mock_verl.DataProto.from_dict.return_value = "proto"

            result = self.dm._dict_to_dataproto({"t": tensor_val})

            call_kwargs = mock_verl.DataProto.from_dict.call_args
            tensors_arg = call_kwargs[1]["tensors"] if "tensors" in call_kwargs[1] else call_kwargs[0][0]
            self.assertIn("t", tensors_arg)
            self.assertIs(tensors_arg["t"], tensor_val)

    def test_dict_to_dataproto_ndarray_numeric(self):
        """Numeric ndarray values are converted via torch.from_numpy into tensors."""
        with patch.object(_verl_data_mod, 'np', mock_np), \
             patch.object(_verl_data_mod, 'torch', mock_torch), \
             patch.object(_verl_data_mod, 'DataProto', mock_verl.DataProto):
            arr = _FakeNdarray()
            arr.dtype = "float32"
            mock_torch.from_numpy.return_value = "converted_tensor"
            mock_verl.DataProto.from_dict.return_value = "proto"

            self.dm._dict_to_dataproto({"arr": arr})

            mock_torch.from_numpy.assert_called_with(arr)
            call_kwargs = mock_verl.DataProto.from_dict.call_args[1]
            self.assertIn("arr", call_kwargs["tensors"])

    def test_dict_to_dataproto_ndarray_object(self):
        """Object-dtype ndarray values go into non_tensors."""
        with patch.object(_verl_data_mod, 'np', mock_np), \
             patch.object(_verl_data_mod, 'torch', mock_torch), \
             patch.object(_verl_data_mod, 'DataProto', mock_verl.DataProto):
            arr = _FakeNdarray()
            arr.dtype = object
            mock_verl.DataProto.from_dict.return_value = "proto"

            self.dm._dict_to_dataproto({"obj": arr})

            call_kwargs = mock_verl.DataProto.from_dict.call_args[1]
            self.assertIn("obj", call_kwargs["non_tensors"])
            self.assertIs(call_kwargs["non_tensors"]["obj"], arr)

    def test_dict_to_dataproto_list_value(self):
        """List values are wrapped with np.array(dtype=object) into non_tensors."""
        with patch.object(_verl_data_mod, 'np', mock_np), \
             patch.object(_verl_data_mod, 'torch', mock_torch), \
             patch.object(_verl_data_mod, 'DataProto', mock_verl.DataProto):
            mock_np.array.return_value = "np_arr"
            mock_verl.DataProto.from_dict.return_value = "proto"

            self.dm._dict_to_dataproto({"lst": [1, 2, 3]})

            mock_np.array.assert_called_with([1, 2, 3], dtype=object)
            call_kwargs = mock_verl.DataProto.from_dict.call_args[1]
            self.assertIn("lst", call_kwargs["non_tensors"])

    def test_dict_to_dataproto_scalar_value(self):
        """Scalar values are wrapped with np.array([value], dtype=object) into non_tensors."""
        with patch.object(_verl_data_mod, 'np', mock_np), \
             patch.object(_verl_data_mod, 'torch', mock_torch), \
             patch.object(_verl_data_mod, 'DataProto', mock_verl.DataProto):
            mock_np.array.return_value = "np_scalar"
            mock_verl.DataProto.from_dict.return_value = "proto"

            self.dm._dict_to_dataproto({"s": 42})

            mock_np.array.assert_called_with([42], dtype=object)
            call_kwargs = mock_verl.DataProto.from_dict.call_args[1]
            self.assertIn("s", call_kwargs["non_tensors"])


if __name__ == '__main__':
    unittest.main()
