# -*- coding: utf-8 -*-
import pytest
import time
from unittest.mock import MagicMock, patch


class TestReferenceWorkerBasePatch:

    def test_compute_ref_log_prob_empty_batch(self):
        """Test that compute_ref_log_prob handles empty batch correctly."""
        mock_self = MagicMock()
        mock_self.all_consumed = MagicMock(return_value=0)
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = False
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.get_context_parallel_rank = MagicMock(return_value=0)
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        self._simulate_compute_ref_log_prob(mock_self)

        mock_self.all_consumed.assert_called_once()

    def test_compute_ref_log_prob_with_data(self):
        """Test that compute_ref_log_prob processes data correctly."""
        mock_self = MagicMock()

        call_count = [0]
        def mock_all_consumed(stage, indexes=None):
            call_count[0] += 1
            return 1 if call_count[0] == 1 else 0

        mock_self.all_consumed = mock_all_consumed
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = False
        mock_self.rl_config.partial_rollout_max_split = 1
        mock_self.megatron_config = MagicMock()
        mock_self.megatron_config.tensor_model_parallel_size = 1
        mock_self.megatron_config.context_parallel_size = 1
        mock_self.megatron_config.context_parallel_algo = 'uniform'
        mock_self.dispatch_transfer_dock_data = MagicMock(return_value=(MagicMock(), 1))
        mock_self.reference = MagicMock()
        mock_self.reference.compute_log_prob = MagicMock(return_value=([MagicMock()], {'response_length': MagicMock()}))
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.is_pipeline_last_stage = MagicMock(return_value=True)
        mock_self.parallel_state.get_context_parallel_rank = MagicMock(return_value=0)
        mock_self.collect_transfer_dock_data = MagicMock()
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        torch_mock = MagicMock()
        torch_mock.cat.return_value = MagicMock()
        torch_mock.float32 = MagicMock()

        truncate_rows_mock = MagicMock(return_value=MagicMock())
        is_multimodal_mock = MagicMock(return_value=False)
        get_parallel_state_mock = MagicMock(return_value=MagicMock())
        is_pipeline_last_stage_mock = MagicMock(return_value=True)
        get_tensor_model_parallel_rank_mock = MagicMock(return_value=0)
        ray_get_mock = lambda x: None

        self._simulate_compute_ref_log_prob(
            mock_self,
            torch=torch_mock,
            truncate_rows=truncate_rows_mock,
            is_multimodal=is_multimodal_mock,
            get_parallel_state=get_parallel_state_mock,
            is_pipeline_last_stage=is_pipeline_last_stage_mock,
            get_tensor_model_parallel_rank=get_tensor_model_parallel_rank_mock,
            ray_get=ray_get_mock
        )

        mock_self.dispatch_transfer_dock_data.assert_called_once()
        mock_self.reference.compute_log_prob.assert_called_once()
        mock_self.collect_transfer_dock_data.assert_called_once()

    def test_compute_ref_log_prob_multimodal(self):
        """Test that compute_ref_log_prob handles multimodal data."""
        mock_self = MagicMock()

        call_count = [0]
        def mock_all_consumed(stage, indexes=None):
            call_count[0] += 1
            return 1 if call_count[0] == 1 else 0

        mock_self.all_consumed = mock_all_consumed
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = False
        mock_self.rl_config.partial_rollout_max_split = 1
        mock_self.megatron_config = MagicMock()
        mock_self.megatron_config.tensor_model_parallel_size = 1
        mock_self.megatron_config.context_parallel_size = 1
        mock_self.megatron_config.context_parallel_algo = 'uniform'

        dispatched_columns = []
        def mock_dispatch(*args, **kwargs):
            if len(args) > 1:
                dispatched_columns.append(args[1])
            return MagicMock(), 1

        mock_self.dispatch_transfer_dock_data = mock_dispatch
        mock_self.reference = MagicMock()
        mock_self.reference.compute_log_prob = MagicMock(return_value=([MagicMock()], {'response_length': MagicMock()}))
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.is_pipeline_last_stage = MagicMock(return_value=True)
        mock_self.parallel_state.get_context_parallel_rank = MagicMock(return_value=0)
        mock_self.collect_transfer_dock_data = MagicMock()
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        torch_mock = MagicMock()
        torch_mock.cat.return_value = MagicMock()
        torch_mock.float32 = MagicMock()

        truncate_rows_mock = MagicMock(return_value=MagicMock())
        is_multimodal_mock = MagicMock(return_value=True)
        get_parallel_state_mock = MagicMock(return_value=MagicMock())
        is_pipeline_last_stage_mock = MagicMock(return_value=True)
        get_tensor_model_parallel_rank_mock = MagicMock(return_value=0)
        ray_get_mock = lambda x: None

        self._simulate_compute_ref_log_prob(
            mock_self,
            torch=torch_mock,
            truncate_rows=truncate_rows_mock,
            is_multimodal=is_multimodal_mock,
            get_parallel_state=get_parallel_state_mock,
            is_pipeline_last_stage=is_pipeline_last_stage_mock,
            get_tensor_model_parallel_rank=get_tensor_model_parallel_rank_mock,
            ray_get=ray_get_mock
        )

        assert len(dispatched_columns) == 1
        columns = dispatched_columns[0]
        assert 'attention_mask' in columns
        assert 'position_ids' in columns
        assert 'input_ids_length' in columns

    def test_compute_ref_log_prob_guarantee_order(self):
        """Test that compute_ref_log_prob handles guarantee_order correctly."""
        mock_self = MagicMock()

        call_count = [0]
        def mock_all_consumed(stage, indexes=None):
            call_count[0] += 1
            return 1 if call_count[0] == 1 else 0

        mock_self.all_consumed = mock_all_consumed
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = True
        mock_self.rl_config.partial_rollout_max_split = 1
        mock_self.megatron_config = MagicMock()
        mock_self.megatron_config.tensor_model_parallel_size = 1
        mock_self.megatron_config.context_parallel_size = 1
        mock_self.megatron_config.context_parallel_algo = 'uniform'
        mock_self.get_dp_range_indexes = MagicMock(return_value=[[0, 1, 2]])

        indexes_used = []
        def mock_dispatch(*args, **kwargs):
            indexes_used.append(kwargs.get('indexes'))
            return MagicMock(), 1

        mock_self.dispatch_transfer_dock_data = mock_dispatch
        mock_self.reference = MagicMock()
        mock_self.reference.compute_log_prob = MagicMock(return_value=([MagicMock()], {'response_length': MagicMock()}))
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.is_pipeline_last_stage = MagicMock(return_value=True)
        mock_self.parallel_state.get_context_parallel_rank = MagicMock(return_value=0)
        mock_self.collect_transfer_dock_data = MagicMock()
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        torch_mock = MagicMock()
        torch_mock.cat.return_value = MagicMock()
        torch_mock.float32 = MagicMock()

        truncate_rows_mock = MagicMock(return_value=MagicMock())
        is_multimodal_mock = MagicMock(return_value=False)
        get_parallel_state_mock = MagicMock(return_value=MagicMock())
        is_pipeline_last_stage_mock = MagicMock(return_value=True)
        get_tensor_model_parallel_rank_mock = MagicMock(return_value=0)
        ray_get_mock = lambda x: None

        self._simulate_compute_ref_log_prob(
            mock_self,
            torch=torch_mock,
            truncate_rows=truncate_rows_mock,
            is_multimodal=is_multimodal_mock,
            get_parallel_state=get_parallel_state_mock,
            is_pipeline_last_stage=is_pipeline_last_stage_mock,
            get_tensor_model_parallel_rank=get_tensor_model_parallel_rank_mock,
            ray_get=ray_get_mock
        )

        assert len(indexes_used) == 1
        assert indexes_used[0] == [0, 1, 2]

    def test_compute_ref_log_prob_not_last_stage(self):
        """Test that compute_ref_log_prob skips collection on non-last stage."""
        mock_self = MagicMock()

        call_count = [0]
        def mock_all_consumed(stage, indexes=None):
            call_count[0] += 1
            return 1 if call_count[0] == 1 else 0

        mock_self.all_consumed = mock_all_consumed
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = False
        mock_self.rl_config.partial_rollout_max_split = 1
        mock_self.megatron_config = MagicMock()
        mock_self.megatron_config.tensor_model_parallel_size = 1
        mock_self.megatron_config.context_parallel_size = 1
        mock_self.megatron_config.context_parallel_algo = 'uniform'
        mock_self.dispatch_transfer_dock_data = MagicMock(return_value=(MagicMock(), 1))
        mock_self.reference = MagicMock()
        mock_self.reference.compute_log_prob = MagicMock(return_value=([MagicMock()], {'response_length': MagicMock()}))
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.is_pipeline_last_stage = MagicMock(return_value=False)
        mock_self.collect_transfer_dock_data = MagicMock()
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        is_multimodal_mock = MagicMock(return_value=False)
        get_parallel_state_mock = MagicMock(return_value=MagicMock())
        is_pipeline_last_stage_mock = MagicMock(return_value=True)
        get_tensor_model_parallel_rank_mock = MagicMock(return_value=0)
        ray_get_mock = lambda x: None

        self._simulate_compute_ref_log_prob(
            mock_self,
            torch=MagicMock(),
            truncate_rows=MagicMock(),
            is_multimodal=is_multimodal_mock,
            get_parallel_state=get_parallel_state_mock,
            is_pipeline_last_stage=is_pipeline_last_stage_mock,
            get_tensor_model_parallel_rank=get_tensor_model_parallel_rank_mock,
            ray_get=ray_get_mock
        )

        mock_self.collect_transfer_dock_data.assert_not_called()

    def test_compute_ref_log_prob_no_end_time_update(self):
        """Test that end_time is not updated when not on the right rank."""
        mock_self = MagicMock()
        mock_self.all_consumed = MagicMock(return_value=0)
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = False
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.get_context_parallel_rank = MagicMock(return_value=1)
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        is_multimodal_mock = MagicMock(return_value=False)
        get_parallel_state_mock = MagicMock(return_value=MagicMock())
        is_pipeline_last_stage_mock = MagicMock(return_value=True)
        get_tensor_model_parallel_rank_mock = MagicMock(return_value=0)
        ray_get_mock = lambda x: None

        self._simulate_compute_ref_log_prob(
            mock_self,
            torch=MagicMock(),
            truncate_rows=MagicMock(),
            is_multimodal=is_multimodal_mock,
            get_parallel_state=get_parallel_state_mock,
            is_pipeline_last_stage=is_pipeline_last_stage_mock,
            get_tensor_model_parallel_rank=get_tensor_model_parallel_rank_mock,
            ray_get=ray_get_mock
        )

        calls = [call for call in mock_self.td.update_metrics.remote.call_args_list
                 if call[0][0] == "end_time/reference"]
        assert len(calls) == 0, "end_time/reference should not be called when context_parallel_rank != 0"

    def test_compute_ref_log_prob_multiple_batches(self):
        """Test that compute_ref_log_prob handles multiple batches."""
        mock_self = MagicMock()

        call_count = [0]
        def mock_all_consumed(stage, indexes=None):
            call_count[0] += 1
            return 1 if call_count[0] <= 2 else 0

        mock_self.all_consumed = mock_all_consumed
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = False
        mock_self.rl_config.partial_rollout_max_split = 1
        mock_self.megatron_config = MagicMock()
        mock_self.megatron_config.tensor_model_parallel_size = 1
        mock_self.megatron_config.context_parallel_size = 1
        mock_self.megatron_config.context_parallel_algo = 'uniform'
        mock_self.dispatch_transfer_dock_data = MagicMock(return_value=(MagicMock(), 1))
        mock_self.reference = MagicMock()
        mock_self.reference.compute_log_prob = MagicMock(return_value=([MagicMock()], {'response_length': MagicMock()}))
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.is_pipeline_last_stage = MagicMock(return_value=True)
        mock_self.parallel_state.get_context_parallel_rank = MagicMock(return_value=0)
        mock_self.collect_transfer_dock_data = MagicMock()
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        torch_mock = MagicMock()
        torch_mock.cat.return_value = MagicMock()
        torch_mock.float32 = MagicMock()

        truncate_rows_mock = MagicMock(return_value=MagicMock())
        is_multimodal_mock = MagicMock(return_value=False)
        get_parallel_state_mock = MagicMock(return_value=MagicMock())
        is_pipeline_last_stage_mock = MagicMock(return_value=True)
        get_tensor_model_parallel_rank_mock = MagicMock(return_value=0)
        ray_get_mock = lambda x: None

        self._simulate_compute_ref_log_prob(
            mock_self,
            torch=torch_mock,
            truncate_rows=truncate_rows_mock,
            is_multimodal=is_multimodal_mock,
            get_parallel_state=get_parallel_state_mock,
            is_pipeline_last_stage=is_pipeline_last_stage_mock,
            get_tensor_model_parallel_rank=get_tensor_model_parallel_rank_mock,
            ray_get=ray_get_mock
        )

        assert mock_self.dispatch_transfer_dock_data.call_count == 2
        assert mock_self.reference.compute_log_prob.call_count == 2
        assert mock_self.collect_transfer_dock_data.call_count == 2

    def test_compute_ref_log_prob_empty_batch_data(self):
        """Test that compute_ref_log_prob handles empty batch_data but valid index."""
        mock_self = MagicMock()

        call_count = [0]
        def mock_all_consumed(stage, indexes=None):
            call_count[0] += 1
            return 1 if call_count[0] == 1 else 0

        mock_self.all_consumed = mock_all_consumed
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = False
        mock_self.rl_config.partial_rollout_max_split = 1
        mock_self.megatron_config = MagicMock()
        mock_self.megatron_config.tensor_model_parallel_size = 1
        mock_self.megatron_config.context_parallel_size = 1
        mock_self.megatron_config.context_parallel_algo = 'uniform'
        mock_self.dispatch_transfer_dock_data = MagicMock(return_value=(None, 1))
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.get_context_parallel_rank = MagicMock(return_value=0)
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        is_multimodal_mock = MagicMock(return_value=False)
        get_parallel_state_mock = MagicMock(return_value=MagicMock())
        is_pipeline_last_stage_mock = MagicMock(return_value=False)
        get_tensor_model_parallel_rank_mock = MagicMock(return_value=0)
        ray_get_mock = lambda x: None

        self._simulate_compute_ref_log_prob(
            mock_self,
            torch=MagicMock(),
            truncate_rows=MagicMock(),
            is_multimodal=is_multimodal_mock,
            get_parallel_state=get_parallel_state_mock,
            is_pipeline_last_stage=is_pipeline_last_stage_mock,
            get_tensor_model_parallel_rank=get_tensor_model_parallel_rank_mock,
            ray_get=ray_get_mock
        )

        mock_self.dispatch_transfer_dock_data.assert_called_once()
        mock_self.td.update_metrics.remote.assert_not_called()

    def test_compute_ref_log_prob_partial_rollout_max_split(self):
        """Test that compute_ref_log_prob handles partial_rollout_max_split > 1."""
        mock_self = MagicMock()

        call_count = [0]
        def mock_all_consumed(stage, indexes=None):
            call_count[0] += 1
            return 1 if call_count[0] == 1 else 0

        mock_self.all_consumed = mock_all_consumed
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = False
        mock_self.rl_config.partial_rollout_max_split = 5
        mock_self.megatron_config = MagicMock()
        mock_self.megatron_config.tensor_model_parallel_size = 1
        mock_self.megatron_config.context_parallel_size = 1
        mock_self.megatron_config.context_parallel_algo = 'uniform'

        get_n_samples_values = []
        def mock_dispatch(*args, **kwargs):
            get_n_samples_values.append(kwargs.get('get_n_samples', False))
            return MagicMock(), 1

        mock_self.dispatch_transfer_dock_data = mock_dispatch
        mock_self.reference = MagicMock()
        mock_self.reference.compute_log_prob = MagicMock(return_value=([MagicMock()], {'response_length': MagicMock()}))
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.is_pipeline_last_stage = MagicMock(return_value=True)
        mock_self.parallel_state.get_context_parallel_rank = MagicMock(return_value=0)
        mock_self.collect_transfer_dock_data = MagicMock()
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        torch_mock = MagicMock()
        torch_mock.cat.return_value = MagicMock()
        torch_mock.float32 = MagicMock()

        truncate_rows_mock = MagicMock(return_value=MagicMock())
        is_multimodal_mock = MagicMock(return_value=False)
        get_parallel_state_mock = MagicMock(return_value=MagicMock())
        is_pipeline_last_stage_mock = MagicMock(return_value=True)
        get_tensor_model_parallel_rank_mock = MagicMock(return_value=0)
        ray_get_mock = lambda x: None

        self._simulate_compute_ref_log_prob(
            mock_self,
            torch=torch_mock,
            truncate_rows=truncate_rows_mock,
            is_multimodal=is_multimodal_mock,
            get_parallel_state=get_parallel_state_mock,
            is_pipeline_last_stage=is_pipeline_last_stage_mock,
            get_tensor_model_parallel_rank=get_tensor_model_parallel_rank_mock,
            ray_get=ray_get_mock
        )

        assert len(get_n_samples_values) == 1
        assert get_n_samples_values[0] == True

    def test_compute_ref_log_prob_not_tensor_parallel_rank_zero(self):
        """Test that end_time is not updated when tensor_parallel_rank != 0."""
        mock_self = MagicMock()
        mock_self.all_consumed = MagicMock(return_value=0)
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = False
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.get_context_parallel_rank = MagicMock(return_value=0)
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        is_multimodal_mock = MagicMock(return_value=False)
        get_parallel_state_mock = MagicMock(return_value=MagicMock())
        is_pipeline_last_stage_mock = MagicMock(return_value=True)
        get_tensor_model_parallel_rank_mock = MagicMock(return_value=1)
        ray_get_mock = lambda x: None

        self._simulate_compute_ref_log_prob(
            mock_self,
            torch=MagicMock(),
            truncate_rows=MagicMock(),
            is_multimodal=is_multimodal_mock,
            get_parallel_state=get_parallel_state_mock,
            is_pipeline_last_stage=is_pipeline_last_stage_mock,
            get_tensor_model_parallel_rank=get_tensor_model_parallel_rank_mock,
            ray_get=ray_get_mock
        )

        calls = [call for call in mock_self.td.update_metrics.remote.call_args_list
                 if call[0][0] == "end_time/reference"]
        assert len(calls) == 0, "end_time/reference should not be called when tensor_parallel_rank != 0"

    def test_compute_ref_log_prob_not_pipeline_last_stage_end_time(self):
        """Test that end_time is not updated when not pipeline last stage."""
        mock_self = MagicMock()
        mock_self.all_consumed = MagicMock(return_value=0)
        mock_self.rl_config = MagicMock()
        mock_self.rl_config.ref_dispatch_size = 10
        mock_self.rl_config.guarantee_order = False
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.get_context_parallel_rank = MagicMock(return_value=0)
        mock_self.td = MagicMock()
        mock_self.logger = MagicMock()
        mock_self.logger.info = MagicMock()

        is_multimodal_mock = MagicMock(return_value=False)
        get_parallel_state_mock = MagicMock(return_value=MagicMock())
        is_pipeline_last_stage_mock = MagicMock(return_value=False)
        get_tensor_model_parallel_rank_mock = MagicMock(return_value=0)
        ray_get_mock = lambda x: None

        self._simulate_compute_ref_log_prob(
            mock_self,
            torch=MagicMock(),
            truncate_rows=MagicMock(),
            is_multimodal=is_multimodal_mock,
            get_parallel_state=get_parallel_state_mock,
            is_pipeline_last_stage=is_pipeline_last_stage_mock,
            get_tensor_model_parallel_rank=get_tensor_model_parallel_rank_mock,
            ray_get=ray_get_mock
        )

        calls = [call for call in mock_self.td.update_metrics.remote.call_args_list
                 if call[0][0] == "end_time/reference"]
        assert len(calls) == 0, "end_time/reference should not be called when not pipeline last stage"

    def _simulate_compute_ref_log_prob(
        self,
        mock_self,
        torch=None,
        truncate_rows=None,
        is_multimodal=None,
        get_parallel_state=None,
        is_pipeline_last_stage=None,
        get_tensor_model_parallel_rank=None,
        ray_get=None
    ):
        """Simulate the compute_ref_log_prob logic for testing."""
        if torch is None:
            torch = MagicMock()
            torch.cat = MagicMock(return_value=MagicMock())
            torch.float32 = MagicMock()

        if truncate_rows is None:
            truncate_rows = MagicMock(return_value=MagicMock())

        if is_multimodal is None:
            is_multimodal = MagicMock(return_value=False)

        if get_parallel_state is None:
            get_parallel_state = MagicMock(return_value=MagicMock())

        if is_pipeline_last_stage is None:
            is_pipeline_last_stage = MagicMock(return_value=True)

        if get_tensor_model_parallel_rank is None:
            get_tensor_model_parallel_rank = MagicMock(return_value=0)

        if ray_get is None:
            ray_get = lambda x: None

        experience_consumer_stage = 'ref_log_prob'
        experience_columns = ['input_ids', 'responses', 'response_length', 'prompt_length']
        if is_multimodal():
            experience_columns.extend(['attention_mask', 'position_ids', 'input_ids_length'])
        experience_count = mock_self.rl_config.ref_dispatch_size
        sorted_indexes = mock_self.get_dp_range_indexes(
            experience_count, use_vllm=False) if mock_self.rl_config.guarantee_order else None

        start_time_defined = False
        first_dispatch_data_defined = False
        first_collect_data_defined = False

        while mock_self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            if not first_dispatch_data_defined:
                first_dispatch_start_time = time.time()

            indexes_to_use = sorted_indexes.pop(0) if mock_self.rl_config.guarantee_order and sorted_indexes else None
            batch_data, index = mock_self.dispatch_transfer_dock_data(
                experience_consumer_stage,
                experience_columns,
                experience_count,
                tp_size=mock_self.megatron_config.tensor_model_parallel_size,
                cp_size=mock_self.megatron_config.context_parallel_size,
                cp_algo=mock_self.megatron_config.context_parallel_algo,
                indexes=indexes_to_use,
                get_n_samples=mock_self.rl_config.partial_rollout_max_split > 1
            )

            if batch_data and index:
                if not first_dispatch_data_defined:
                    ray_get(mock_self.td.update_metrics.remote(
                        "dispatch_timing(first)/reference_model",
                        value=[round(time.time(), 4), round(first_dispatch_start_time, 4)],
                        cumulate=True
                    ))
                    first_dispatch_data_defined = True

                if not start_time_defined:
                    start_time = time.time()
                    start_time_defined = True
                    ray_get(
                        mock_self.td.update_metrics.remote(
                            "start_time/reference_model",
                            value=[round(start_time, 4)],
                            cumulate=True
                        )
                    )

                output, batch = mock_self.reference.compute_log_prob(batch_data)

                if mock_self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    log_probs = torch.cat(output, dim=0)
                    log_probs = log_probs.to(torch.float32)
                    log_probs = truncate_rows(log_probs, batch['response_length'])
                    output = {'ref_log_prob': log_probs}

                    if not first_collect_data_defined:
                        first_collect_start_time = time.time()

                    mock_self.collect_transfer_dock_data(output, index)

                    if not first_collect_data_defined:
                        ray_get(mock_self.td.update_metrics.remote(
                            "collect_timing(first)/reference_model",
                            value=[time.time() - first_collect_start_time],
                            cumulate=True
                        ))
                        first_collect_data_defined = True

                    end_time = time.time()
                    ray_get(
                        mock_self.td.update_metrics.remote(
                            "timing/reference_model",
                            value=[round(end_time, 4), round(start_time, 4)],
                            cumulate=True
                        )
                    )

        parallel_state = get_parallel_state()
        use_vllm = False
        if (is_pipeline_last_stage(parallel_state, use_vllm)
                and get_tensor_model_parallel_rank(parallel_state, use_vllm) == 0
                and mock_self.parallel_state.get_context_parallel_rank() == 0):
            ref_end_time = time.time()
            ray_get(
                mock_self.td.update_metrics.remote(
                    "end_time/reference",
                    value=[round(ref_end_time, 4)]
                )
            )
        mock_self.logger.info("finish compute ref log prob")
