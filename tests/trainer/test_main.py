#!/usr/bin/env python3
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

import pytest
import yaml
from ray.exceptions import RayError


class TestMain:

    @pytest.fixture(scope="class")
    def add_params(self):
        self.raw_argv = sys.argv
        sys.argv = ["", "--config-path", "test"]
        yield
        sys.argv = self.raw_argv

    @pytest.fixture(scope="class")
    def patch_modules(self, add_params):
        with patch.dict(sys.modules, {"agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo": MagicMock()}):
            yield

    @pytest.fixture(scope="class")
    def mock_for_main(self, patch_modules):
        with patch('agentic_rl.trainer.main.FileCheck.check_data_path_is_valid') as mock_check_data_path, \
                patch('agentic_rl.trainer.main.FileCheck.check_file_size') as mock_check_file_size, \
                patch('builtins.open') as mock_open, \
                patch('yaml.safe_load') as mock_yaml_load, \
                patch('ray.is_initialized') as mock_ray_is_initialized, \
                patch('ray.init') as mock_ray_init, \
                patch('ray.get') as mock_ray_get, \
                patch('agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo.train') as mock_train, \
                patch('agentic_rl.base.utils.logger_patch.patch'):
            def fake_ray_get(data):
                return data

            mock_ray_get.side_effect = fake_ray_get

            mock_yaml_load.return_value = {}

            patches = {
                "mock_check_data_path": mock_check_data_path,
                "mock_check_file_size": mock_check_file_size,
                "mock_open": mock_open,
                "mock_yaml_load": mock_yaml_load,
                "mock_ray_is_initialized": mock_ray_is_initialized,
                "mock_ray_init": mock_ray_init,
                "mock_ray_get": mock_ray_get,
                "mock_train": mock_train
            }

            import agentic_rl.trainer.main as main_dir
            from agentic_rl.trainer.main import main
            from agentic_rl.base.log.loggers import Loggers

            main_dir.logger = Loggers()

            yield main, patches

    def test_main_config_path_check_data_path_failed(self, mock_for_main, caplog):
        main, patches = mock_for_main
        mock_check_data_path = patches["mock_check_data_path"]

        mock_check_data_path.side_effect = ValueError("No such file or directory")
        with pytest.raises(SystemExit):
            with caplog.at_level('ERROR', ):
                main()
        assert "Checking config_path failed with value not correct" in caplog.text

        mock_check_data_path.side_effect = TypeError("error")
        with pytest.raises(SystemExit):
            with caplog.at_level('ERROR', ):
                main()
        assert "Checking config_path failed with type not correct" in caplog.text

        mock_check_data_path.side_effect = Exception("error")
        with pytest.raises(SystemExit):
            with caplog.at_level('ERROR', ):
                main()
        assert "Unexpected error occurred when checking config_path" in caplog.text

        mock_check_data_path.side_effect = None

    def test_main_load_config_failed(self, mock_for_main, caplog):
        main, patches = mock_for_main
        mock_yaml_load = patches["mock_yaml_load"]

        mock_yaml_load.side_effect = yaml.YAMLError("error")
        with pytest.raises(SystemExit):
            with caplog.at_level('ERROR', ):
                main()
        assert "load config failed of yaml content" in caplog.text

        mock_yaml_load.side_effect = RecursionError("error")
        with pytest.raises(SystemExit):
            with caplog.at_level('ERROR', ):
                main()
        assert "failed to parse yaml file, nesting depth is too deep or a circular alias was detected" in caplog.text

        mock_yaml_load.side_effect = Exception("error")
        with pytest.raises(SystemExit):
            with caplog.at_level('ERROR', ):
                main()
        assert "Unexpected error occurred when load config" in caplog.text

        mock_yaml_load.side_effect = None

    def test_main_import_ray_failed(self, mock_for_main, caplog, monkeypatch):
        main, _ = mock_for_main
        monkeypatch.setitem(sys.modules, 'ray', None)
        with pytest.raises(SystemExit):
            with caplog.at_level('ERROR', ):
                main()
        assert "ray is not installed, please install ray first" in caplog.text

    def test_main_with_train_failed(self, mock_for_main, caplog):
        main, patches = mock_for_main
        mock_ray_is_initialized = patches["mock_ray_is_initialized"]
        mock_train = patches["mock_train"]

        mock_ray_is_initialized.return_value = True
        with pytest.raises(SystemExit):
            with caplog.at_level('ERROR', ):
                main()
        assert "ray should be initialized by agentic_rl, but has already been initialized." in caplog.text

        mock_ray_is_initialized.return_value = False
        mock_train.remote.side_effect = RayError("error")
        with pytest.raises(SystemExit):
            with caplog.at_level('ERROR', ):
                main()
        assert "Training using mindspeed_rl failed with ray" in caplog.text

        mock_train.remote.side_effect = Exception("error")
        with pytest.raises(SystemExit):
            with caplog.at_level('ERROR', ):
                main()
        assert "Unexpected error occurred when training using mindspeed_rl" in caplog.text

        mock_train.remote.side_effect = None

    def test_main_success(self, mock_for_main, caplog):
        main, patches = mock_for_main
        mock_ray_is_initialized = patches["mock_ray_is_initialized"]
        mock_ray_init = patches["mock_ray_init"]
        mock_ray_get = patches["mock_ray_get"]

        mock_ray_init.reset_mock()
        mock_ray_get.reset_mock()

        mock_ray_is_initialized.return_value = False

        with caplog.at_level('INFO', ):
            main()
        assert "start initializing local ray cluster, when the ray cluster is not initialized" in caplog.text
        mock_ray_init.assert_called_once()
        mock_ray_get.assert_called_once()
