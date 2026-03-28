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
import unittest

from base import SystemTestBase


class TestSmokeBackendDispatch(SystemTestBase):
    """Smoke tests for backend dispatch."""

    def test_unknown_backend_rejected(self):
        """Unknown backend is rejected."""
        config_path = self.get_fixture_path("unknown_backend.yaml")
        result = self.run_cli(config_path)

        self.assertExitFailure(result, "Expected non-zero exit code for unknown backend")
        self.assertLogContains(
            result,
            "train_backend unknown is not supported",
            "Expected unsupported backend error message",
        )

    def test_msrl_section_missing_rejected(self):
        """mindspeed_rl section missing is rejected."""
        config_path = self.get_fixture_path("missing_msrl_section.yaml")
        result = self.run_cli(config_path)

        self.assertExitFailure(result)
        self.assertLogContains(
            result,
            "mindspeed_rl config section is required when train_backend is 'mindspeed_rl'",
            "Expected specific error for missing mindspeed_rl section",
        )

class TestSmokePathValidation(SystemTestBase):
    """Smoke tests for path validation."""

    def test_tokenizer_path_validation_failure(self):
        """Tokenizer path validation failure is diagnosable."""
        config_path = self.get_fixture_path("invalid_tokenizer_path.yaml")
        result = self.run_cli(config_path)

        self.assertExitFailure(result)
        self.assertLogContainsAll(
            result,
            ["tokenizer_name_or_path", " Path is not a string or path is not existed"],
            "Expected path validation error",
        )

class TestSmokeModelSupport(SystemTestBase):
    """Smoke tests for model support."""

    def test_msrl_unsupported_model_rejected(self):
        """MSRL unsupported model name is rejected."""
        config_path = self.get_fixture_path("unsupported_model.yaml")
        result = self.run_cli(config_path)

        self.assertExitFailure(result)
        self.assertLogContains(result, "Model unsupport is not supported", "Expected unsupported model error")

class TestSmokeMinimalTraining(SystemTestBase):
    """Smoke tests for minimal training runs."""

    def test_msrl_minimal_training(self):
        """MSRL minimal training step (1 iter) can run."""
        config_path = self.get_fixture_path("valid_msrl_config.yaml")
        result = self.run_cli(config_path)

        self.assertExitSuccess(result)
        self.assertLogContains(result, "training successfully!", "Expected training success log message")


if __name__ == "__main__":
    unittest.main()