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


class Test01ImportModule(SystemTestBase):
    """Test 1: Verify module can be imported successfully"""

    def test_import_agentic_rl(self):
        """Importing agentic_rl module should succeed"""
        try:
            import aura.aura
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import agentic_rl: {e}")


class Test02MissingConfigName(SystemTestBase):
    """Test 2: Verify error handling when --config-name is missing"""

    def test_missing_config_name(self):
        """Omitting --config-name should fail with error message"""
        result = self.cli_runner.run_without_args()
        self.assertExitFailure(result)
        self.assertLogContains(result, "--config-name", "Expected error message to mention '--config-name'")


class Test03NonExistentConfig(SystemTestBase):
    """Test 3: Verify error handling when config file does not exist"""

    def test_non_existent_config(self):
        """Specifying non-existent config file should fail"""
        result = self.run_cli("non_existent_config_file.yaml")
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["Cannot find primary config"],
            "Expected error message about missing config file"
        )


class Test04InvalidMode(SystemTestBase):
    """Test 4: Verify error handling for invalid mode"""

    def test_invalid_mode(self):
        """Specifying invalid mode in config file should fail"""
        result = self.run_cli("presmoke_invalid_mode.yaml")
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["not supported", "invalid", "mode", "不支持"],
            "Expected error message about invalid mode"
        )


class Test05MissingAgenticAISection(SystemTestBase):
    """Test 5: Verify error handling when agentic_ai section is missing"""

    def test_missing_agentic_ai_section(self):
        """Missing agentic_ai section in config file should fail"""
        result = self.run_cli("presmoke_missing_agentic_ai.yaml")
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["agentic_ai", "required", "missing", "必需"],
            "Expected error message about missing agentic_ai section"
        )


class Test06InvalidTrainBackend(SystemTestBase):
    """Test 6: Verify error handling for invalid train_backend"""

    def test_invalid_train_backend(self):
        """Specifying invalid train_backend in config file should fail"""
        result = self.run_cli("presmoke_invalid_train_backend.yaml")
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["train_backend", "not supported", "unknown", "不支持"],
            "Expected error message about invalid train_backend"
        )


class Test07MissingTrainInstances(SystemTestBase):
    """Test 7: Verify error handling when train_instances is missing in direct mode"""

    def test_missing_train_instances(self):
        """Missing train_instances in direct mode should fail"""
        result = self.run_cli("presmoke_missing_train_instances.yaml")
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["train_instances", "entrypoints", "train"],
            "Expected error message about missing train_instances"
        )


class Test08InvalidClusterMode(SystemTestBase):
    """Test 8: Verify error handling for invalid cluster_mode"""

    def test_invalid_cluster_mode(self):
        """Specifying invalid cluster_mode in config file should fail"""
        result = self.run_cli("presmoke_invalid_cluster_mode.yaml")
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["cluster_mode", "not supported", "invalid", "不支持"],
            "Expected error message about invalid cluster_mode"
        )


class Test09InvalidTrainEngine(SystemTestBase):
    """Test 9: Verify error handling for invalid train_engine"""

    def test_invalid_train_engine(self):
        """Specifying invalid train_engine in config file should fail"""
        result = self.run_cli("presmoke_invalid_train_engine.yaml")
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["train_engine", "not supported", "unknown", "不支持"],
            "Expected error message about invalid train_engine"
        )


class Test10ValidDirectModeConfig(SystemTestBase):
    """Test 10: Verify valid direct mode config can start normally"""
    cli_timeout: int = 600

    def test_valid_direct_mode_config(self):
        """Valid direct mode config should start normally (verify config parsing success)"""
        result = self.run_cli("presmoke_valid_direct_mode.yaml")
        self.assertLogContainsAny(
            result,
            ["Agentic AI Config", "mode", "direct"],
            "Expected config loading success message"
        )


if __name__ == "__main__":
    unittest.main()
