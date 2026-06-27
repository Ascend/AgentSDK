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

from base import SystemTestBase, get_local_ip


class Test01ImportModule(SystemTestBase):
    """Test 1: Verify module can be imported successfully"""

    def test_import_aura(self):
        """Importing aura module should succeed"""
        try:
            import aura

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import aura: {e}")


class Test02MissingBaseConf(SystemTestBase):
    """Test 2: Verify error handling when base.conf is missing"""

    def test_missing_base_conf(self):
        """Removing base.conf should fail with error message about the config file"""
        base_conf = self.project_root / "aura" / "configs" / "base.conf"
        self.delete_file(base_conf)
        result = self.run_cli()
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["base.conf", "is not exist", "config file"],
            "Expected error message about missing base.conf",
        )


class Test03InvalidWorkMode(SystemTestBase):
    """Test 3: Verify error handling when work_mode is invalid in base.conf"""

    def test_invalid_work_mode(self):
        """Setting work_mode to an unsupported value should fail with error message"""
        base_conf = self.project_root / "aura" / "configs" / "base.conf"
        self.replace_in_file(base_conf, "work_mode=one_step_off", "work_mode=invalid_mode")
        result = self.run_cli()
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["invalid WORK_MODE", "invalid_mode", "supported values"],
            "Expected error message about invalid work_mode",
        )


class Test04EmptyTrainConfigName(SystemTestBase):
    """Test 4: Verify error handling when train_config_name is empty in base.conf"""

    def test_empty_train_config_name(self):
        """Setting train_config_name to empty should fail with error message"""
        local_ip = get_local_ip()
        hosts_conf = self.project_root / "aura" / "configs" / "hosts.conf"
        self.modify_file(hosts_conf, f"{local_ip},0,1,1\n")

        base_conf = self.project_root / "aura" / "configs" / "base.conf"
        self.replace_in_file(base_conf, "work_mode=one_step_off", "work_mode=hybrid")
        self.replace_in_file(
            base_conf,
            "train_config_name=verl_train_async_A3_t16_qwen3_32b_math_fsdp",
            "train_config_name=",
        )
        result = self.run_cli()
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["missing required argument", "--config-name"],
            "Expected error message about missing --config-name",
        )


class Test05NonExistentTrainConfig(SystemTestBase):
    """Test 5: Verify error handling when train_config_name points to a non-existent file"""

    def test_non_existent_train_config(self):
        """Setting train_config_name to a non-existent file should fail with error message"""
        local_ip = get_local_ip()
        hosts_conf = self.project_root / "aura" / "configs" / "hosts.conf"
        self.modify_file(hosts_conf, f"{local_ip},0,1,1\n")

        base_conf = self.project_root / "aura" / "configs" / "base.conf"
        self.replace_in_file(base_conf, "work_mode=one_step_off", "work_mode=hybrid")
        self.replace_in_file(
            base_conf,
            "train_config_name=verl_train_async_A3_t16_qwen3_32b_math_fsdp",
            "train_config_name=non_existent_train_config",
        )
        result = self.run_cli()
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["train config file not found", "non_existent_train_config"],
            "Expected error message about non-existent train config file",
        )


class Test06EmptyInferConfigName(SystemTestBase):
    """Test 6: Verify error handling when infer_config_name is empty in base.conf"""

    def test_empty_infer_config_name(self):
        """Setting infer_config_name to empty should fail with error message"""
        local_ip = get_local_ip()
        hosts_conf = self.project_root / "aura" / "configs" / "hosts.conf"
        self.modify_file(hosts_conf, f"{local_ip},0,1,1\n")

        base_conf = self.project_root / "aura" / "configs" / "base.conf"
        self.replace_in_file(
            base_conf,
            "infer_config_name=vllm_infer_i16_qwen3_32b",
            "infer_config_name=",
        )
        result = self.run_cli()
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["infer_config_name is not set", "infer_config_name"],
            "Expected error message about empty infer_config_name",
        )


class Test07NonExistentInferConfig(SystemTestBase):
    """Test 7: Verify error handling when infer_config_name points to a non-existent file"""

    def test_non_existent_infer_config(self):
        """Setting infer_config_name to a non-existent file should fail with error message"""
        local_ip = get_local_ip()
        hosts_conf = self.project_root / "aura" / "configs" / "hosts.conf"
        self.modify_file(hosts_conf, f"{local_ip},0,1,1\n")

        base_conf = self.project_root / "aura" / "configs" / "base.conf"
        self.replace_in_file(
            base_conf,
            "infer_config_name=vllm_infer_i16_qwen3_32b",
            "infer_config_name=non_existent_infer_config",
        )
        result = self.run_cli()
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["infer config file not found", "non_existent_infer_config"],
            "Expected error message about non-existent infer config file",
        )


class Test08MissingHostsConf(SystemTestBase):
    """Test 8: Verify error handling when hosts.conf is missing"""

    def test_missing_hosts_conf(self):
        """Removing hosts.conf should fail with error message about the config file"""
        hosts_conf = self.project_root / "aura" / "configs" / "hosts.conf"
        self.delete_file(hosts_conf)
        result = self.run_cli()
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["hosts.conf", "is not exist"],
            "Expected error message about missing hosts.conf",
        )


class Test09HostsConfNonLocalIP(SystemTestBase):
    """Test 9: Verify error handling when hosts.conf contains no local IP in hybrid mode"""

    def test_hosts_conf_non_local_ip(self):
        """In hybrid mode, a non-local IP in hosts.conf should fail with error message"""
        base_conf = self.project_root / "aura" / "configs" / "base.conf"
        self.replace_in_file(base_conf, "work_mode=one_step_off", "work_mode=hybrid")

        hosts_conf = self.project_root / "aura" / "configs" / "hosts.conf"
        self.modify_file(hosts_conf, "192.168.99.99,0,1,1\n")

        result = self.run_cli()
        self.assertExitFailure(result)
        self.assertLogContainsAny(
            result,
            ["no host in hosts.conf matches local IP", "hosts.conf"],
            "Expected error message about non-local IP in hosts.conf",
        )


class Test10ValidDirectModeConfig(SystemTestBase):
    """Test 10: Verify valid direct mode config can be loaded normally"""

    def test_valid_direct_mode_config(self):
        """Valid direct mode config should be loaded successfully (verify config parsing success)"""
        local_ip = get_local_ip()
        hosts_conf = self.project_root / "aura" / "configs" / "hosts.conf"
        self.modify_file(hosts_conf, f"{local_ip},0,1,1\n")

        self.copy_to_train_configs("presmoke_valid_direct_mode.yaml")

        base_conf = self.project_root / "aura" / "configs" / "base.conf"
        self.replace_in_file(
            base_conf,
            "train_config_name=verl_train_async_A3_t16_qwen3_32b_math_fsdp",
            "train_config_name=presmoke_valid_direct_mode",
        )

        result = self.run_cli()
        self.assertLogContainsAny(
            result,
            ["Agentic AI Config", "mode", "direct"],
            "Expected config loading success message",
        )


if __name__ == "__main__":
    unittest.main()
