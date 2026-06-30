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

import json
import os
import unittest

from base import SystemTestBase


class TestSingleInstanceConfig(SystemTestBase):
    """Test: deploy.sh config generates correct output for a single instance."""

    BASE_PORT = 18789

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._result = cls.cli_runner.run_config(
            config_dir=cls._temp_dir,
            base_port=cls.BASE_PORT,
            instance_count=1,
        )
        cls._instance_dir = os.path.join(cls._temp_dir, "instance-1")

    def test_01_exit_code_success(self):
        """deploy.sh config should exit with code 0."""
        self.assertExitSuccess(
            self._result,
            f"deploy.sh config failed with exit code {self._result.exit_code}",
        )

    def test_02_instance_directory_exists(self):
        """Instance directory should be created."""
        self.assertFileExists(
            self._instance_dir,
            f"Instance directory not found: {self._instance_dir}",
        )

    def test_03_openclaw_json_exists_and_valid(self):
        """openclaw.json should exist and be valid JSON."""
        path = os.path.join(self._instance_dir, "openclaw.json")
        self.assertJsonValid(path, f"openclaw.json is not valid JSON: {path}")

    def test_04_openclaw_json_gateway_port(self):
        """openclaw.json should have correct gateway.port."""
        path = os.path.join(self._instance_dir, "openclaw.json")
        self.assertFileExists(path, f"openclaw.json not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.assertEqual(
            config["gateway"]["port"],
            self.BASE_PORT,
            f"Expected gateway.port={self.BASE_PORT}, got {config['gateway']['port']}",
        )

    def test_05_models_json_exists_and_valid(self):
        """agents/main/agent/models.json should exist and be valid JSON."""
        path = os.path.join(self._instance_dir, "agents", "main", "agent", "models.json")
        self.assertJsonValid(path, f"models.json is not valid JSON: {path}")

    def test_06_sshd_config_exists(self):
        """ssh/sshd_config should exist."""
        path = os.path.join(self._instance_dir, "ssh", "sshd_config")
        self.assertFileExists(path, f"sshd_config not found: {path}")

    def test_07_ssh_passwd_exists(self):
        """ssh/passwd should exist."""
        path = os.path.join(self._instance_dir, "ssh", "passwd")
        self.assertFileExists(path, f"ssh/passwd not found: {path}")

    def test_08_start_sshd_sh_executable(self):
        """ssh/start_sshd.sh should exist and be executable."""
        path = os.path.join(self._instance_dir, "ssh", "start_sshd.sh")
        self.assertFileExecutable(
            path, f"start_sshd.sh not found or not executable: {path}"
        )

    def test_09_sftp_password_exists(self):
        """ssh/sftp_password should exist and be non-empty."""
        path = os.path.join(self._instance_dir, "ssh", "sftp_password")
        self.assertFileExists(path, f"sftp_password not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        self.assertTrue(
            len(content) > 0,
            f"sftp_password is empty: {path}",
        )

    def test_10_health_monitor_sh_executable(self):
        """health_monitor.sh should exist and be executable."""
        path = os.path.join(self._instance_dir, "health_monitor.sh")
        self.assertFileExecutable(
            path, f"health_monitor.sh not found or not executable: {path}"
        )

    def test_11_health_monitor_sh_content(self):
        """health_monitor.sh should contain health check loop logic."""
        path = os.path.join(self._instance_dir, "health_monitor.sh")
        self.assertFileContains(
            path,
            "HEALTH_CHECK_PORT",
            "health_monitor.sh should reference HEALTH_CHECK_PORT",
        )
        self.assertFileContainsAny(
            path,
            ["failure_count", "wget"],
            "health_monitor.sh should contain health check logic",
        )

    def test_12_claude_settings_json_exists(self):
        """.claude/settings.json should exist."""
        path = os.path.join(self._instance_dir, ".claude", "settings.json")
        self.assertFileExists(path, f".claude/settings.json not found: {path}")

    def test_13_hermes_config_yaml_exists(self):
        """.hermes/config.yaml should exist."""
        path = os.path.join(self._instance_dir, ".hermes", "config.yaml")
        self.assertFileExists(path, f".hermes/config.yaml not found: {path}")

    def test_14_gateway_token_exists(self):
        """.gateway_token should exist and be non-empty."""
        path = os.path.join(self._instance_dir, ".gateway_token")
        self.assertFileExists(path, f".gateway_token not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        self.assertTrue(
            len(content) > 0,
            f".gateway_token is empty: {path}",
        )

    def test_15_docker_compose_yml_exists(self):
        """docker-compose.yml should exist at config base dir."""
        path = os.path.join(self._temp_dir, "docker-compose.yml")
        self.assertFileExists(path, f"docker-compose.yml not found: {path}")

    def test_16_docker_compose_healthcheck(self):
        """docker-compose.yml should contain healthcheck configuration."""
        path = os.path.join(self._temp_dir, "docker-compose.yml")
        self.assertFileContains(
            path, "healthcheck:", "docker-compose.yml missing healthcheck section"
        )
        self.assertFileContains(
            path, "interval: 30s", "docker-compose.yml missing healthcheck interval"
        )
        self.assertFileContains(
            path, "retries: 3", "docker-compose.yml missing healthcheck retries"
        )

    def test_17_docker_compose_health_check_port(self):
        """docker-compose.yml should set HEALTH_CHECK_PORT env var."""
        path = os.path.join(self._temp_dir, "docker-compose.yml")
        expected = f"HEALTH_CHECK_PORT={self.BASE_PORT}"
        self.assertFileContains(
            path, expected, f"docker-compose.yml missing {expected}"
        )

    def test_18_docker_compose_health_monitor_command(self):
        """docker-compose.yml should use health_monitor.sh as command."""
        path = os.path.join(self._temp_dir, "docker-compose.yml")
        self.assertFileContains(
            path,
            "health_monitor.sh",
            "docker-compose.yml missing health_monitor.sh command",
        )


class TestMultiInstanceConfig(SystemTestBase):
    """Test: deploy.sh config generates correct output for multiple instances."""

    BASE_PORT = 18789
    INSTANCE_COUNT = 3

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._result = cls.cli_runner.run_config(
            config_dir=cls._temp_dir,
            base_port=cls.BASE_PORT,
            instance_count=cls.INSTANCE_COUNT,
        )

    def test_01_exit_code_success(self):
        """deploy.sh config should exit with code 0."""
        self.assertExitSuccess(
            self._result,
            f"deploy.sh config failed with exit code {self._result.exit_code}",
        )

    def test_02_instance_directories_exist(self):
        """All instance directories should be created."""
        for i in range(1, self.INSTANCE_COUNT + 1):
            instance_dir = os.path.join(self._temp_dir, f"instance-{i}")
            self.assertFileExists(
                instance_dir,
                f"Instance directory not found: {instance_dir}",
            )

    def test_03_docker_compose_has_all_services(self):
        """docker-compose.yml should contain all instance services."""
        path = os.path.join(self._temp_dir, "docker-compose.yml")
        for i in range(1, self.INSTANCE_COUNT + 1):
            self.assertFileContains(
                path,
                f"openclaw-{i}:",
                f"docker-compose.yml missing service openclaw-{i}",
            )

    def test_04_instance_ports_calculated_correctly(self):
        """Each instance should have correct port: GW_PORT = BASE + (N-1)*4."""
        for i in range(1, self.INSTANCE_COUNT + 1):
            expected_gw = self.BASE_PORT + (i - 1) * 4
            config_path = os.path.join(
                self._temp_dir, f"instance-{i}", "openclaw.json"
            )
            self.assertFileExists(
                config_path,
                f"openclaw.json not found for instance {i}",
            )
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.assertEqual(
                config["gateway"]["port"],
                expected_gw,
                f"Instance {i}: expected gateway.port={expected_gw}, "
                f"got {config['gateway']['port']}",
            )

    def test_05_docker_compose_ports_per_instance(self):
        """docker-compose.yml should expose correct ports for each instance."""
        path = os.path.join(self._temp_dir, "docker-compose.yml")
        self.assertFileExists(path, f"docker-compose.yml not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        for i in range(1, self.INSTANCE_COUNT + 1):
            expected_gw = self.BASE_PORT + (i - 1) * 4
            expected_sftp = expected_gw + 1
            # Verify GW port mapping exists
            gw_port_line = f'"{expected_gw}:{expected_gw}"'
            self.assertIn(
                gw_port_line,
                content,
                f"docker-compose.yml missing GW port mapping {gw_port_line} "
                f"for instance {i}",
            )
            # Verify SFTP port mapping exists
            sftp_port_line = f'"{expected_sftp}:{expected_sftp}"'
            self.assertIn(
                sftp_port_line,
                content,
                f"docker-compose.yml missing SFTP port mapping {sftp_port_line} "
                f"for instance {i}",
            )

    def test_06_health_monitor_per_instance(self):
        """Each instance should have its own health_monitor.sh."""
        for i in range(1, self.INSTANCE_COUNT + 1):
            path = os.path.join(
                self._temp_dir, f"instance-{i}", "health_monitor.sh"
            )
            self.assertFileExecutable(
                path,
                f"health_monitor.sh not found/executable for instance {i}",
            )

    def test_07_healthcheck_per_instance_in_compose(self):
        """Each service in docker-compose.yml should have healthcheck."""
        path = os.path.join(self._temp_dir, "docker-compose.yml")
        self.assertFileExists(path, f"docker-compose.yml not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Count healthcheck sections — should equal instance count
        healthcheck_count = content.count("healthcheck:")
        self.assertEqual(
            healthcheck_count,
            self.INSTANCE_COUNT,
            f"Expected {self.INSTANCE_COUNT} healthcheck sections, "
            f"got {healthcheck_count}",
        )

        # Each instance should have its own HEALTH_CHECK_PORT
        for i in range(1, self.INSTANCE_COUNT + 1):
            expected_gw = self.BASE_PORT + (i - 1) * 4
            expected_env = f"HEALTH_CHECK_PORT={expected_gw}"
            self.assertIn(
                expected_env,
                content,
                f"docker-compose.yml missing {expected_env} for instance {i}",
            )


if __name__ == "__main__":
    unittest.main()
