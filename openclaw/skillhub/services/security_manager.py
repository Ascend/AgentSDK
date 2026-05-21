"""Security manager implementation."""

import json
import os
import subprocess
from typing import List

from skillhub.config import Settings
from skillhub.interfaces.security_manager import SecurityManager
from skillhub.models.security import InstallEvent, SandboxOptions, SandboxResult
from skillhub.utils.checksum import compute_checksum


class SecurityManagerImpl(SecurityManager):
    def __init__(self, config: Settings):
        self.config = config
        self.audit_log_file = config.data_dir / "audit.json"
        self._audit_log: List[dict] = []
        self._load_audit_log()

    def _load_audit_log(self):
        if self.audit_log_file.exists():
            with open(self.audit_log_file, "r", encoding="utf-8") as f:
                self._audit_log = json.load(f)

    def _save_audit_log(self):
        with open(self.audit_log_file, "w", encoding="utf-8") as f:
            json.dump(self._audit_log, f, indent=2, default=str)

    async def verify_signature(
        self,
        content: bytes,
        signature: str,
        public_key: str,
    ) -> bool:
        try:
            import gnupg

            gpg = gnupg.GPG()
            verified = gpg.verify_data(signature, content)
            return verified.valid
        except Exception:
            return False

    async def verify_checksum(
        self,
        content: bytes,
        expected_checksum: str,
        algorithm: str = "sha256",
    ) -> bool:
        actual = compute_checksum(content, algorithm)
        return actual.lower() == expected_checksum.lower()

    async def compute_checksum(
        self,
        content: bytes,
        algorithm: str = "sha256",
    ) -> str:
        return compute_checksum(content, algorithm)

    async def execute_in_sandbox(
        self,
        command: str,
        args: List[str],
        options: SandboxOptions,
    ) -> SandboxResult:
        import time

        start_time = time.time()

        try:
            env = os.environ.copy()

            cmd = [command] + args

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=options.timeout,
                env=env,
                check=False,
            )

            duration = time.time() - start_time

            return SandboxResult(
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                duration=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr="Execution timed out",
                duration=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=duration,
            )

    async def set_secure_permissions(self, path: str) -> None:
        if os.name != "posix":
            return

        for root, dirs, files in os.walk(path):
            for d in dirs:
                dir_path = os.path.join(root, d)
                os.chmod(dir_path, 0o755)
            for f in files:
                file_path = os.path.join(root, f)
                os.chmod(file_path, 0o644)

    async def log_install(self, event: InstallEvent) -> None:
        self._audit_log.append(event.model_dump())
        self._save_audit_log()

    def is_trusted_source(self, source: str) -> bool:
        return source in self.config.security.trusted_sources

    def is_trusted_author(self, author: str) -> bool:
        return author in self.config.security.trusted_authors
