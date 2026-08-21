#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Custom command validator.

Runs a custom command via subprocess to validate stage output.
Exit code 0 means pass; non-zero means failure.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

from . import ValidationResult

logger = logging.getLogger(__name__)


async def validate_custom(
    spec: dict[str, Any],
    workspace_dir: str = "",
) -> ValidationResult:
    """Run a custom command validation.

    spec format:
    {
        "type": "custom",
        "command": "pytest tests/",
        "cwd": ".",
        "timeout": 60,
        "env": {"KEY": "VALUE"},
        "shell": false,
        "pass_message": "All tests passed",
        "fail_message": "Tests failed",
    }

    Args:
        spec: validator spec dict
        workspace_dir: workspace directory

    Returns:
        ValidationResult: validation result
    """
    command = spec.get("command", "")
    if not command:
        return ValidationResult(
            passed=False,
            message="custom: no command specified",
            validator_type="custom",
        )

    cwd = spec.get("cwd", workspace_dir or ".")
    cwd_path = Path(cwd)
    if not cwd_path.is_absolute() and workspace_dir:
        cwd_path = Path(workspace_dir) / cwd

    timeout = int(spec.get("timeout", 60))
    if timeout > 3600:
        logger.warning("custom: timeout %d exceeds max 3600s; clamping to 3600", timeout)
        timeout = 3600
    env = spec.get("env", {})
    use_shell = bool(spec.get("shell", False))
    pass_message = spec.get("pass_message", "Command succeeded")
    fail_message = spec.get("fail_message", "Command failed")

    # Build environment variables
    import os

    process_env = os.environ.copy()
    process_env.update({str(k): str(v) for k, v in env.items()})

    try:
        if use_shell:
            result = await asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,  # nosec B602
                cwd=str(cwd_path),
                capture_output=True,
                timeout=timeout,
                env=process_env,
                text=True,
                check=False,
            )
        else:
            # Split command (shell-style: honors quotes/escapes)
            parts = shlex.split(command)
            result = await asyncio.to_thread(
                subprocess.run,
                parts,
                shell=False,
                cwd=str(cwd_path),
                capture_output=True,
                timeout=timeout,
                env=process_env,
                text=True,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            passed=False,
            message=f"custom: command timed out after {timeout}s",
            validator_type="custom",
            detail={"command": command, "cwd": str(cwd_path)},
        )
    except FileNotFoundError:
        return ValidationResult(
            passed=False,
            message=f"custom: command not found: {command}",
            validator_type="custom",
            detail={"command": command, "cwd": str(cwd_path)},
        )
    except Exception as exc:
        return ValidationResult(
            passed=False,
            message=f"custom: command execution failed: {exc}",
            validator_type="custom",
            detail={"command": command, "error": str(exc)},
        )

    passed = result.returncode == 0

    # Collect output summary
    stdout_tail = result.stdout.strip()[-500:] if result.stdout else ""
    stderr_tail = result.stderr.strip()[-500:] if result.stderr else ""

    return ValidationResult(
        passed=passed,
        message=pass_message if passed else f"{fail_message} (exit code: {result.returncode})",
        validator_type="custom",
        detail={
            "command": command,
            "exit_code": result.returncode,
            "cwd": str(cwd_path),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        },
    )
