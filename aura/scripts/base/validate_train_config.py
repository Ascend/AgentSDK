#!/usr/bin/env python3
# coding=utf-8
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
"""CPU-only routing guard for shell launchers; never compose or start a Ray cluster."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omegaconf import OmegaConf

from aura.trainer.config_validation.launch import validate_launch_config
from aura.trainer.config_validation.models import ConfigValidationError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--work-mode", required=True)
    parser.add_argument("--train-engine")
    args = parser.parse_args()
    try:
        validate_launch_config(OmegaConf.load(args.config_file), args.work_mode, args.train_engine or None)
    except ConfigValidationError as exc:
        for issue in exc.report.issues:
            print(f"[FAIL] {issue.path}: {issue.message} value={issue.value!r} expected={issue.expected}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"[FAIL] launch_config={args.config_file} error={exc}", file=sys.stderr)
        return 1
    print(f"[PASS] launch_mode_consistency={args.work_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
