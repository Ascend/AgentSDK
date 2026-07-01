#!/bin/bash
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

set -euo pipefail

umask 0027

echo "======================================"
echo "[INFO] OpenClaw Pre-smoke test start"
echo "======================================"

# ------------------------------------------------------------------------------
# 1. Setup paths and log
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/presmoke_cases.log"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[INFO] Script dir    : $SCRIPT_DIR"
echo "[INFO] Log file      : $LOG_FILE"
echo "[INFO] Project root  : $PROJECT_ROOT"

# Clear previous log
> "$LOG_FILE"

# ------------------------------------------------------------------------------
# 2. Check prerequisites
# ------------------------------------------------------------------------------
echo "[INFO] Checking prerequisites..." | tee -a "$LOG_FILE"

if ! command -v envsubst &> /dev/null; then
    echo "[ERROR] envsubst is not installed. Please install gettext package." | tee -a "$LOG_FILE"
    echo "[INFO]   Ubuntu/Debian: apt-get install gettext-base" | tee -a "$LOG_FILE"
    echo "[INFO]   CentOS/RHEL:   yum install gettext" | tee -a "$LOG_FILE"
    exit 1
fi

if ! command -v openssl &> /dev/null; then
    echo "[ERROR] openssl is not installed." | tee -a "$LOG_FILE"
    exit 1
fi
pip3 install pytest --break-system-packages
if ! python3 -m pytest --version &> /dev/null; then
    echo "[ERROR] pytest is not installed for python3" | tee -a "$LOG_FILE"
    exit 1
fi

echo "[INFO] Prerequisites check passed" | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# 3. Run smoke tests
# ------------------------------------------------------------------------------
cd "$PROJECT_ROOT" || { echo "[ERROR] Cannot change to project root $PROJECT_ROOT" | tee -a "$LOG_FILE"; exit 1; }

export PYTHONPATH="$PROJECT_ROOT/openclaw":"$PROJECT_ROOT/presmoke/openclaw/cases":${PYTHONPATH:-}

echo "" | tee -a "$LOG_FILE"
echo "[TEST] Running OpenClaw deploy config tests" | tee -a "$LOG_FILE"
python3 -m pytest presmoke/openclaw/cases/test_deploy.py -v --tb=short 2>&1 | tee -a "$LOG_FILE" || true
pytest_exit_code=${PIPESTATUS[0]}

if [ $pytest_exit_code -ne 0 ]; then
    echo "[ERROR] Test cases failed with exit code $pytest_exit_code" | tee -a "$LOG_FILE"
    exit 1
fi
echo "[TEST] Passed (all tests completed successfully)" | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# 4. Success
# ------------------------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "======================================="
echo "[SUCCESS] OpenClaw smoke tests PASSED"
echo "======================================="
echo "[SUCCESS] All smoke tests passed. See log for details: $LOG_FILE" | tee -a "$LOG_FILE"
