#!/bin/bash
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

set -euo pipefail

umask 0027

echo "======================================"
echo "[INFO] Pre-smoke test start"
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
# Utility: retry_command
# ------------------------------------------------------------------------------
# Retry a command up to max_attempts times, sleeping sleep_seconds between failures.
retry_command() {
    local max_attempts="$1" sleep_seconds="$2"
    shift 2
    local attempt
    for attempt in $(seq 1 "$max_attempts"); do
        if "$@" 2>&1 | tee -a "$LOG_FILE"; then
            return 0
        fi
        if [ "$attempt" -lt "$max_attempts" ]; then
            echo "[WARN] Command failed (attempt $attempt/$max_attempts): $* — retrying in ${sleep_seconds}s..." | tee -a "$LOG_FILE"
            sleep "$sleep_seconds"
        fi
    done
    return 1
}

# ------------------------------------------------------------------------------
# 2. Run smoke tests
# ------------------------------------------------------------------------------
echo "[INFO] Starting smoke tests..." | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT" || { echo "[ERROR] Cannot change to project root $PROJECT_ROOT" | tee -a "$LOG_FILE"; exit 1; }

# Check prerequisites
echo "[INFO] Checking python3..." | tee -a "$LOG_FILE"
python3 --version

echo "[INFO] Checking pytest..." | tee -a "$LOG_FILE"
if ! python3 -m pytest --version &> /dev/null; then
    echo "[WARN] pytest not found, attempting to install..." | tee -a "$LOG_FILE"
    if ! retry_command 3 5 pip3 install pytest --break-system-packages; then
        echo "[ERROR] Failed to install pytest after 3 attempts. Please install it manually: pip3 install pytest" | tee -a "$LOG_FILE"
        exit 1
    fi
    if ! python3 -m pytest --version &> /dev/null; then
        echo "[ERROR] Failed to install pytest. Please install it manually: pip3 install pytest" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "[INFO] pytest installed successfully" | tee -a "$LOG_FILE"
fi
python3 -m pytest --version

echo "[INFO] Setting PYTHONPATH..." | tee -a "$LOG_FILE"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/clawcodex-ascend"

# Run all test cases
echo "" | tee -a "$LOG_FILE"
echo "[TEST] Running all pytest test cases" | tee -a "$LOG_FILE"
set +e
python3 -m pytest presmoke/clawcodex/cases/test_scaffold.py -v --tb=short 2>&1 | tee -a "$LOG_FILE"
pytest_exit_code=${PIPESTATUS[0]}
set -e

if [ $pytest_exit_code -ne 0 ]; then
    echo "[ERROR] Test cases failed with exit code $pytest_exit_code" | tee -a "$LOG_FILE"
    exit 1
fi
echo "[TEST] Passed (all tests completed successfully)" | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# 3. Success
# ------------------------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "======================================="
echo "[SUCCESS] All smoke tests PASSED"
echo "======================================="
echo "[SUCCESS] All smoke tests passed. See log for details: $LOG_FILE" | tee -a "$LOG_FILE"
