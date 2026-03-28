#!/usr/bin/env bash
set -euo pipefail

umask 0027

echo "[INFO] Pre-smoke test start..."

# ------------------------------------------------------------------------------
# 1. Setup paths and log
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/presmoke_cases.log"
INSTALL_PATH="${SCRIPT_DIR}/../presmoke_install"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[INFO] Script dir    : $SCRIPT_DIR"
echo "[INFO] Install path  : $INSTALL_PATH"
echo "[INFO] Log file      : $LOG_FILE"
echo "[INFO] Project root  : $PROJECT_ROOT"

# Clear previous log
> "$LOG_FILE"

# ------------------------------------------------------------------------------
# 2. Run smoke test commands
# ------------------------------------------------------------------------------
echo "[INFO] Starting smoke tests..." | tee -a "$LOG_FILE"

# Test 1: Import module
echo "[TEST 1] Import agentic_rl module" | tee -a "$LOG_FILE"
python -c "import agentic_rl" 2>&1 | tee -a "$LOG_FILE"
echo "[TEST 1] Passed" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Test 2: CLI help (check that usage line is present, ignore exit code)
echo "[TEST 2] Run agentic_rl and verify output (expected to fail)" | tee -a "$LOG_FILE"
set +e
output=$(agentic_rl </dev/null 2>&1)
exit_code=$?
set -e
echo "$output" | tee -a "$LOG_FILE"
echo "[TEST 2] Exit code: $exit_code" | tee -a "$LOG_FILE"
# Only verify output contains expected usage string
if [[ "$output" != *"usage: agentic_rl --config-path CONFIG_PATH"* ]]; then
    echo "[ERROR] Test 2 failed: expected help text to contain 'usage: agentic_rl --config-path CONFIG_PATH'" | tee -a "$LOG_FILE"
    exit 1
fi
echo "[TEST 2] Passed (got expected failure and correct error message)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Test 3: CLI with invalid config path (expected failure)
echo "[TEST 3] Run agentic_rl --config-path=$INSTALL_PATH/agent/configs/agentic_parameters.yaml (expected to fail)" | tee -a "$LOG_FILE"
set +e
output=$(agentic_rl --config-path=$INSTALL_PATH/agent/configs/agentic_parameters.yaml 2>&1)
exit_code=$?
set -e
echo "$output" | tee -a "$LOG_FILE"
echo "[TEST 3] Exit code: $exit_code" | tee -a "$LOG_FILE"
if [ $exit_code -eq 0 ]; then
    echo "[ERROR] Command succeeded but expected to fail" | tee -a "$LOG_FILE"
    exit 1
fi
if [[ "$output" != *"Checking config_path failed with value not correct, error: There are '..' characters in path"* ]]; then
    echo "[ERROR] Unexpected error message. Expected to contain: 'Checking config_path failed with value not correct, error: There are '..' characters in path'" | tee -a "$LOG_FILE"
    exit 1
fi
echo "[TEST 3] Passed (got expected failure and correct error message)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Test 4: Run pytest test cases
echo "[TEST 4-8] Running pytest test cases" | tee -a "$LOG_FILE"
cd "$PROJECT_ROOT" || { echo "[ERROR] Cannot change to project root $PROJECT_ROOT" | tee -a "$LOG_FILE"; exit 1; }

# check if pytest is available and installed
if ! command -v pytest &> /dev/null && ! python -m pytest --version &> /dev/null; then
    echo "[ERROR] pytest is not installed or not in PATH" | tee -a "$LOG_FILE"
    exit 1
fi

# run end to end test cases
pytest presmoke/cases/test_endtoend.py -v --tb=short 2>&1 | tee -a "$LOG_FILE"
pytest_exit_code=${PIPESTATUS[0]}

if [ $pytest_exit_code -ne 0 ]; then
    echo "[ERROR] End to end test cases failed with exit code $pytest_exit_code" | tee -a "$LOG_FILE"
    exit 1
fi
echo "[TEST 4-8] Passed (pytest completed successfully)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# 3. Success
# ------------------------------------------------------------------------------
echo "[SUCCESS] All smoke tests passed. See log for details: $LOG_FILE" | tee -a "$LOG_FILE"