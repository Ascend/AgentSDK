#!/usr/bin/env bash
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
# 2. Run smoke tests
# ------------------------------------------------------------------------------
echo "[INFO] Starting smoke tests..." | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT" || { echo "[ERROR] Cannot change to project root $PROJECT_ROOT" | tee -a "$LOG_FILE"; exit 1; }

# Check if pytest is available
if ! command -v pytest &> /dev/null && ! python -m pytest --version &> /dev/null; then
    echo "[ERROR] pytest is not installed or not in PATH" | tee -a "$LOG_FILE"
    exit 1
fi

pip install mlflow -i https://mirrors.aliyun.com/pypi/simple
pip install ray==2.53.0 -i https://mirrors.aliyun.com/pypi/simple --timeout 1000
apt update
apt install net-tools -y
apt install iproute2 -y
export PYTHONPATH="$PROJECT_ROOT/aura":$PYTHONPATH

LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "[INFO] Local IP             : $LOCAL_IP"
export LOCAL_IP

DEFAULT_SOCKET_IFNAME=$(ip -o addr show | awk -v ip="$LOCAL_IP" '$4 ~ "^"ip"/" {print $2}')
echo "[INFO] Default socket ifname: $DEFAULT_SOCKET_IFNAME"
export DEFAULT_SOCKET_IFNAME

# Run all test cases (Test 1-10)
echo "" | tee -a "$LOG_FILE"
echo "[TEST 1-10] Running all pytest test cases" | tee -a "$LOG_FILE"
pytest presmoke/aura/cases/test_endtoend.py -v --tb=short 2>&1 | tee -a "$LOG_FILE"
pytest_exit_code=${PIPESTATUS[0]}

if [ $pytest_exit_code -ne 0 ]; then
    echo "[ERROR] Test cases failed with exit code $pytest_exit_code" | tee -a "$LOG_FILE"
    exit 1
fi
echo "[TEST 1-10] Passed (all tests completed successfully)" | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# 3. Success
# ------------------------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "======================================="
echo "[SUCCESS] All smoke tests PASSED"
echo "======================================="
echo "[SUCCESS] All smoke tests passed. See log for details: $LOG_FILE" | tee -a "$LOG_FILE"
