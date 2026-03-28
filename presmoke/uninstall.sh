#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Pre-smoke uninstall start..."

# ------------------------------------------------------------------------------
# 1. Setup paths
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_PATH="${SCRIPT_DIR}/../presmoke_install"
LOG_FILE="${SCRIPT_DIR}/uninstall.log"

echo "[INFO] Script dir     : $SCRIPT_DIR"
echo "[INFO] Install path   : $INSTALL_PATH"
echo "[INFO] Uninstall log  : $LOG_FILE"

# If the installation root directory does not exist, assume already uninstalled and exit successfully
if [ ! -d "$INSTALL_PATH" ]; then
    echo "[WARN] Install path does not exist: $INSTALL_PATH" | tee -a "$LOG_FILE"
    echo "[WARN] Assuming already uninstalled." | tee -a "$LOG_FILE"
    exit 0
fi

# ------------------------------------------------------------------------------
# 2. Run uninstall script
# ------------------------------------------------------------------------------
UNINSTALL_SCRIPT="$INSTALL_PATH/agent/script/uninstall.sh"

if [ ! -f "$UNINSTALL_SCRIPT" ]; then
    echo "[ERROR] Uninstall script not found: $UNINSTALL_SCRIPT"
    exit 1
fi

echo "[INFO] Found uninstall script: $UNINSTALL_SCRIPT" | tee -a "$LOG_FILE"
echo "[INFO] Executing uninstall..." | tee -a "$LOG_FILE"

# Execute uninstall and log output
bash "$UNINSTALL_SCRIPT" 2>&1 | tee "$LOG_FILE"

echo "[INFO] Uninstall command finished" | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# 3. Verify removal of installed files (check specific paths from install script)
# ------------------------------------------------------------------------------
echo "[INFO] Verifying uninstall cleanup..." | tee -a "$LOG_FILE"

# Check if the config file has been removed
if [ -f "$INSTALL_PATH/agent/configs/agentic_parameters.yaml" ]; then
    echo "[ERROR] File still exists: $INSTALL_PATH/agent/configs/agentic_parameters.yaml" | tee -a "$LOG_FILE"
    exit 1
fi

# Check if the agent path (file, directory, or symlink) has been removed
if [ -e "$INSTALL_PATH/agent" ]; then
    echo "[ERROR] Path still exists: $INSTALL_PATH/agent" | tee -a "$LOG_FILE"
    ls -ld "$INSTALL_PATH/agent"
    exit 1
fi

# ------------------------------------------------------------------------------
# 4. Verify Python package uninstalled
# ------------------------------------------------------------------------------
echo "[INFO] Checking Python package 'agentic_rl'..."

if pip3 list 2>/dev/null | grep -q agentic_rl; then
    echo "[ERROR] Python package 'agentic_rl' is still installed." | tee -a "$LOG_FILE"
    pip3 list | grep agentic_rl
    exit 1
else
    echo "[INFO] Python package 'agentic_rl' is removed." | tee -a "$LOG_FILE"
fi

# ------------------------------------------------------------------------------
# 5. Success
# ------------------------------------------------------------------------------
echo "[SUCCESS] Pre-smoke uninstall check PASSED" | tee -a "$LOG_FILE"