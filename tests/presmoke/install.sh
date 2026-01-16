#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Pre-smoke install start..."

# ------------------------------------------------------------------------------
# 1. Pepare dependcies
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_PATH="${SCRIPT_DIR}/../../presmoke_install"
LOG_FILE="${SCRIPT_DIR}/install.log"

echo "[INFO] Script dir     : $SCRIPT_DIR"
echo "[INFO] Install path   : $INSTALL_PATH"
echo "[INFO] Install log    : $LOG_FILE"

mkdir -p "$INSTALL_PATH"

# ------------------------------------------------------------------------------
# 2. Find release packages
# ------------------------------------------------------------------------------
PKG_SRC_DIR="$(cd "$SCRIPT_DIR/../../../" && pwd)"
PKG_DST_DIR="$SCRIPT_DIR"

echo "[INFO] Searching release package in: $PKG_SRC_DIR"

mapfile -t PKG_LIST < <(
    find "$PKG_SRC_DIR" -maxdepth 1 -type f -name "Ascend-mindxsdk-agentsdk_*.run"
)

if [ "${#PKG_LIST[@]}" -ne 1 ]; then
    echo "[ERROR] Expect exactly one install package in $PKG_SRC_DIR, found: ${#PKG_LIST[@]}"
    if [ "${#PKG_LIST[@]}" -eq 0 ]; then
        echo "  - <none>"
    else
        printf '  - %s\n' "${PKG_LIST[@]}"
    fi
    exit 1
fi

PKG_SRC="${PKG_LIST[0]}"
PKG_NAME="$(basename "$PKG_SRC")"
PKG_DST="$PKG_DST_DIR/$PKG_NAME"

echo "[INFO] Found release package: $PKG_SRC"

if [ ! -f "$PKG_DST" ]; then
    echo "[INFO] Moving package to $PKG_DST"
    cp "$PKG_SRC" "$PKG_DST"
else
    echo "[INFO] Package already exists in script dir, skip move"
fi

chmod u+x "$PKG_DST"

PKG="$PKG_DST"
echo "[INFO] Using install package: $PKG"

# ------------------------------------------------------------------------------
# 3. Install
# ------------------------------------------------------------------------------
echo "[INFO] Start installing Agent SDK..."

"$PKG" --install --install-path="$INSTALL_PATH" \
    2>&1 | tee "$LOG_FILE"

echo "[INFO] Install command finished"

# ------------------------------------------------------------------------------
# 4. Checking install artifacts
# ------------------------------------------------------------------------------
echo "[INFO] Checking install artifacts..."

if [ ! -d "$INSTALL_PATH" ]; then
    echo "[ERROR] Install path not found: $INSTALL_PATH"
    exit 1
fi

if [ ! -d "$INSTALL_PATH/agent" ]; then
    echo "[ERROR] Missing directory: $INSTALL_PATH/agent"
    exit 1
fi

if [ ! -L "$INSTALL_PATH/agent" ]; then
    echo "[ERROR] Must be symlink: $INSTALL_PATH/agent"
    exit 1
fi

if [ ! -f "$INSTALL_PATH/agent/configs/agentic_parameters.yaml" ]; then
    echo "[ERROR] Missing config file: $INSTALL_PATH/agent/configs/agentic_parameters.yaml"
    exit 1
fi

agent_pip=$(pip3 list | grep agentic_rl)
if [ ! -n "$agent_pip" ]; then
    echo "[ERROR] Agent not install for Python"
	exit 1
fi

echo "[INFO] Install artifacts check passed"

# ------------------------------------------------------------------------------
# 5. Log check
# ------------------------------------------------------------------------------
echo "[INFO] Verifying install log..."

if ! grep -q "INFO: Install Agent SDK successfully." "$LOG_FILE"; then
    echo "[ERROR] Install log does not contain success message"
    echo "[ERROR] Last 50 lines of install log:"
    tail -n 50 "$LOG_FILE"
    exit 1
fi

echo "[INFO] Install log success message found"

# ------------------------------------------------------------------------------
# 6. Success
# ------------------------------------------------------------------------------
echo "[SUCCESS] Pre-smoke install check PASSED"