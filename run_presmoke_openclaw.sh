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

echo "======================================"
echo "[INFO] Openclaw pre-smoke test start"
echo "======================================"

# ------------------------------------------------------------------------------
# 1. Setup paths
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESMOKE_SCRIPT="${SCRIPT_DIR}/presmoke/openclaw/presmoke.sh"

echo "[INFO] Script dir       : $SCRIPT_DIR"
echo "[INFO] Presmoke script  : $PRESMOKE_SCRIPT"

# ------------------------------------------------------------------------------
# 2. Execute openclaw pre-smoke
# ------------------------------------------------------------------------------
if [ ! -f "$PRESMOKE_SCRIPT" ]; then
    echo "[ERROR] openclaw presmoke script not found: $PRESMOKE_SCRIPT"
    exit 1
fi

echo ""
echo "--------------------------------------"
echo "[INFO] Running openclaw presmoke"
echo "--------------------------------------"

chmod u+x "$PRESMOKE_SCRIPT"
bash "$PRESMOKE_SCRIPT"

echo ""
echo "======================================="
echo "[SUCCESS] Openclaw pre-smoke PASSED"
echo "======================================="
