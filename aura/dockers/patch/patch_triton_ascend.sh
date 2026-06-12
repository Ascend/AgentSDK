#!/usr/bin/env bash
set -e

echo "[patch_triton_ascend] patching triton-ascend for CANN 9.0.0"

ASCEND_PKG=$(python3 -c "import ascend, os; print(os.path.dirname(ascend.__file__))")
PATCH_FILE=/home/work/AgentSDK/aura/third_party/patch/triton-ascend.patch
if [ -n "$ASCEND_PKG" ] && [ -f "$PATCH_FILE" ]; then
    cd "$ASCEND_PKG"
    if patch -p1 -N < "$PATCH_FILE"; then
        echo "[patch_triton_ascend] done"
    else
        echo "[patch_triton_ascend] warning: patch failed"
    fi
else
    echo "[patch_triton_ascend] warning: skip (ASCEND_PKG=${ASCEND_PKG:-<empty>} or PATCH_FILE missing)"
fi
