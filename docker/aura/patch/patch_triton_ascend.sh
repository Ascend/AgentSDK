#!/usr/bin/env bash
set -e

echo "[patch_triton_ascend] patching triton-ascend for CANN 9.0.0"

ASCEND_PKG=$(python3 -c "import triton, os; print(os.path.dirname(triton.__file__))")

PATCH_DIR=/home/work/AgentSDK/aura/third_party/patch

DRIVER_PATCH="$PATCH_DIR/triton-ascend_driver.patch"
NPU_PATCH="$PATCH_DIR/triton-ascend_npu_utils.patch"

if [ -z "$ASCEND_PKG" ]; then
    echo "[patch_triton_ascend] error: ASCEND_PKG empty"
    exit 1
fi

if [ ! -d "$ASCEND_PKG/backends/ascend" ]; then
    echo "[patch_triton_ascend] error: backends/ascend not found in $ASCEND_PKG"
    exit 1
fi

cd "$ASCEND_PKG/backends/ascend"

echo "[patch_triton_ascend] cwd=$(pwd)"

if [ -f "$DRIVER_PATCH" ]; then
    echo "[patch] applying driver patch"
    if patch -p0 -N < "$DRIVER_PATCH"; then
        echo "[patch] driver patch done"
    else
        echo "[patch] driver patch failed (or already applied)"
    fi
else
    echo "[patch] missing $DRIVER_PATCH"
fi

if [ -f "$NPU_PATCH" ]; then
    echo "[patch] applying npu_utils patch"
    if patch -p0 -N < "$NPU_PATCH"; then
        echo "[patch] npu_utils patch done"
    else
        echo "[patch] npu_utils patch failed"
    fi
else
    echo "[patch] missing $NPU_PATCH"
fi

echo "[patch_triton_ascend] done"
