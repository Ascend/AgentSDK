#!/usr/bin/env bash
set -e

echo "[patch_vllm_ascend] patching vllm-ascend for one-step-off mode"

ASCEND_PKG=$(python3 -c "import vllm_ascend, os; print(os.path.dirname(vllm_ascend.__file__))")
PATCH_FILE=/home/work/AgentSDK/aura/third_party/patch/vllm-ascend.patch
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
