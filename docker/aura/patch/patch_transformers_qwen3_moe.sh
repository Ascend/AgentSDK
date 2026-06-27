#!/usr/bin/env bash
set -e

echo "[patch_transformers_qwen3_moe] patching transformers for Qwen3-MoE"

TRANSFORMERS_PKG=/home/work/model_env/qwen3_moe/lib/python3.11/site-packages/transformers
TR_PATCH_FILE=/home/work/AgentSDK/aura/third_party/patch/qwen3_moe/transformers.patch
if [ -n "$TRANSFORMERS_PKG" ] && [ -f "$TR_PATCH_FILE" ]; then
    cd "$TRANSFORMERS_PKG"
    if patch -p0 -N < "$TR_PATCH_FILE"; then
        echo "[patch_transformers_qwen3_moe] done"
    else
        echo "[patch_transformers_qwen3_moe] warning: patch failed"
    fi
else
    echo "[patch_transformers_qwen3_moe] warning: skip (TRANSFORMERS_PKG=${TRANSFORMERS_PKG:-<empty>} or patch missing)"
fi
