#!/usr/bin/env bash

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
artifact_dir=""
repo_root=""
args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      artifact_dir="$2"
      args+=("$1" "$2")
      shift 2
      ;;
    --repo-root)
      repo_root="$2"
      args+=("$1" "$2")
      shift 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

python_bin="${PTY_ADAPTIVE_PYTHON:-}"
if [[ -z "$python_bin" && -n "$repo_root" && -x "$repo_root/.venv/bin/python3" ]]; then
  python_bin="$repo_root/.venv/bin/python3"
elif [[ -z "$python_bin" && -n "$repo_root" && -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
elif [[ -z "$python_bin" ]]; then
  python_bin="python3"
fi

"$python_bin" "$script_dir/pty_adaptive_driver.py" "${args[@]}"
exit_code=$?

echo "PTY_ADAPTIVE_DRIVER_EXIT=$exit_code"
if [[ -n "$artifact_dir" ]]; then
  echo "PTY_ADAPTIVE_DRIVER_ARTIFACT_DIR=$artifact_dir"
  if [[ -f "$artifact_dir/driver-error.json" ]]; then
    echo "PTY_ADAPTIVE_DRIVER_ERROR_JSON_BEGIN"
    cat "$artifact_dir/driver-error.json"
    echo "PTY_ADAPTIVE_DRIVER_ERROR_JSON_END"
  fi
  if [[ -f "$artifact_dir/adaptive-driver.jsonl" ]]; then
    echo "PTY_ADAPTIVE_DRIVER_JSONL_FIRST_BEGIN"
    sed -n '1p' "$artifact_dir/adaptive-driver.jsonl"
    echo "PTY_ADAPTIVE_DRIVER_JSONL_FIRST_END"
  fi
  if [[ -f "$artifact_dir/result.json" ]]; then
    echo "PTY_ADAPTIVE_DRIVER_RESULT_JSON=$artifact_dir/result.json"
  fi
fi

exit "$exit_code"
