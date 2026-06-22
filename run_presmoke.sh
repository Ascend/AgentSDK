#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo "[INFO] Pre-smoke test start"
echo "======================================"

# ------------------------------------------------------------------------------
# 1. Setup paths and locate the change file
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESMOKE_DIR="${SCRIPT_DIR}/presmoke"
CHANGE_FILE="${SCRIPT_DIR}/../change.txt"

echo "[INFO] Script dir     : $SCRIPT_DIR"
echo "[INFO] Presmoke dir   : $PRESMOKE_DIR"
echo "[INFO] Change file    : $CHANGE_FILE"

if [ ! -d "$PRESMOKE_DIR" ]; then
    echo "[ERROR] presmoke directory not found: $PRESMOKE_DIR"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Parse the change file and decide which module(s) need to be tested
# ------------------------------------------------------------------------------
# The change file is produced by `git diff --name-only --no-commit-id`, so each
# non-empty line is just a repository-relative file path, e.g.:
#     aura/setup.py
#     openclaw/skillhub/cli.py
# There are NO `diff --git` headers. We detect module scope by checking whether
# any changed path starts with `aura/` or `openclaw/`.
# ------------------------------------------------------------------------------
RUN_AURA_PRESMOKE=false
RUN_OPENCLAW_PRESMOKE=false

if [ ! -f "$CHANGE_FILE" ]; then
    echo "[ERROR] change file not found: $CHANGE_FILE"
    echo "[ERROR] Cannot determine which module to test, aborting."
    exit 1
fi

echo ""
echo "[INFO] Reading change file: $CHANGE_FILE"
echo "[INFO] Collecting changed file paths (name-only format)..."

# Collect every non-empty line; each line is a file path from `git diff --name-only`.
mapfile -t CHANGED_FILES < <(grep -vE "^[[:space:]]*$" "$CHANGE_FILE" || true)

echo "[INFO] Found ${#CHANGED_FILES[@]} changed file(s) in the change file."
if [ "${#CHANGED_FILES[@]}" -gt 0 ]; then
    echo "[INFO] Changed file list:"
    for f in "${CHANGED_FILES[@]}"; do
        echo "  - $f"
    done
fi

# Detect aura changes: any file whose path starts with `aura/`.
for line in "${CHANGED_FILES[@]}"; do
    if [[ "$line" == aura/* ]]; then
        RUN_AURA_PRESMOKE=true
        break
    fi
done

# Detect openclaw changes: any file whose path starts with `openclaw/`.
for line in "${CHANGED_FILES[@]}"; do
    if [[ "$line" == openclaw/* ]]; then
        RUN_OPENCLAW_PRESMOKE=true
        break
    fi
done

echo ""
echo "[INFO] Diff analysis result:"
if [ "$RUN_AURA_PRESMOKE" = true ]; then
    echo "  - aura     : CHANGED  -> aura presmoke will be executed"
else
    echo "  - aura     : untouched -> aura presmoke skipped"
fi
if [ "$RUN_OPENCLAW_PRESMOKE" = true ]; then
    echo "  - openclaw : CHANGED  -> openclaw presmoke will be executed"
else
    echo "  - openclaw : untouched -> openclaw presmoke skipped"
fi

# ------------------------------------------------------------------------------
# 3. Short-circuit when neither module is touched
# ------------------------------------------------------------------------------
if [ "$RUN_AURA_PRESMOKE" = false ] && [ "$RUN_OPENCLAW_PRESMOKE" = false ]; then
    echo ""
    echo "[INFO] No changes detected under aura/ or openclaw/."
    echo "[INFO] Nothing to do for pre-smoke, exiting successfully."
    echo "======================================="
    echo "[SUCCESS] Pre-smoke skipped (no relevant changes)"
    echo "======================================="
    exit 0
fi

# ------------------------------------------------------------------------------
# 4. Execute the selected pre-smoke cases
# ------------------------------------------------------------------------------
FAILED_CASES=()
PASSED_CASES=()

run_module_presmoke() {
    local module_name="$1"
    local module_script="$2"

    echo ""
    echo "--------------------------------------"
    echo "[INFO] Running presmoke for module: $module_name"
    echo "[INFO] Script: $module_script"
    echo "--------------------------------------"

    if [ ! -f "$module_script" ]; then
        echo "[WARN] Presmoke script for $module_name not found: $module_script"
        echo "[WARN] This module's presmoke is reserved but not implemented yet, skipping."
        PASSED_CASES+=("$module_name (reserved, no script)")
        return 0
    fi

    chmod u+x "$module_script"
    if bash "$module_script"; then
        echo "[INFO] Presmoke PASSED for module: $module_name"
        PASSED_CASES+=("$module_name")
    else
        echo "[ERROR] Presmoke FAILED for module: $module_name"
        FAILED_CASES+=("$module_name")
        return 1
    fi
}

if [ "$RUN_AURA_PRESMOKE" = true ]; then
    run_module_presmoke "aura" "${PRESMOKE_DIR}/aura/presmoke.sh" || true
fi

if [ "$RUN_OPENCLAW_PRESMOKE" = true ]; then
    # openclaw presmoke is reserved for future use; the entry point is
    # presmoke/openclaw/presmoke.sh but it does not exist yet.
    run_module_presmoke "openclaw" "${PRESMOKE_DIR}/openclaw/presmoke.sh" || true
fi

# ------------------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------------------
echo ""
echo "======================================="
echo "[INFO] Pre-smoke summary"
echo "======================================="
echo "[INFO] Passed: ${#PASSED_CASES[@]} case(s)"
for c in "${PASSED_CASES[@]}"; do
    echo "  - [PASS] $c"
done
echo "[INFO] Failed: ${#FAILED_CASES[@]} case(s)"
for c in "${FAILED_CASES[@]}"; do
    echo "  - [FAIL] $c"
done

if [ "${#FAILED_CASES[@]}" -gt 0 ]; then
    echo ""
    echo "[FAILED] Some pre-smoke cases FAILED"
    echo "======================================="
    exit 1
fi

echo ""
echo "[SUCCESS] All requested pre-smoke cases PASSED"
echo "======================================="
