#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo "[INFO] Pre-smoke test start"
echo "======================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESMOKE_DIR="${SCRIPT_DIR}/tests/presmoke"

if [ ! -d "$PRESMOKE_DIR" ]; then
    echo "[ERROR] presmoke directory not found: $PRESMOKE_DIR"
    exit 1
fi

# ------------------------------------------------------------------------------
# collecting cases
# ------------------------------------------------------------------------------
mapfile -t CASES < <(
    find "$PRESMOKE_DIR" -maxdepth 1 -type f -name "*.sh" | sort
)

if [ "${#CASES[@]}" -eq 0 ]; then
    echo "[ERROR] No presmoke cases found in $PRESMOKE_DIR"
    exit 1
fi

echo "[INFO] Found ${#CASES[@]} presmoke case(s):"
for c in "${CASES[@]}"; do
    echo "  - $(basename "$c")"
done

# ------------------------------------------------------------------------------
# performing tests
# ------------------------------------------------------------------------------
for case in "${CASES[@]}"; do
    case_name="$(basename "$case")"

    echo
    echo "--------------------------------------"
    echo "[INFO] Running presmoke case: $case_name"
    echo "--------------------------------------"

    chmod u+x "$case"
    bash "$case"

    echo "[INFO] Case passed: $case_name"
done

echo
echo "======================================"
echo "[SUCCESS] All presmoke cases PASSED"
echo "======================================"