#!/usr/bin/env bash
# ============================================================================
#  install.sh — One-click installer for clawcodex (agent-friendly edition)
# ----------------------------------------------------------------------------
#  - OS detection (Linux / macOS / WSL / Git Bash)
#  - Git prerequisite check
#  - uv installation (no sudo, via official astral.sh installer)
#  - Python 3.11+ provisioning (via uv)
#  - Repo clone/update to ~/.clawcodex/clawcodex
#  - Local release .env bootstrap from .env.example (never overwrites)
#  - Venv creation (uv-managed)
#  - Dependency install: pip install -e ".[all]"
#  - Local pre-commit hook install (best-effort; never blocks CLI install)
#  - Global commands: ~/.local/bin/clawcodex  +  ~/.local/bin/clawcodex-dev
#  - Shell rc patch: .bashrc / .zshrc / .profile  (PATH += ~/.local/bin)
#
#  Subcommands (use exactly one, or omit for default 'install'):
#     ./install.sh                # install (default)
#     ./install.sh install        # explicit install
#     ./install.sh status         # show current install state
#     ./install.sh doctor         # diagnose the environment
#     ./install.sh verify         # health-check an existing install
#     ./install.sh update         # pull latest + reinstall deps
#     ./install.sh uninstall      # remove everything this script created
#     ./install.sh help           # show usage
#     ./install.sh --help         # English help
#     ./install.sh --help-zh      # 中文使用说明
#     ./install.sh --version      # print installer version
#
#  Agent-friendly features:
#     - Subcommands (status / doctor / verify) for inspection without side effects
#     - --dry-run             preview every change before applying
#     - --yes / -y            assume yes for any prompts
#     - --log-file <path>     tee all output to a log file
#     - [install.sh] prefix on every line when stdout is not a TTY
#     - "DONE: success|FAILED" summary line on exit (grep-friendly)
#     - Each die() includes a "Next steps" block with actionable fixes
# ----------------------------------------------------------------------------
set -euo pipefail
# ERR trap: if a command fails under set -e, print the line number and
# failing command before exit.  Makes headless / TTY / CI failures
# self-diagnosing without requiring "bash -x".
set -E
trap 'log_err "Installer crash at line $LINENO: $BASH_COMMAND"' ERR

# ============================================================================
#  Config (read-only defaults)
# ============================================================================
# Versioning scheme
# -----------------
#   INSTALLER_VERSION  — version of this install.sh script
#   CLAWCODEX_VERSION  — version of clawcodex that THIS install.sh installs
#   REPO_REF           — git ref (tag/branch) the install clones
#
# The install.sh script is released alongside the clawcodex tag it ships
# with, so INSTALLER_VERSION == CLAWCODEX_VERSION for the bundle. Small
# script changes (typo fixes, log tweaks, new flags) do NOT bump the
# installer version — the version moves only when a new clawcodex tag
# is cut. To install a different clawcodex version, fetch the install.sh
# from that version's tag; an old install.sh still installs the old
# clawcodex (with its old uv.lock), never the bleeding edge.
# If REPO_REF doesn't resolve on the remote, the install falls back to the
# default branch with a loud warning — useful during the pre-tag period of
# a release but should never ship in a tagged installer.
readonly INSTALLER_VERSION="2026.6.24"
readonly CLAWCODEX_VERSION="2026.6.24"
# REPO_REF is intentionally NOT readonly — it gets reassigned when the user
# passes --ref. Same for CLAWCODEX_HOME / CLAWCODEX_PARENT_DIR / CONFIG_DIR
# (derived from overridable defaults below).
REPO_REF="v${CLAWCODEX_VERSION}"
readonly REPO_URL="https://gitcode.com/Ascend/AgentSDK"
# --- Overridable paths (defaults; overridden by --install-dir / --config-dir) ---
# Install dir = where the project source is cloned and (by default) the .venv lives.
# Config dir  = where clawcodex-dev stores its runtime state (sessions, auth, history).
#               Exposed to the runtime via $CLAWCODEX_CONFIG_DIR; the wrapper scripts
#               below set that env var on every invocation.
readonly DEFAULT_INSTALL_DIR="$HOME/.clawcodex/clawcodex"
readonly DEFAULT_CONFIG_DIR="$HOME/.clawcodex"
readonly LOCAL_BIN="$HOME/.local/bin"
readonly PYTHON_MIN_VERSION="3.11"
readonly PYTHON_MAX_SUPPORTED="3.13"
readonly ENTRY_POINT="clawcodex-dev"   # the single registered entry in pyproject.toml
readonly RC_MARKER="# clawcodex installer — managed by install.sh"
# --- Upstream source for src/ directory (Claude Code upstream fork) ---
# When src/ is not present in the repo, the installer pulls it from the
# upstream source at the pinned commit and applies the corresponding patches.
readonly UPSTREAM_URL="https://github.com/agentforce314/clawcodex.git"
# UPSTREAM_REF is intentionally NOT readonly — update this on each version sync
# to match the patches/upstream/<commit>/ directory.
UPSTREAM_REF="398b44f"
# ============================================================================
#  UI helpers
# ============================================================================
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
    C_RED=$'\033[0;31m'; C_GREEN=$'\033[0;32m'; C_YELLOW=$'\033[1;33m'
    C_BLUE=$'\033[0;34m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
    C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_BOLD=''; C_RESET=''
fi

# Agent-friendly line prefix. Emitted only when stdout/stderr is not a TTY
# (i.e. when the script is being driven by another process, an agent, a CI
# runner, or a piped tee). Interactive users see clean output.
_script_p1() { [[ ! -t 1 ]] && printf '[install.sh] '; return 0; }
_script_p2() { [[ ! -t 2 ]] && printf '[install.sh] ' >&2; return 0; }

log_info() { _script_p1; echo -e "${C_BLUE}==>${C_RESET} ${C_BOLD}$1${C_RESET}"; }
log_ok()   { _script_p1; echo -e "  ${C_GREEN}✓${C_RESET} $1"; }
log_warn() { _script_p1; echo -e "  ${C_YELLOW}!${C_RESET} $1"; }
log_err()  { _script_p2; echo -e "${C_RED}✗${C_RESET} $1" >&2; }
log_step() { _script_p1; echo -e "\n${C_BOLD}${C_BLUE}>>>${C_RESET} ${C_BOLD}$1${C_RESET}"; }
die() { log_err "$1"; exit 1; }
# Like die(), but accepts 0+ "next steps" lines that are printed in a clear
# "what to do next" block. Designed for agent-driven installs where the
# failure handler needs to know what to retry.
#
# Usage: die_with_help "primary error message" \
#                      "try this first" \
#                      "or this second"
die_with_help() {
    local header="$1"; shift
    _script_p2
    echo -e "${C_RED}✗${C_RESET} $header" >&2
    if [[ $# -gt 0 ]]; then
        echo "" >&2
        echo "  Next steps to try:" >&2
        for step in "$@"; do
            echo "    → $step" >&2
        done
    fi
    echo "" >&2
    echo "  For diagnosis, run:    $0 doctor" >&2
    echo "  For full usage, run:    $0 --help" >&2
    exit 1
}

# Wrap a command: if DRY_RUN=1, just print what would happen. Otherwise run it.
# Returns the exit status of the wrapped command.
run_or_dry() {
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        _script_p1
        echo "[DRY-RUN] would run: $*"
        return 0
    fi
    "$@"
}

# Exit-time summary. Emitted by the EXIT trap after the script's main work
# is done (success or failure). Agents tail the log for this line to know
# whether the install succeeded.
_on_exit_summary() {
    local rc=$1
    local elapsed=$(( $(date +%s) - SCRIPT_START_TS ))
    if [[ $rc -eq 0 ]]; then
        _script_p1
        echo "DONE: success in ${elapsed}s"
        if [[ -n "${LOG_FILE:-}" ]]; then
            _script_p1
            echo "DONE: full log saved to: $LOG_FILE"
        fi
    else
        _script_p2
        echo "DONE: FAILED (exit $rc) after ${elapsed}s" >&2
        if [[ -n "${LOG_FILE:-}" ]]; then
            _script_p2
            echo "DONE: failure log saved to: $LOG_FILE" >&2
        else
            _script_p2
            echo "DONE: re-run with --log-file <path> to capture full output." >&2
        fi
    fi
}

# ============================================================================
#  OS detection
# ============================================================================
detect_os() {
    local ostype="${OSTYPE:-}"
    if [[ "$ostype" == "linux-gnu"* || "$ostype" == "linux-musl"* ]]; then
        # Distinguish WSL from native Linux
        if [[ -r /proc/version ]] && grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
            echo "wsl"
        else
            echo "linux"
        fi
    elif [[ "$ostype" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$ostype" == "msys"* || "$ostype" == "cygwin"* || "$ostype" == "win32" ]]; then
        echo "windows-like"
    elif [[ -r /proc/version ]] && grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
        echo "wsl"
    else
        echo "unknown"
    fi
}

os_install_hint() {
    case "$1" in
        linux|wsl)
            cat <<'EOF'
    Install Git for your distro, e.g.:
        Debian/Ubuntu : sudo apt update && sudo apt install -y git
        Fedora/RHEL   : sudo dnf install -y git
        Arch          : sudo pacman -S --noconfirm git
        openSUSE      : sudo zypper install -y git
EOF
            ;;
        macos)
            cat <<'EOF'
    Install Git on macOS:
        xcode-select --install          # Apple Command Line Tools
        — or —
        brew install git
EOF
            ;;
        windows-like)
            cat <<'EOF'
    On Windows, install one of:
        Git for Windows : https://git-scm.com/download/win  (then run from Git Bash)
        WSL             : https://learn.microsoft.com/windows/wsl/install  (recommended)
EOF
            ;;
    esac
}

# One-liner variant for the doctor output.
os_install_hint_oneliner() {
    case "$1" in
        linux|wsl) echo "sudo apt install -y git   (or your distro's package manager)" ;;
        macos)     echo "xcode-select --install    (or: brew install git)" ;;
        windows-like) echo "install Git for Windows or WSL" ;;
        *)         echo "install git via your package manager" ;;
    esac
}

# ============================================================================
#  Prerequisite: Git
# ============================================================================
check_git() {
    if ! command -v git >/dev/null 2>&1; then
        log_err "Git is not installed."
        os_install_hint "$OS"
        exit 1
    fi
    local version
    version=$(git --version)
    log_ok "$version"
}

# ============================================================================
#  Install / locate uv (Astral's Python package manager, via pip)
# ============================================================================
install_uv() {
    if command -v uv >/dev/null 2>&1; then
        log_ok "uv $(uv --version | awk '{print $2}') already installed"
        return
    fi

    log_info "Installing uv via pip (trusted PyPI source)..."
    local py
    py=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "python3")
    if ! run_or_dry "$py" -m pip install uv 2>&1; then
        die_with_help "Failed to install uv via pip." \
                      "Check your network connection and proxy settings." \
                      "Retry:    $0" \
                      "Manual:   pip install uv"
    fi

    # Make uv visible to this session, then verify.
    export PATH="$HOME/.local:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if [[ "${DRY_RUN:-0}" -eq 0 ]] && ! command -v uv >/dev/null 2>&1; then
        die_with_help "uv still not on PATH after install." \
                      "Check:    ls -la $HOME/.local/bin/uv" \
                      "Or:       export PATH=\$HOME/.local:\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH" \
                      "Then:     $0"
    fi
    log_ok "uv $(uv --version | awk '{print $2}') installed"
}

# ============================================================================
#  Python 3.10+ provisioning (via uv)
# ============================================================================
ensure_python() {
    # Ask uv for any 3.10+ interpreter it can see (system or uv-managed).
    local py
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        _script_p1
        echo "[DRY-RUN] would check for Python $PYTHON_MIN_VERSION+ via uv"
        return 0
    fi
    if py=$(uv python find "$PYTHON_MIN_VERSION" 2>/dev/null) && [[ -n "$py" && -x "$py" ]]; then
        log_ok "Python $($py --version 2>&1 | awk '{print $1, $2}')"
        return
    fi

    log_info "Python $PYTHON_MIN_VERSION+ not found — provisioning via uv (no sudo)..."
    if ! run_or_dry uv python install "$PYTHON_MIN_VERSION"; then
        die_with_help "Failed to install Python $PYTHON_MIN_VERSION via uv." \
                      "Retry:    $0" \
                      "Manual:   uv python install $PYTHON_MIN_VERSION" \
                      "Or:       install Python $PYTHON_MIN_VERSION+ from https://python.org"
    fi
    py=$(uv python find "$PYTHON_MIN_VERSION" 2>/dev/null || true)
    if [[ -z "$py" || ! -x "$py" ]]; then
        die_with_help "Python $PYTHON_MIN_VERSION still not found after uv install." \
                      "Retry:    $0" \
                      "Diagnose: $0 doctor"
    fi
    log_ok "Python $($py --version 2>&1 | awk '{print $1, $2}')"
}

# ============================================================================
#  Python 3.14+ pyo3-ffi compatibility guard
#  Detects Python >= 3.14 (too new for pyo3-ffi used by outlines-core) and
#  auto-mitigates by switching to a compatible version (3.11 - 3.13).
#  If no compatible version is found, fails with actionable instructions.
# ============================================================================
ensure_python_pyo3_compat() {
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        _script_p1
        echo "[DRY-RUN] would check Python <= $PYTHON_MAX_SUPPORTED for pyo3-ffi compatibility"
        return 0
    fi

    local py
    py=$(uv python find "$PYTHON_MIN_VERSION" 2>/dev/null || true)
    if [[ -z "$py" || ! -x "$py" ]]; then return 0; fi

    local ver major minor
    ver=$($py -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)

    # No mitigation needed for Python <= 3.13
    if [[ "$major" -lt 3 ]] || { [[ "$major" -eq 3 && "$minor" -le 13 ]]; }; then
        return 0
    fi

    log_warn "Python $ver detected, but pyo3-ffi (used by outlines-core) only supports up to $PYTHON_MAX_SUPPORTED"
    log_info "Searching for a compatible Python ($PYTHON_MIN_VERSION - $PYTHON_MAX_SUPPORTED)..."

    local compat_py compat_ver
    for target in 3.13 3.12 3.11; do
        compat_py=$(uv python find "$target" 2>/dev/null || true)
        if [[ -n "$compat_py" && -x "$compat_py" ]]; then
            compat_ver=$($compat_py -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            log_ok "Found compatible Python: $compat_py ($compat_ver)"
            export UV_PYTHON="$compat_py"
            return 0
        fi
    done

    # No compatible interpreter locally — try uv provisioning
    log_info "No compatible Python found locally — provisioning Python $PYTHON_MAX_SUPPORTED via uv..."
    if run_or_dry uv python install "$PYTHON_MAX_SUPPORTED"; then
        compat_py=$(uv python find "$PYTHON_MAX_SUPPORTED" 2>/dev/null || true)
        if [[ -n "$compat_py" && -x "$compat_py" ]]; then
            compat_ver=$($compat_py -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            log_ok "Python $PYTHON_MAX_SUPPORTED installed via uv ($compat_py)"
            export UV_PYTHON="$compat_py"
            return 0
        fi
    fi

    # Auto-mitigation failed — defensive error with clear next steps
    die_with_help "Python $ver is incompatible with clawcodex dependencies" \
        "The detected Python ($ver) is too new for pyo3-ffi (max $PYTHON_MAX_SUPPORTED)." \
        "outlines-core will fail to compile with Rust errors." \
        "" \
        "Recommended fixes (pick one):" \
        "  1. uv python install 3.13   (then re-run install.sh)" \
        "  2. pyenv install 3.13 && pyenv local 3.13" \
        "  3. Install Python 3.11-3.13 from https://python.org" \
        "  4. Set PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 and retry (risky)" \
        "The installer attempted to auto-provision a compatible Python but failed."
}

# ============================================================================
#  Clone or update the repo
# ============================================================================
clone_or_update_repo() {
    # Try to detect if we're already inside an AgentSDK git repo
    local repo_root
    repo_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
    if [[ -n "$repo_root" && -f "$repo_root/README.md" ]]; then
        # We are inside a git repo — use it directly
        CLAWCODEX_HOME="$repo_root"
        log_info "Using existing repo at $CLAWCODEX_HOME"
        if [[ -d "$CLAWCODEX_HOME/.git" ]]; then
            log_info "Pulling latest changes..."
            if (cd "$CLAWCODEX_HOME" && git pull --ff-only) >/dev/null 2>&1; then
                log_ok "Updated via fast-forward"
            else
                die_with_help "git pull failed — the local repository has diverged or has uncommitted changes." \
                              "To resolve:  cd $CLAWCODEX_HOME && git status" \
                              "             git stash && git pull --ff-only  (save local changes first)" \
                              "             git pull --rebase               (rebase your changes)" \
                              "Then re-run: $0 update"
            fi
        fi
        return
    fi

    if [[ -d "$CLAWCODEX_HOME/.git" ]]; then
        log_info "Existing repo found at $CLAWCODEX_HOME — pulling latest changes..."
        if (cd "$CLAWCODEX_HOME" && git pull --ff-only) >/dev/null 2>&1; then
            log_ok "Updated via fast-forward"
        else
            die_with_help "git pull failed — the local repository has diverged or has uncommitted changes." \
                          "To resolve:  cd $CLAWCODEX_HOME && git status" \
                          "             git stash && git pull --ff-only  (save local changes first)" \
                          "             git pull --rebase               (rebase your changes)" \
                          "Then re-run: $0 update"
        fi
        return
    fi

    if [[ -e "$CLAWCODEX_HOME" ]]; then
        # Exists but isn't a git repo — back it up so we don't clobber user work.
        local stamp
        stamp=$(date +%Y%m%d%H%M%S)
        log_warn "$CLAWCODEX_HOME exists but is not a git checkout. Backing up to ${CLAWCODEX_HOME}.bak.${stamp}"
        mv "$CLAWCODEX_HOME" "${CLAWCODEX_HOME}.bak.${stamp}"
    fi

    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        _script_p1
        echo "[DRY-RUN] would create parent dir: $CLAWCODEX_PARENT_DIR"
        _script_p1
        echo "[DRY-RUN] would clone: $REPO_URL (ref: $REPO_REF) -> $CLAWCODEX_HOME"
        return 0
    fi

    mkdir -p "$CLAWCODEX_PARENT_DIR"
    log_info "Cloning $REPO_URL (ref: $REPO_REF) → $CLAWCODEX_HOME"
    # Try the pinned ref first. This is what makes the install version-stable:
    # the matching uv.lock at REPO_REF pins every transitive dep to a known-good
    # version, so old install.sh + old clawcodex + old deps always line up.
    if git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$CLAWCODEX_HOME" 2>/dev/null; then
        log_ok "Cloned ref $REPO_REF (clawcodex $CLAWCODEX_VERSION)"
        return
    fi

    # The ref doesn't exist on the remote yet (e.g. tag not pushed). Loud
    # warning, then fall back to the default branch so install can still
    # succeed in dev / pre-release scenarios.
    log_warn "Ref '$REPO_REF' not found on $REPO_URL — falling back to default branch."
    log_warn "  This install will pull the LATEST clawcodex, not v$CLAWCODEX_VERSION."
    log_warn "  Push a '$REPO_REF' git tag (or update REPO_REF) to enforce the version."
    if ! git clone --depth 1 "$REPO_URL" "$CLAWCODEX_HOME"; then
        die_with_help "git clone failed." \
                      "Check your network connection." \
                      "Verify:  curl -I $REPO_URL" \
                      "Retry:   $0" \
                      "Diagnose: $0 doctor"
    fi
    log_ok "Cloned default branch (clawcodex version NOT pinned)"
}

# ============================================================================
#  Initialize local release environment
# ============================================================================
ensure_local_env_file() {
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        _script_p1
        echo "[DRY-RUN] would create $CLAWCODEX_HOME/.env from .env.example if missing"
        return 0
    fi

    cd "$CLAWCODEX_HOME"
    if [[ -f ".env" ]]; then
        log_ok "Local .env already exists (not modified)"
        return 0
    fi

    if [[ -f ".env.example" ]]; then
        cp ".env.example" ".env"
    else
        cat > ".env" <<'EOF'
# Local F-73 release credentials. Never commit real token values.
GITCODE_TOKEN=
TEST_PYPI_TOKEN=
# PYPI_TOKEN=
GITCODE_OWNER=
GITCODE_REPO=
GITCODE_API_ROOT=https://api.gitcode.com
EOF
    fi
    chmod 600 ".env" 2>/dev/null || true
    log_ok "Created local .env template (fill tokens before release publishing)"
}

# ============================================================================
#  Create venv
# ============================================================================
create_venv() {
    if [[ "$USE_VENV" -eq 0 ]]; then
        log_info "--no-venv specified — skipping venv creation (deps will install to system Python)"
        return
    fi
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        _script_p1
        echo "[DRY-RUN] would run: uv venv --python $PYTHON_MIN_VERSION .venv   (in $CLAWCODEX_HOME)"
        return 0
    fi
    cd "$CLAWCODEX_HOME"
    if [[ -d ".venv" ]]; then
        log_ok "Existing venv at $CLAWCODEX_HOME/.venv"
        return
    fi
    log_info "Creating venv with Python $PYTHON_MIN_VERSION..."
    if ! run_or_dry uv venv --python "$PYTHON_MIN_VERSION" .venv; then
        die_with_help "uv venv failed." \
                      "Check:    uv --version" \
                      "Retry:    $0" \
                      "Diagnose: $0 doctor"
    fi
    log_ok "Venv created"
}

# ============================================================================
#  Install dependencies
# ============================================================================
install_deps() {
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        log_info "Installing project + [all] extra (lock-pinned when possible)..."
        _script_p1
        echo "[DRY-RUN] would run: uv sync --extra all   (in $CLAWCODEX_HOME)"
        return 0
    fi
    cd "$CLAWCODEX_HOME"
    log_info "Installing project + [all] extra (lock-pinned to uv.lock when possible)..."

    # Two-stage install: prefer `uv sync` (honors uv.lock → exact transitive
    # versions), fall back to `uv pip install` if the project doesn't declare
    # the [all] extra or the lock is out of sync.
    #
    # Why a fallback?
    #   - Production (matched install.sh + clawcodex release): uv sync uses
    #     the lock file from the matching git tag, giving every user the
    #     SAME resolved dep set. This is what prevents "old install.sh
    #     picks up latest deps and breaks" — the dep versions are baked
    #     into uv.lock at release time.
    #   - Mismatch (new install.sh running on old clawcodex, OR a clawcodex
    #     release that pre-dates the [all] extra): uv sync rejects
    #     `--extra all` because the extra doesn't exist in pyproject.toml.
    #     We fall back to uv pip install, which is lenient about missing
    #     extras. Deps are no longer lock-pinned in this case, but the
    #     install at least succeeds.

    # --- With venv: uv sync operates on the venv created by create_venv.
    # --- Without venv (--no-venv): install to the active Python via --system.
    local install_target_args=()
    if [[ "$USE_VENV" -eq 1 ]]; then
        [[ -d ".venv" ]] || die "Venv missing at $CLAWCODEX_HOME/.venv — run without --no-venv or re-clone."
        install_target_args=(--python .venv/bin/python)
    else
        install_target_args=(--system)
    fi

    local uv_sync_log uv_pip_log
    uv_sync_log=$(mktemp -t uv-sync-XXXXXX 2>/dev/null || mktemp)
    uv_pip_log=$(mktemp -t uv-pip-XXXXXX 2>/dev/null || mktemp)

    if uv sync --extra all "${install_target_args[@]}" 2>"$uv_sync_log"; then
        log_ok "Dependencies installed (lock-pinned to uv.lock at $REPO_REF)"
        rm -f "$uv_sync_log" "$uv_pip_log"
        return
    fi

    # uv sync failed — inspect why and decide.
    local sync_err
    sync_err=$(cat "$uv_sync_log" 2>/dev/null || true)

    # Dump full log to stderr so --log-file (tee) captures it
    log_warn "uv sync failed — full error log:"
    cat "$uv_sync_log" >&2 2>/dev/null || true

    rm -f "$uv_sync_log"

    if echo "$sync_err" | grep -qE 'Extra `all` is not defined'; then
        # The clawcodex version we're installing doesn't have the [all]
        # extra in its pyproject.toml. This is expected for any release
        # that pre-dates the [all] extra (added with install.sh v1.1).
        log_warn "This clawcodex version has no [all] extra — falling back to uv pip install."
        log_warn "  Dependency versions will be resolved fresh (NOT lock-pinned)."
        log_warn "  For strict version pinning, use an install.sh whose"
        log_warn "  CLAWCODEX_VERSION matches a release that includes [all]."
    else
        log_warn "uv sync failed; falling back to uv pip install."
        log_warn "  Sync error was: ${sync_err:-<no stderr captured>}"
    fi

    if ! uv pip install "${install_target_args[@]}" -e ".[all]" 2>"$uv_pip_log"; then
        local pip_err
        pip_err=$(cat "$uv_pip_log" 2>/dev/null || true)
        log_warn "uv pip install failed — full error log:"
        cat "$uv_pip_log" >&2 2>/dev/null || true
        rm -f "$uv_pip_log"
        # uv's PEP 668 message has changed wording across versions; match
        # both the structured error code ("externally-managed-environment")
        # and the human message ("externally managed") defensively.
        if [[ "$USE_VENV" -eq 0 ]] && echo "$pip_err" | grep -qiE 'externally[ -]managed'; then
            log_warn "System Python is externally managed (PEP 668). Retrying with --break-system-packages."
            if ! uv pip install "${install_target_args[@]}" --break-system-packages -e ".[all]"; then
                die_with_help "uv pip install to system failed even with --break-system-packages." \
                              "Inspect the error above for missing system libraries." \
                              "Retry:    $0" \
                              "Or:       $0 uninstall && $0   (fresh install with venv)"
            fi
        else
            log_err "uv pip install failed: ${pip_err:-<no stderr captured>}"
            die_with_help "Both uv sync and uv pip install failed." \
                          "Re-run with --log-file <path> to capture full output." \
                          "Retry:    $0" \
                          "Diagnose: $0 doctor" \
                          "Clean:    $0 uninstall && $0"
        fi
    fi
    rm -f "$uv_pip_log"
    log_ok "Dependencies installed (fresh-resolve, NOT lock-pinned; target: $([[ $USE_VENV -eq 1 ]] && echo .venv || echo system))"
}

# ============================================================================
#  Locate the venv's entry-point binary
# ============================================================================
find_venv_entry() {
    local venv_dir="$1" name="$2"
    # Linux/macOS layout
    if [[ -x "$venv_dir/bin/$name" ]]; then
        echo "$venv_dir/bin/$name"; return 0
    fi
    # Windows layout (Git Bash / WSL interop)
    if [[ -x "$venv_dir/Scripts/$name.exe" ]]; then
        echo "$venv_dir/Scripts/$name.exe"; return 0
    fi
    if [[ -x "$venv_dir/Scripts/$name" ]]; then
        echo "$venv_dir/Scripts/$name"; return 0
    fi
    return 1
}

# ============================================================================
#  Register global commands
#  - We write tiny wrapper scripts in ~/.local/bin (more portable than symlinks
#    on Windows / Git Bash, and survives venv re-creation).
#  - `clawcodex` is registered as an alias for `clawcodex-dev` (the only
#    declared entry point in pyproject.toml).
# ============================================================================
register_commands() {
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        _script_p1
        echo "[DRY-RUN] would register: $LOCAL_BIN/clawcodex-dev, $LOCAL_BIN/clawcodex"
        return 0
    fi
    mkdir -p "$LOCAL_BIN"

    local entry
    if [[ "$USE_VENV" -eq 1 ]]; then
        # Venv mode: look for the entry inside the project's .venv
        if ! entry=$(find_venv_entry "$CLAWCODEX_HOME/.venv" "$ENTRY_POINT"); then
            die "Entry point '$ENTRY_POINT' not found inside $CLAWCODEX_HOME/.venv — dependency install may have failed."
        fi
    else
        # --no-venv mode: look for the entry on PATH (uv pip install --system
        # drops scripts in /usr/local/bin or ~/.local/bin). We check a few
        # common locations explicitly so we don't depend on the just-installed
        # PATH being effective in this very shell.
        entry=""
        for candidate in \
            "$HOME/.local/bin/$ENTRY_POINT" \
            "/usr/local/bin/$ENTRY_POINT" \
            "$(command -v "$ENTRY_POINT" 2>/dev/null || true)"; do
            if [[ -n "$candidate" && ( -x "$candidate" || -L "$candidate" ) ]]; then
                entry="$candidate"; break
            fi
        done
        [[ -n "$entry" ]] || die "Entry point '$ENTRY_POINT' not found on PATH after system install — check 'which $ENTRY_POINT'."
    fi

    write_wrapper() {
        local name="$1" target="$2"
        local wrapper="$LOCAL_BIN/$name"

        # Always (re)write so the wrapper reflects any new install dir.
        if [[ -L "$wrapper" || -e "$wrapper" ]]; then
            rm -f "$wrapper"
        fi

        cat > "$wrapper" <<EOF
#!/usr/bin/env bash
# Auto-generated by clawcodex install.sh — do not edit by hand.
# Regenerate by re-running install.sh.
# Point the runtime at the configured config dir; the wrapper itself is
# pinned to the install dir baked in at generation time, but the config
# dir can be re-pointed at runtime by the user via this env var.
export CLAWCODEX_CONFIG_DIR="\${CLAWCODEX_CONFIG_DIR:-${CONFIG_DIR}}"
exec "$target" "\$@"
EOF
        chmod +x "$wrapper"
        log_ok "$wrapper → $target  (CLAWCODEX_CONFIG_DIR=${CONFIG_DIR})"
    }

    write_wrapper "clawcodex-dev" "$entry"
    write_wrapper "clawcodex"    "$entry"
}

# ============================================================================
#  Patch shell rc files to include ~/.local/bin in PATH
# ============================================================================
update_shell_rc() {
    local path_line='export PATH="$HOME/.local:$HOME/.local/bin:$PATH"'
    local current_shell
    current_shell=$(basename "${SHELL:-bash}" 2>/dev/null || echo "bash")
    local rc_file=""

    case "$current_shell" in
        bash) rc_file="$HOME/.bashrc" ;;
        zsh)  rc_file="$HOME/.zshrc"  ;;
        *)    rc_file="$HOME/.profile" ;;
    esac

    if [[ ! -f "$rc_file" ]]; then
        log_warn "Shell rc file $rc_file not found — please add '$path_line' to your shell's startup file."
        return
    fi

    if grep -qF '$HOME/.local/bin' "$rc_file" 2>/dev/null; then
        log_ok "PATH already contains ~/.local/bin in $rc_file"
        return
    fi

    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        _script_p1
        echo "[DRY-RUN] would append PATH entry to: $rc_file"
        return
    fi

    {
        echo ""
        echo "$RC_MARKER"
        echo "$path_line"
    } >> "$rc_file"
    log_ok "Patched $rc_file (added ~/.local/bin to PATH)"
}

# ============================================================================
#  Post-install setup wizard (the interactive first-run configuration)
# ============================================================================
run_post_install_setup() {
    if [[ "$RUN_SETUP" -eq 0 ]]; then
        log_warn "Setup wizard skipped (--no-setup). Run 'clawcodex-dev' manually to configure."
        return
    fi

    log_info "Post-install setup wizard is available — launching clawcodex-dev setup…"
    # We intentionally do NOT exec a blocking interactive REPL here. The
    # install script must remain non-interactive so it can run unattended
    # in CI / Docker / by orchestrators. The wizard itself (if present) is
    # a subcommand the user runs themselves; we just announce it.
    if command -v clawcodex-dev >/dev/null 2>&1; then
        log_ok "Run one of:"
        echo -e "    ${C_BOLD}clawcodex-dev${C_RESET}          # start the interactive REPL (triggers first-run setup if config is empty)"
        echo -e "    ${C_BOLD}clawcodex-dev --help${C_RESET}  # see all options"
    else
        log_warn "clawcodex-dev not on PATH yet — run 'source ~/.bashrc' (or ~/.zshrc) first."
    fi
}

# ============================================================================
#  Install pipeline
# ============================================================================
install_main() {
    echo -e "${C_BOLD}clawcodex installer v${INSTALLER_VERSION}${C_RESET}"
    echo -e "  ${C_BOLD}OS:${C_RESET}          $OS"
    echo -e "  ${C_BOLD}Install dir:${C_RESET} $CLAWCODEX_HOME"
    echo -e "  ${C_BOLD}Config dir:${C_RESET}  $CONFIG_DIR"
    echo -e "  ${C_BOLD}Git ref:${C_RESET}     $REPO_REF"
    echo -e "  ${C_BOLD}Venv:${C_RESET}        $([[ $USE_VENV -eq 1 ]] && echo "create at $CLAWCODEX_HOME/.venv" || echo "${C_YELLOW}skipped (--no-venv, system Python)${C_RESET}")"
    echo -e "  ${C_BOLD}Setup wizard:${C_RESET} $([[ $RUN_SETUP -eq 1 ]] && echo "announce only (non-blocking)" || echo "${C_YELLOW}skipped (--no-setup)${C_RESET}")"
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        echo -e "  ${C_BOLD}Mode:${C_RESET}        ${C_YELLOW}DRY-RUN (no changes will be made)${C_RESET}"
    fi
    if [[ -n "${LOG_FILE:-}" ]]; then
        echo -e "  ${C_BOLD}Log file:${C_RESET}    $LOG_FILE"
    fi
    if [[ "${DEBUG:-0}" -eq 1 ]]; then
        echo -e "  ${C_BOLD}Debug:${C_RESET}       ${C_YELLOW}ON (set -x trace)${C_RESET}"
    fi

    log_step "1/10  Checking prerequisites"
    check_git

    log_step "2/10  Installing uv (Astral, no sudo)"
    # Re-source in case it wasn't on PATH at the top of the script.
    export PATH="$HOME/.local:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    install_uv

    log_step "3/10  Provisioning Python $PYTHON_MIN_VERSION+"
    ensure_python
    ensure_python_pyo3_compat

    log_step "4/10  Cloning / updating repository"
    clone_or_update_repo

    log_step "5/10  Initializing local release .env"
    ensure_local_env_file

    log_step "7/10  $([[ $USE_VENV -eq 1 ]] && echo "Creating virtual environment" || echo "Preparing (no venv — using system Python)")"
    create_venv

    log_step "8/10  Installing dependencies (uv sync --extra all, lock-pinned)"
    install_deps

    # Git hook installation deferred to follow-up version
    log_step "9/10  Registering global commands & patching PATH"
    register_commands
    update_shell_rc

    echo ""
    log_ok "Installation complete!"
    echo ""
    echo -e "  ${C_BOLD}Try it:${C_RESET}"
    echo -e "    clawcodex-dev --help      # primary command"
    echo -e "    clawcodex    --help       # alias of clawcodex-dev"
    echo ""
    echo -e "  ${C_BOLD}Installed at:${C_RESET}  $CLAWCODEX_HOME"
    echo -e "  ${C_BOLD}Config at:${C_RESET}    $CONFIG_DIR"
    echo -e "  ${C_BOLD}Commands at:${C_RESET}   $LOCAL_BIN/{clawcodex,clawcodex-dev}"
    echo ""

    # Post-install setup lives outside the numbered pipeline because it is
    # optional and varies the most.
    run_post_install_setup

    log_warn "Open a new shell, or run:  source ~/.bashrc   (or ~/.zshrc)"
}

# ============================================================================
#  CLI argument parser — populates the *OVERRIDE globals, then they're
#  resolved into the actual install/config/ref variables below.
# ============================================================================
REF_OVERRIDE=""
INSTALL_DIR_OVERRIDE=""
CONFIG_DIR_OVERRIDE=""
USE_VENV=1       # --no-venv flips to 0
RUN_SETUP=1      # --no-setup flips to 0
DRY_RUN=0        # --dry-run flips to 1
ASSUME_YES=0     # --yes/-y flips to 1
LOG_FILE=""      # --log-file <path>
DEBUG=0          # --debug flips to 1 (set -x trace)
SUBCOMMAND=""    # positional verb
SCRIPT_START_TS=$(date +%s)

print_usage_hint() {
    echo "Try '$0 --help' for usage." >&2
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            # --- Subcommands ---
            install)
                SUBCOMMAND="$1"; shift ;;

            # --- Option flags ---
            --ref)
                [[ $# -ge 2 ]] || { log_err "--ref requires a value (commit/tag/branch)"; print_usage_hint; exit 1; }
                REF_OVERRIDE="$2"; shift 2 ;;
            --install-dir)
                [[ $# -ge 2 ]] || { log_err "--install-dir requires a path"; print_usage_hint; exit 1; }
                INSTALL_DIR_OVERRIDE="$2"; shift 2 ;;
            --config-dir)
                [[ $# -ge 2 ]] || { log_err "--config-dir requires a path"; print_usage_hint; exit 1; }
                CONFIG_DIR_OVERRIDE="$2"; shift 2 ;;
            --log-file)
                [[ $# -ge 2 ]] || { log_err "--log-file requires a path"; print_usage_hint; exit 1; }
                LOG_FILE="$2"; shift 2 ;;
            --dry-run)
                DRY_RUN=1; shift ;;
            --yes|-y)
                ASSUME_YES=1; shift ;;
            --no-venv)
                USE_VENV=0; shift ;;
            --no-setup)
                RUN_SETUP=0; shift ;;
            --debug)
                DEBUG=1; shift ;;
            --version|-v)
                echo "install.sh v${INSTALLER_VERSION} (installs clawcodex v${CLAWCODEX_VERSION})"
                exit 0 ;;

            --)
                shift; break ;;
            -*)
                log_err "Unknown option: $1"; print_usage_hint; exit 1 ;;
            *)
                log_err "Unexpected positional argument: $1"; print_usage_hint; exit 1 ;;
        esac
    done
}

# ============================================================================
#  Entry point
# ============================================================================
parse_args "$@"

# Resolve overrides → effective install/config paths. Must run AFTER
# parse_args, otherwise INSTALL_DIR_OVERRIDE / REF_OVERRIDE are still
# empty when CLAWCODEX_HOME / REPO_REF get resolved and the flags are
# silently ignored.
CLAWCODEX_HOME="${INSTALL_DIR_OVERRIDE:-$DEFAULT_INSTALL_DIR}"
CLAWCODEX_PARENT_DIR="$(dirname -- "$CLAWCODEX_HOME")"
CONFIG_DIR="${CONFIG_DIR_OVERRIDE:-$DEFAULT_CONFIG_DIR}"
[[ -n "$REF_OVERRIDE" ]] && REPO_REF="$REF_OVERRIDE"

OS=$(detect_os)

# Bail out for native Windows shells — this script targets bash, not cmd/PS.
if [[ "$OS" == "unknown" ]] && [[ -n "${COMSPEC:-}" || -n "${WINDIR:-}" ]]; then
    cat >&2 <<'END_MSG'
✗ Native Windows shell detected (cmd.exe or PowerShell).

  install.sh is a bash script and cannot run directly in cmd or PowerShell.
  Please use one of the following options:

  Option A — Git Bash (recommended, zero-config):
    1. Install Git for Windows from https://git-scm.com/download/win
    2. Open "Git Bash" from the Start menu
    3. In Git Bash, run:    bash install.sh

  Option B — WSL2 (full Linux environment):
    1. Open PowerShell as Administrator and run:
         wsl --install -d Ubuntu
    2. Restart your computer
    3. Open the Ubuntu terminal and run:
         sudo apt update && sudo apt install -y git curl
         bash install.sh

  Option C — Install manually from source:
    1. Install Git, Python 3.11+, and curl
    2. Run:
         git clone https://gitcode.com/Ascend/AgentSDK %TEMP%/agentsdk
         cd %TEMP%/agentsdk/clawcodex-ascend
         pip install -e ".[all]"
    (See https://gitcode.com/Ascend/AgentSDK for details)
    On Linux/macOS, replace %TEMP% with /tmp.

END_MSG
    exit 1
fi

# Make uv visible early in case it's already installed but not on PATH for this shell.
export PATH="$HOME/.local:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Set up log-file tee if requested. Must happen AFTER parse_args so LOG_FILE
# is set, but BEFORE any other output. After this exec, [[ -t 1 ]] is false
# (it's a pipe), so the [install.sh] prefix is added on every line.
if [[ -n "$LOG_FILE" ]]; then
    log_file_dir=$(dirname -- "$LOG_FILE")
    if [[ ! -d "$log_file_dir" ]]; then
        mkdir -p "$log_file_dir" 2>/dev/null || { log_warn "Cannot create log dir $log_file_dir; --log-file ignored"; LOG_FILE=""; }
    fi
    if [[ -n "$LOG_FILE" ]]; then
        exec > >(tee -a "$LOG_FILE") 2>&1
    fi
fi

# Install the EXIT trap. Runs on every exit (normal, error, or signal).
# Emits a structured "DONE: success|FAILED" line plus log-file location.
trap '_on_exit_summary $?' EXIT

# Activate debug mode (set -x) if --debug was passed.  Must come after the
# trap setup so the trace output is visible from the very first command.
if [[ "$DEBUG" -eq 1 ]]; then
    set -x
fi

# Dispatch to subcommand. Default to 'install' when none was given.
case "${SUBCOMMAND:-install}" in
    install)   install_main ;;
    *)
        log_err "Unknown subcommand: $SUBCOMMAND"
        echo "Usage: $0 install"
        exit 1
        ;;
esac
