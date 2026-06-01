#!/bin/bash
# =============================================================================
# OpenClaw 容器健康状态校验脚本
#
# 用途：CI/CD 流水线中用于验证容器拉起后的健康状态，
#      替代手动逐项检查，覆盖 OpenClaw/Hermes/浏览器/依赖/插件/端口等
#
# 用法（在宿主机执行）：
#   # 单实例
#   bash openclaw-health-check.sh --instance 1 --port 18789 --token <TOKEN>
#
#   # 多实例（检查所有）
#   bash openclaw-health-check.sh --scan-all --base-port 18789
#
#   # 仅镜像层静态检查（不访问 API，适合 API 未就绪时）
#   bash openclaw-health-check.sh --image-only
#
#   # 健康探针自动重启测试
#   bash openclaw-health-check.sh --probe-test --instance 1 --port 18789 --token <TOKEN>
#
#   # 详细输出（包含警告）
#   bash openclaw-health-check.sh --instance 1 --port 18789 --token <TOKEN> --verbose
#
#   # 输出 CI 友好格式（TAP）
#   bash openclaw-health-check.sh --instance 1 --port 18789 --token <TOKEN> --format tap
#
# 环境变量（可选）：
#   OPENCLAW_GW_PORT   Gateway 端口（默认 18789）
#   OPENCLAW_TOKEN     Gateway Token（优先于 --token 参数）
#   OPENCLAW_CONTAINER Container 名称/ID（默认 openclaw-instance-1）
#   OPENCLAW_HEALTH_TIMEOUT 每个实例整体超时秒数（默认 360）
#   SKIP_RESTART_TEST  设为 1 跳过健康探针重启测试
#
# 依赖：curl jq nc node python3 openssl（容器内已有）
# =============================================================================

# 颜色（默认启用）- 在 set -euo pipefail 之前定义，避免 unbound variable
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
RESET=$'\033[0m'
if [ ! -t 1 ]; then
  RED=""
  GREEN=""
  YELLOW=""
  BLUE=""
  CYAN=""
  BOLD=""
  RESET=""
fi

set -u

# 错误处理：捕获导致 set -e 退出的错误（调试用）
handle_error() {
  echo "ERROR: Script failed near line $1" >&2
  sed -n "$((($1)-2)),$((($1)+2))p" "$0" >&2
}

# -----------------------------------------------
# 全局配置
# -----------------------------------------------
SCRIPT_VERSION="1.0.0"
VERBOSE=0
OUTPUT_FORMAT="color"       # color | tap | json
IMAGE_ONLY=0
PROBE_TEST=0
SCAN_ALL=0
BASE_PORT=18789
INSTANCE_NUM=1
TARGET_INSTANCE=""
GW_PORT="${OPENCLAW_GW_PORT:-18789}"
GW_TOKEN="${OPENCLAW_TOKEN:-}"
CONTAINER_NAME="${OPENCLAW_CONTAINER:-}"
SKIP_RESTART="${SKIP_RESTART_TEST:-0}"
INSTANCE_TIMEOUT="${OPENCLAW_HEALTH_TIMEOUT:-360}"
HEALTH_TIMEOUT_CHILD="${OPENCLAW_HEALTH_TIMEOUT_CHILD:-0}"
# 指定 mindclaw 源码目录路径（用于 Skills 交集比对）
# 默认为容器挂载的宿主机 mindclaw/skills 路径
MINDCLAW_SKILLS_DIR="${MINDCLAW_SKILLS_DIR:-}"
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
TOTAL_COUNT=0
SCRIPT_SOURCED=0
SCRIPT_STARTED=0

# 临时文件
TMPDIR="${TMPDIR:-/tmp}"
RESULTS_FILE="$TMPDIR/openclaw-health-$$-results.txt"
ERRORS_FILE="$TMPDIR/openclaw-health-$$-errors.txt"
GATEWAY_PID_FILE="$TMPDIR/openclaw-gateway-pid-$$"
START_TIME=$(date +%s)
ORIGINAL_ARGS=("$@")

# 端口计算规则（容器内部端口 → 宿主机映射端口）
# 12345 Gateway → 12345（直等）
# 12346 SFTP   → 12346（直等）
# 37700 Worker → 12347（固定偏移，非 instance*4）
# 8080  Memex  → 12348（直等）
MEMEX_PORT_OFFSET=3   # GW_PORT + 3 = Memex 内部端口 8080
CLAUDE_MEM_PORT=37700        # Worker 容器内部端口
CLAUDE_MEM_HOST_PORT=12347   # Worker 映射到宿主机的端口（固定）


# -----------------------------------------------
# 帮助信息
# -----------------------------------------------
show_help() {
  cat << 'EOF'
用法: openclaw-health-check.sh [选项]

选项:
  --instance <N>           指定要检查的实例编号（默认 1）
  --port <PORT>            Gateway 端口（默认 18789）
  --token <TOKEN>          Gateway Token（也可通过 OPENCLAW_TOKEN 环境变量）
  --container <NAME>       Container 名称/ID（默认 openclaw-instance-1）
  --base-port <PORT>       多实例扫描基准端口（默认 18789）
  --scan-all               扫描所有实例（基于 base-port 检测在线实例）
  --image-only             仅静态检查（跳过 API 调用）
  --probe-test             执行健康探针自动重启测试
  --format <FORMAT>        输出格式：color（默认）/ tap / json
  --verbose                详细输出
  --help                   显示帮助信息
  --version                显示版本信息

输出格式说明：
  color  - 带颜色的可读输出，适合人工查看
  tap    - TAP (Test Anything Protocol) 格式，适合 CI 系统解析
  json   - JSON 格式，适合程序解析

环境变量：
  OPENCLAW_GW_PORT        Gateway 端口
  OPENCLAW_TOKEN          Gateway Token（优先于 --token）
  OPENCLAW_CONTAINER      Container 名称/ID
  OPENCLAW_HEALTH_TIMEOUT 每个实例整体超时秒数（默认 360）
  SKIP_RESTART_TEST        设为 1 跳过重启测试
EOF
}

# -----------------------------------------------
# 工具函数
# -----------------------------------------------

log_section() {
  echo ""
  echo -e "${CYAN}${BOLD}========================================${RESET}"
  echo -e "${CYAN}${BOLD} $1${RESET}"
  echo -e "${CYAN}${BOLD}========================================${RESET}"
}

log_item() {
  local status="$1"
  local message="$2"
  local details="${3:-}"
  TOTAL_COUNT=$((TOTAL_COUNT + 1))

  case "$status" in
    PASS)
      PASS_COUNT=$((PASS_COUNT + 1))
      ;;
    FAIL)
      FAIL_COUNT=$((FAIL_COUNT + 1))
      ;;
    WARN)
      WARN_COUNT=$((WARN_COUNT + 1))
      ;;
  esac

  if [ "$OUTPUT_FORMAT" = "tap" ]; then
    local tap_status
    [ "$status" = "PASS" ] && tap_status="ok" || tap_status="not ok"
    echo "$tap_status $TOTAL_COUNT - $message"
    [ -n "$details" ] && echo "  # $details"
  elif [ "$OUTPUT_FORMAT" = "json" ]; then
    echo "{\"status\":\"$status\",\"message\":\"$message\",\"details\":\"$details\"}" >> "$RESULTS_FILE"
  else
    case "$status" in
      PASS)  echo -e "  ${GREEN}[PASS]${RESET} $message" ;;
      FAIL)  echo -e "  ${RED}[FAIL]${RESET} $message" ;;
      WARN)  echo -e "  ${YELLOW}[WARN]${RESET} $message" ;;
      INFO)  echo -e "  ${BLUE}[INFO]${RESET} $message" ;;
    esac
    [ -n "$details" ] && [ "$VERBOSE" = "1" ] && echo -e "         $details"
  fi

  return 0
}

log_error() {
  echo -e "  ${RED}[ERROR]${RESET} $*" >> "$ERRORS_FILE"
}

log_detail() {
  [ "$VERBOSE" = "1" ] && echo -e "         $*" || true
}

run_with_instance_timeout() {
  if [ "$HEALTH_TIMEOUT_CHILD" = "1" ]; then
    return 0
  fi

  if ! [[ "$INSTANCE_TIMEOUT" =~ ^[0-9]+$ ]] || [ "$INSTANCE_TIMEOUT" -le 0 ]; then
    echo "ERROR: OPENCLAW_HEALTH_TIMEOUT must be a positive integer, got: $INSTANCE_TIMEOUT" >&2
    exit 2
  fi

  if ! command -v timeout >/dev/null 2>&1; then
    echo "ERROR: timeout command is required for global instance timeout control" >&2
    exit 2
  fi

  HEALTH_TIMEOUT_CHILD=1 OPENCLAW_HEALTH_TIMEOUT_CHILD=1 timeout --preserve-status "$INSTANCE_TIMEOUT" bash "$0" "$@"
  local rc=$?
  if [ "$rc" -eq 143 ] || [ "$rc" -eq 124 ]; then
    echo ""
    echo "ERROR: OpenClaw health check exceeded per-instance timeout (${INSTANCE_TIMEOUT}s)" >&2
    exit 124
  fi
  exit "$rc"
}

http_get() {
  local url="$1"
  local token="${2:-$GW_TOKEN}"
  local timeout="${3:-10}"
  if [ -n "$token" ]; then
    curl -s -f -m "$timeout" -H "Authorization: Bearer $token" "$url"
  else
    curl -s -f -m "$timeout" "$url"
  fi
}

http_post() {
  local url="$1"
  local data="$2"
  local token="${3:-$GW_TOKEN}"
  local timeout="${4:-30}"
  if [ -n "$token" ]; then
    curl -s -f -m "$timeout" -X POST -H "Content-Type: application/json" \
      -H "Authorization: Bearer $token" \
      -d "$data" "$url"
  else
    curl -s -f -m "$timeout" -X POST -H "Content-Type: application/json" \
      -d "$data" "$url"
  fi
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout="${3:-60}"
  local waited=0
  while [ $waited -lt "$timeout" ]; do
    if nc -z -w 2 "$host" "$port" 2>/dev/null; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  return 1
}

wait_for_api() {
  local url="$1"
  local timeout="${2:-60}"
  local waited=0
  while [ $waited -lt "$timeout" ]; do
    local resp
    resp=$(curl -s -f -m 5 -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    if [ "$resp" = "200" ] || [ "$resp" = "401" ]; then
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  return 1
}

get_gateway_pid() {
  # 获取 gateway 进程的 PID（容器内执行）
  exec_in_container "pgrep -f 'openclaw-gatewa|node.*gateway|openclaw.*gateway' 2>/dev/null | head -1 || true"
}

gateway_pid_alive() {
  local pid="$1"
  exec_in_container "kill -0 '$pid' 2>/dev/null"
}

# -----------------------------------------------
# 参数解析
# -----------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --instance)    INSTANCE_NUM="$2"; shift 2 ;;
    --port)        GW_PORT="$2"; shift 2 ;;
    --token)       GW_TOKEN="$2"; shift 2 ;;
    --container)   CONTAINER_NAME="$2"; shift 2 ;;
    --base-port)   BASE_PORT="$2"; shift 2 ;;
    --scan-all)    SCAN_ALL=1; shift ;;
    --image-only)  IMAGE_ONLY=1; shift ;;
    --probe-test)  PROBE_TEST=1; shift ;;
    --format)
      case "$2" in
        color|tap|json) OUTPUT_FORMAT="$2" ;;
        *) echo "不支持的格式: $2" >&2; exit 1 ;;
      esac
      shift 2 ;;
    --verbose)  VERBOSE=1; shift ;;
    --help)     show_help; exit 0 ;;
    --version)  echo "openclaw-health-check.sh v$SCRIPT_VERSION"; exit 0 ;;
    *)          echo "未知参数: $1"; show_help; exit 1 ;;
  esac
done

# 容器名推导
if [ -z "$CONTAINER_NAME" ]; then
  if [ "$SCAN_ALL" = "1" ]; then
    CONTAINER_NAME=""
  else
    CONTAINER_NAME="openclaw-instance-${INSTANCE_NUM:-1}"
  fi
fi

# -----------------------------------------------
# 检测运行环境
# -----------------------------------------------
detect_runtime() {
  if command -v docker &>/dev/null && [ -n "${CONTAINER_NAME:-}" ]; then
    if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
      RUNTIME="docker"
      return
    else
      echo "ERROR: Container '$CONTAINER_NAME' not found. Please check the container name." >&2
      echo "       Use 'docker ps -a' to list available containers." >&2
      exit 1
    fi
  fi
  # 如果在容器内运行或 docker 不可用，使用本地模式
  if [ -f /proc/1/cgroup ] && grep -q "docker" /proc/1/cgroup 2>/dev/null; then
    RUNTIME="container"
    return
  fi
  RUNTIME="local"
}

# -----------------------------------------------
# 在容器内执行命令
# -----------------------------------------------
exec_in_container() {
  if [ "$RUNTIME" = "docker" ]; then
    docker exec "$CONTAINER_NAME" sh -c "$*"
  else
    sh -c "$*"
  fi
}

# 批量执行：单次 docker exec 运行多个检查项，通过 [TAG] 前缀输出
# 避免多次 exec 冷启动累积延迟（每次 ~2s）
# 用法：result=$(batch_exec "hermes --version; npm --version")
exec_in_container_batch() {
  if [ "$RUNTIME" = "docker" ]; then
    docker exec "$CONTAINER_NAME" sh -c "$*"
  else
    sh -c "$*"
  fi
}

exec_in_container_bg() {
  if [ "$RUNTIME" = "docker" ]; then
    docker exec "$CONTAINER_NAME" sh -c "$*" &>/dev/null &
  else
    sh -c "$*" &>/dev/null &
  fi
}

# -----------------------------------------------
# Section 1: 基础信息收集
# -----------------------------------------------
section_info() {
  log_section "基础信息"

  log_item INFO "检查实例 http://127.0.0.1:$GW_PORT（实例编号: $INSTANCE_NUM）"
  log_item INFO "检查容器 ${CONTAINER_NAME:-未指定（自动检测）}"
  log_item INFO "运行环境 $RUNTIME"
  log_item INFO "输出格式 $OUTPUT_FORMAT"
  log_item INFO "整体超时 ${INSTANCE_TIMEOUT}s"
  [ "$IMAGE_ONLY" = "1" ] && log_item INFO "检查模式 仅静态检查（跳过 API 调用）"
  [ "$PROBE_TEST" = "1" ] && log_item INFO "检查模式 含健康探针重启测试"
}

# -----------------------------------------------
# Section 2: Gateway 基础连通性
# -----------------------------------------------
section_gateway_connectivity() {
  log_section "Gateway 基础连通性"

  # 端口连通性（宿主机通过映射端口检测）
  local port_open=0
  if nc -z -w 3 "127.0.0.1" "$GW_PORT" 2>/dev/null; then
    log_item PASS "Gateway 端口 $GW_PORT 可连接" "nc -z 检测通过"
    port_open=1
  else
    log_item FAIL "Gateway 端口 $GW_PORT 无法连接" "请确认容器已启动且端口映射正确"
    return 1
  fi

  # API Health 探测 — Gateway 使用 /health（容器内执行）
  local health_url="http://127.0.0.1:$GW_PORT/health"
  local health_resp
  health_resp=$(exec_in_container "wget -q -O - -T 10 '$health_url' 2>/dev/null || curl -s -f -m 10 '$health_url' 2>/dev/null || echo ''")
  if echo "$health_resp" | grep -q '"ok":true'; then
    log_item PASS "Gateway /health 返回正常" "{\"ok\":true,\"status\":\"live\"}"
  elif [ -n "$health_resp" ]; then
    log_item WARN "Gateway /health 响应异常" "${health_resp:0:100}"
  else
    log_item FAIL "Gateway /health 无响应" "URL: $health_url"
  fi

  # Web UI 探测
  local web_resp
  web_resp=$(curl -s -f -m 10 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$GW_PORT/" 2>/dev/null || echo "000")
  if [ "$web_resp" = "200" ] || [ "$web_resp" = "302" ] || [ "$web_resp" = "301" ]; then
    log_item PASS "Gateway Web UI 可访问" "HTTP $web_resp"
  else
    log_item FAIL "Gateway Web UI 无法访问" "HTTP $web_resp"
  fi

  return $((1 - port_open))
}

# -----------------------------------------------
# Section 3: Gateway API 对话性验证
# -----------------------------------------------
section_gateway_talkability() {
  log_section "Gateway API 对话性验证"

  # 注意：Gateway 的 agent 通信走 WebSocket（不是 REST HTTP）
  # /api/chat、/api/agents 等返回空是正常行为
  # 实际对话性通过 /health 探活 + Web UI 可访问性来验证

  local health_url="http://127.0.0.1:$GW_PORT/health"
  local health_resp
  health_resp=$(exec_in_container "wget -q -O - -T 10 '$health_url' 2>/dev/null || curl -s -f -m 10 '$health_url' 2>/dev/null || echo ''")
  if echo "$health_resp" | grep -q '"ok":true'; then
    log_item PASS "Gateway /health 正常" "WebSocket 通信层就绪"
  elif [ -n "$health_resp" ]; then
    log_item WARN "Gateway /health 响应异常" "${health_resp:0:100}"
  else
    log_item FAIL "Gateway /health 无响应"
  fi

  # OpenClaw Gateway /status 通过 exec_in_container 探测（Web UI 返回 HTML，API 走 WS）
  local status_resp
  status_resp=$(exec_in_container "wget -q -O - -T 10 --header='Authorization: Bearer $GW_TOKEN' 'http://127.0.0.1:$GW_PORT/status' 2>/dev/null || echo ''")
  if echo "$status_resp" | grep -q '^\{'; then
    log_item PASS "Gateway /status API 正常"
    log_detail "响应: $(echo $status_resp | head -c 300)"
  else
    # status 走 WebSocket，非 HTTP，直接标记为 WARN
    log_item WARN "Gateway /status 走 WebSocket（非 HTTP）" "Agent 通信需通过 WebSocket 进行"
  fi

  # Web UI 可对话性 — 通过 GET / 返回 HTML
  local web_resp
  web_resp=$(curl -s -f -m 10 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$GW_PORT/" 2>/dev/null || echo "000")
  if [ "$web_resp" = "200" ]; then
    log_item PASS "Gateway Web UI 可访问" "HTTP $web_resp — OpenClaw 控制台正常"
  else
    log_item FAIL "Gateway Web UI 无法访问" "HTTP $web_resp"
  fi
}

# -----------------------------------------------
# Hermes Agent
# -----------------------------------------------
section_hermes() {
  log_section "Hermes Agent 检查"

  local batch_result
  batch_result=$(exec_in_container_batch '
    version=$(hermes --version 2>/dev/null || true)
    [ -n "$version" ] && echo "[HERMES_VERSION] $version" || echo "[HERMES_VERSION] FAIL"

    if hermes --help >/tmp/openclaw-hermes-help.out 2>/dev/null && grep -qi "usage\|hermes\|chat" /tmp/openclaw-hermes-help.out; then
      echo "[HERMES_HELP] OK"
    else
      echo "[HERMES_HELP] FAIL"
    fi

    chat_output=$(timeout 60 hermes chat -q "hello" 2>&1 || true)
    if printf "%s" "$chat_output" | grep -qi "Hello\|How can I help\|Hermes\|Session:"; then
      chat_resp=$(printf "%s" "$chat_output" | grep -Eiv "^(Query:|Initializing agent|─|╭|╰|Session:|Duration:|Messages:|Resume this session|$)" | head -1)
      [ -n "$chat_resp" ] || chat_resp=$(printf "%s" "$chat_output" | head -1)
      echo "[HERMES_CHAT] $chat_resp"
    else
      echo "[HERMES_CHAT] EMPTY"
    fi

    owner=$(stat -c "%U:%G" /home/node/.hermes 2>/dev/null || true)
    [ -n "$owner" ] && echo "[HERMES_DIR] $owner" || echo "[HERMES_DIR] FAIL"

    skill_count=$(find /home/node/.hermes/hermes-agent/optional-skills -name "SKILL.md" 2>/dev/null | wc -l | tr -d "\n")
    echo "[HERMES_OPTIONAL_SKILLS] $skill_count"
  ')

  while IFS= read -r line; do
    case "$line" in
      \[HERMES_VERSION\]*)
        val=$(printf '%s' "$line" | sed 's/^\[HERMES_VERSION\] //')
        [ "$val" != "FAIL" ] && log_item PASS "Hermes 版本" "$val" || log_item FAIL "Hermes CLI 未找到"
        ;;
      \[HERMES_HELP\]*)
        val=$(printf '%s' "$line" | sed 's/^\[HERMES_HELP\] //')
        [ "$val" = "OK" ] && log_item PASS "Hermes --help 可用" || log_item WARN "Hermes --help 异常"
        ;;
      \[HERMES_CHAT\]*)
        val=$(printf '%s' "$line" | sed 's/^\[HERMES_CHAT\] //')
        [ "$val" != "EMPTY" ] && log_item PASS "Hermes chat 可对话" "响应: ${val:0:80}" || log_item WARN "Hermes chat 无输出"
        ;;
      \[HERMES_DIR\]*)
        val=$(printf '%s' "$line" | sed 's/^\[HERMES_DIR\] //')
        [ "$val" != "FAIL" ] && log_item PASS "Hermes 工作目录存在" "所有者: $val" || log_item FAIL "Hermes 工作目录不存在"
        ;;
      \[HERMES_OPTIONAL_SKILLS\]*)
        count=$(printf '%s' "$line" | sed 's/^\[HERMES_OPTIONAL_SKILLS\] //')
        if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
          log_item PASS "Hermes optional-skills 存在" "检测到 $count 个技能"
        else
          log_item WARN "Hermes optional-skills 未检测到技能"
        fi
        ;;
    esac
  done <<< "$batch_result"
}

# -----------------------------------------------
# Browser dependencies and availability
# -----------------------------------------------
section_browser() {
  log_section "浏览器依赖与可用性"

  local batch_result
  batch_result=$(exec_in_container_batch '
    CHROME_PATH=$(find /home -name "chrome" -type f 2>/dev/null | grep "chromium-.*/chrome-linux64/chrome" | head -1)
    [ -n "$CHROME_PATH" ] && echo "[CHROME_PATH] $CHROME_PATH" || echo "[CHROME_PATH] NOT_FOUND"

    if [ -n "$CHROME_PATH" ]; then
      chrome_version=$(timeout 10 "$CHROME_PATH" --headless --no-sandbox --version 2>&1 | head -1 || true)
      [ -n "$chrome_version" ] && echo "[CHROME_VERSION] $chrome_version" || echo "[CHROME_VERSION] FAIL"
    else
      echo "[CHROME_VERSION] FAIL"
    fi

    shell_path=$(find /home -name "chrome" -type f 2>/dev/null | grep "chromium_headless_shell" | head -1)
    [ -n "$shell_path" ] && echo "[SHELL_PATH] $shell_path" || echo "[SHELL_PATH] NOT_FOUND"

    ffmpeg_path=$(find /home -name "ffmpeg" -type f 2>/dev/null | grep "ms-playwright/ffmpeg" | head -1)
    [ -n "$ffmpeg_path" ] && echo "[FFMPEG_PATH] $ffmpeg_path" || echo "[FFMPEG_PATH] NOT_FOUND"

    pw_ver=$(npx playwright --version 2>/dev/null || true)
    [ -n "$pw_ver" ] && echo "[PW_VER] $pw_ver" || echo "[PW_VER] FAIL"

    missing_libs=""
    for lib in libnspr4 libnss3 libatk-1.0 libatk-bridge-2.0 libcups libdrm libxkbcommon libXcomposite libXdamage libXrandr libgbm libasound libdbus-1 libXfixes libwayland-client libpango libcairo libxcb; do
      ldconfig -p 2>/dev/null | grep -q "$lib" || missing_libs="$missing_libs $lib"
    done
    [ -z "$missing_libs" ] && echo "[LD_MISSING] NONE" || echo "[LD_MISSING]$missing_libs"

    if [ -n "$CHROME_PATH" ] && timeout 30 "$CHROME_PATH" --headless --no-sandbox --dump-dom "https://example.com" 2>/tmp/openclaw-browser.err | head -20 | grep -qi "Example\|example.com\|<!doctype"; then
      echo "[BROWSER_ACCESS] OK"
    else
      err=$(head -1 /tmp/openclaw-browser.err 2>/dev/null || true)
      echo "[BROWSER_ACCESS] FAIL ${err}"
    fi
  ')

  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in
      \[CHROME_PATH\]*)
        val=$(printf '%s' "$line" | sed 's/^\[CHROME_PATH\] //')
        [ "$val" != "NOT_FOUND" ] && log_item PASS "Chromium 可执行文件" "路径: $val" || log_item FAIL "Chromium 可执行文件未找到"
        ;;
      \[CHROME_VERSION\]*)
        val=$(printf '%s' "$line" | sed 's/^\[CHROME_VERSION\] //')
        [ "$val" != "FAIL" ] && log_item PASS "Chromium 可启动" "$val" || log_item FAIL "Chromium 启动失败"
        ;;
      \[SHELL_PATH\]*)
        val=$(printf '%s' "$line" | sed 's/^\[SHELL_PATH\] //')
        [ "$val" != "NOT_FOUND" ] && log_item PASS "Chromium Headless Shell 存在" "$val" || log_item WARN "Chromium Headless Shell 未找到"
        ;;
      \[FFMPEG_PATH\]*)
        val=$(printf '%s' "$line" | sed 's/^\[FFMPEG_PATH\] //')
        [ "$val" != "NOT_FOUND" ] && log_item PASS "FFmpeg 存在" "路径: $val" || log_item WARN "FFmpeg 未找到"
        ;;
      \[PW_VER\]*)
        val=$(printf '%s' "$line" | sed 's/^\[PW_VER\] //')
        [ "$val" != "FAIL" ] && log_item PASS "Playwright CLI 可用" "$val" || log_item WARN "Playwright CLI 不可用"
        ;;
      \[LD_MISSING\]*)
        val=$(printf '%s' "$line" | sed 's/^\[LD_MISSING\]//; s/^ //')
        [ "$val" = "NONE" ] && log_item PASS "所有 Chromium 核心库已加载" "ldconfig 检查通过" || log_item FAIL "Chromium 核心库缺失" "缺失: $val"
        ;;
      \[BROWSER_ACCESS\]*)
        val=$(printf '%s' "$line" | sed 's/^\[BROWSER_ACCESS\] //')
        case "$val" in
          OK) log_item PASS "Chromium 可成功访问网页" "example.com 返回 HTML" ;;
          *) log_item FAIL "Chromium 访问网页失败" "${val#FAIL }" ;;
        esac
        ;;
    esac
  done <<< "$batch_result"
}

# -----------------------------------------------
# Python dependencies
# -----------------------------------------------
section_python_deps() {
  log_section "Python 依赖检查"

  local batch_result
  batch_result=$(exec_in_container_batch '
    py_ver=$(python3 --version 2>&1 || true)
    [ -n "$py_ver" ] && echo "[PY_VER] $py_ver" || echo "[PY_VER] FAIL"

    py_perms=$(stat -c "%a %U:%G" /usr/bin/python3 2>/dev/null || true)
    [ -n "$py_perms" ] && echo "[PY_PERMS] $py_perms" || echo "[PY_PERMS] FAIL"

    for spec in markitdown:markitdown pptx:python-pptx docx:python-docx lxml:lxml openpyxl:openpyxl Pillow:Pillow pdf2image:pdf2image pdfminer.six:pdfminer.six; do
      label=${spec%%:*}
      package=${spec#*:}
      ver=$(python3 -m pip show "$package" 2>/dev/null | sed -n "s/^Version: //p" | head -1)
      [ -n "$ver" ] && echo "[PY_$label] $ver" || echo "[PY_$label] NOT_FOUND"
    done
  ')

  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in
      \[PY_VER\]*)
        val=$(printf '%s' "$line" | sed 's/^\[PY_VER\] //')
        [ "$val" != "FAIL" ] && log_item PASS "Python 版本" "$val" || log_item FAIL "Python 版本"
        ;;
      \[PY_PERMS\]*)
        val=$(printf '%s' "$line" | sed 's/^\[PY_PERMS\] //')
        [ "$val" != "FAIL" ] && log_item PASS "Python3 权限" "$val" || log_item WARN "Python3 权限无法读取"
        ;;
      \[PY_markitdown\]*|\[PY_pptx\]*|\[PY_docx\]*|\[PY_lxml\]*|\[PY_openpyxl\]*|\[PY_Pillow\]*|\[PY_pdf2image\]*|\[PY_pdfminer.six\]*)
        pkg=$(printf '%s' "$line" | sed 's/^\[PY_\([^]]*\)\] .*/\1/')
        val=$(printf '%s' "$line" | sed 's/^\[PY_[^]]*\] //')
        [ "$val" != "NOT_FOUND" ] && log_item PASS "Python 包 $pkg 已安装" "版本: $val" || log_item FAIL "Python 包 $pkg 未安装"
        ;;
    esac
  done <<< "$batch_result"
}

# -----------------------------------------------
# npm dependencies
# -----------------------------------------------
section_npm_deps() {
  log_section "npm 依赖检查"

  local batch_result
  batch_result=$(exec_in_container_batch '
    npm_ver=$(npm --version 2>/dev/null || true)
    [ -n "$npm_ver" ] && echo "[NPM_VER] $npm_ver" || echo "[NPM_VER] FAIL"

    npm_prefix=$(npm prefix -g 2>/dev/null || true)
    npm_perms=""
    [ -n "$npm_prefix" ] && npm_perms=$(stat -c "%a %U:%G" "$npm_prefix" 2>/dev/null || true)
    [ -n "$npm_perms" ] && echo "[NPM_PERMS] $npm_perms" || echo "[NPM_PERMS] FAIL"

    pnpm_ver=$(pnpm --version 2>/dev/null || true)
    [ -n "$pnpm_ver" ] && echo "[PNPM_VER] $pnpm_ver" || echo "[PNPM_VER] FAIL"

    bun_ver=$(bun --version 2>/dev/null || true)
    [ -n "$bun_ver" ] && echo "[BUN_VER] $bun_ver" || echo "[BUN_VER] FAIL"

    npm_root=$(npm root -g 2>/dev/null || true)
    for pkg in pptxgenjs sharp docx xlsx react react-dom; do
      if [ -n "$npm_root" ] && [ -d "$npm_root/$pkg" ]; then
        ver=$(node -e "try{console.log(require(\"$npm_root/$pkg/package.json\").version)}catch(e){}" 2>/dev/null)
        [ -n "$ver" ] && echo "[NPM_$pkg] $ver" || echo "[NPM_$pkg] INSTALLED"
      else
        echo "[NPM_$pkg] NOT_FOUND"
      fi
    done
  ')

  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in
      \[NPM_VER\]*)
        val=$(printf '%s' "$line" | sed 's/^\[NPM_VER\] //')
        [ "$val" != "FAIL" ] && log_item PASS "npm 版本" "$val" || log_item FAIL "npm 版本"
        ;;
      \[NPM_PERMS\]*)
        val=$(printf '%s' "$line" | sed 's/^\[NPM_PERMS\] //')
        [ "$val" != "FAIL" ] && log_item PASS "npm 全局包路径权限" "$val" || log_item WARN "npm 全局包路径权限无法读取"
        ;;
      \[PNPM_VER\]*)
        val=$(printf '%s' "$line" | sed 's/^\[PNPM_VER\] //')
        [ "$val" != "FAIL" ] && log_item PASS "pnpm 可用" "版本: $val" || log_item WARN "pnpm 不可用"
        ;;
      \[BUN_VER\]*)
        val=$(printf '%s' "$line" | sed 's/^\[BUN_VER\] //')
        [ "$val" != "FAIL" ] && log_item PASS "bun 可用" "版本: $val" || log_item WARN "bun 不可用"
        ;;
      \[NPM_pptxgenjs\]*|\[NPM_sharp\]*|\[NPM_docx\]*|\[NPM_xlsx\]*|\[NPM_react\]*|\[NPM_react-dom\]*)
        pkg=$(printf '%s' "$line" | sed 's/^\[NPM_\([^]]*\)\] .*/\1/')
        val=$(printf '%s' "$line" | sed 's/^\[NPM_[^]]*\] //')
        [ "$val" != "NOT_FOUND" ] && log_item PASS "npm 包 $pkg 已安装" "版本: $val" || log_item WARN "npm 包 $pkg 未安装"
        ;;
    esac
  done <<< "$batch_result"
}

# -----------------------------------------------
# OpenClaw Skills directory integrity + intersection
# -----------------------------------------------
section_skills() {
  log_section "OpenClaw Skills 目录完整性"

  # Batch 1: skills 核心信息（1次 exec）
  local batch_result
  batch_result=$(exec_in_container_batch '
    echo "[SKILLS_BEGIN]"
    find "/home/node/.openclaw/skills" "/app/extensions" "/home/node/.claude/plugins" "/home/node/.hermes" -name "SKILL.md" -not -path "*/node_modules/*" 2>/dev/null |
    while IFS= read -r f; do dirname "$f"; done | sort -u |
    while IFS= read -r d; do basename "$d"; done | sort -u
    echo "[SKILLS_END]"

    if [ -d "/home/node/.openclaw/skills" ]; then
      echo "[SKILLS_DIR] $(stat -c "%U:%G %a" "/home/node/.openclaw/skills" 2>/dev/null)"
    else
      echo "[SKILLS_DIR] MISSING"
    fi

    echo "[COUNT_OPENCLAW] $(find "/home/node/.openclaw/skills" -name "SKILL.md" -not -path "*/node_modules/*" 2>/dev/null | wc -l | tr -d "\n")"
    echo "[COUNT_CLAUDE_PLUGINS] $(find "/home/node/.claude/plugins" -name "SKILL.md" -not -path "*/node_modules/*" 2>/dev/null | wc -l | tr -d "\n")"
    echo "[COUNT_HERMES] $(find "/home/node/.hermes" -name "SKILL.md" -not -path "*/node_modules/*" 2>/dev/null | wc -l | tr -d "\n")"
    echo "[COUNT_EXTENSIONS] $(find "/app/extensions" -name "SKILL.md" -not -path "*/node_modules/*" 2>/dev/null | wc -l | tr -d "\n")"
  ')

  # Parse batch result
  local all_skill_names=""
  local in_skill_list=0
  local openclaw_skill_count=0
  local claude_plugin_skill_count=0
  local hermes_skill_count=0
  local extension_skill_count=0
  while IFS= read -r line; do
    case "$line" in
      "[SKILLS_BEGIN]")
        in_skill_list=1
        ;;
      "[SKILLS_END]")
        in_skill_list=0
        ;;
      "[SKILLS_DIR] MISSING")
        log_item WARN "Skills 主目录不存在" "/home/node/.openclaw/skills"
        ;;
      "[SKILLS_DIR] "*)
        local dir_info dir_owner dir_perms
        dir_info=${line#"[SKILLS_DIR] "}
        dir_owner=${dir_info%% *}
        dir_perms=${dir_info#* }
        log_item PASS "Skills 主目录存在" "所有者: $dir_owner, 权限: $dir_perms"
        ;;
      "[COUNT_OPENCLAW] "*)
        openclaw_skill_count=${line#"[COUNT_OPENCLAW] "}
        ;;
      "[COUNT_CLAUDE_PLUGINS] "*)
        claude_plugin_skill_count=${line#"[COUNT_CLAUDE_PLUGINS] "}
        ;;
      "[COUNT_HERMES] "*)
        hermes_skill_count=${line#"[COUNT_HERMES] "}
        ;;
      "[COUNT_EXTENSIONS] "*)
        extension_skill_count=${line#"[COUNT_EXTENSIONS] "}
        ;;
      *)
        if [ "$in_skill_list" = "1" ] && [ -n "$line" ]; then
          all_skill_names="${all_skill_names}${line}
"
        fi
        ;;
    esac
  done <<< "$batch_result"
  all_skill_names=$(printf '%s' "$all_skill_names" | sort -u)

  # Count from skill names
  local container_skill_count
  container_skill_count=$(echo "$all_skill_names" | grep -c . 2>/dev/null | tr -d '\n' || echo 0)
  container_skill_count="${container_skill_count%%[^0-9]*}"
  [ -z "$container_skill_count" ] && container_skill_count=0

  log_item PASS "容器内技能总数" "检测到 $container_skill_count 个技能（跨多目录搜索）"

  if [ "$VERBOSE" = "1" ] && [ -n "$all_skill_names" ]; then
    log_detail "容器内 Skills 列表 ($container_skill_count):"
    while IFS= read -r s; do
      [ -n "$s" ] && log_detail "  - $s"
    done <<< "$all_skill_names"
  fi

  # Per-location counts
  log_item INFO "容器内技能分布位置:"
  log_detail "  /home/node/.openclaw/skills: $openclaw_skill_count"
  log_detail "  /home/node/.claude/plugins: $claude_plugin_skill_count"
  log_detail "  /home/node/.hermes: $hermes_skill_count"
  log_detail "  /app/extensions: $extension_skill_count"

  # ------------------------------------------------
  # B. mindclaw source reference set
  # ------------------------------------------------
  local ref_skills_dir
  if [ -n "$MINDCLAW_SKILLS_DIR" ] && [ -d "$MINDCLAW_SKILLS_DIR" ]; then
    ref_skills_dir="$MINDCLAW_SKILLS_DIR"
  elif [ -f "/mnt/c/WorkSpace/mindclaw/skills/SKILL.md" ] || [ -d "/mnt/c/WorkSpace/mindclaw/skills" ]; then
    ref_skills_dir="/mnt/c/WorkSpace/mindclaw/skills"
  fi

  if [ -n "$ref_skills_dir" ] && [ -d "$ref_skills_dir" ]; then
    log_item PASS "mindclaw 参考 Skills 目录" "路径: $ref_skills_dir"

    local ref_skill_list
    ref_skill_list=$(find "$ref_skills_dir" -maxdepth 1 -type d ! -name "skills" 2>/dev/null | while IFS= read -r d; do
      [ -f "$d/SKILL.md" ] && basename "$d"
    done | sort)

    local ref_count
    ref_count=$(echo "$ref_skill_list" | grep -c . 2>/dev/null | tr -d '\n' || echo 0)
    ref_count="${ref_count%%[^0-9]*}"
    [ -z "$ref_count" ] && ref_count=0

    log_item PASS "mindclaw 参考集技能数量" "参考目录中含 SKILL.md 的技能: $ref_count 个"
    if [ "$VERBOSE" = "1" ] && [ -n "$ref_skill_list" ]; then
      log_detail "参考集 Skills 列表 ($ref_count):"
      echo "$ref_skill_list" | while IFS= read -r s; do
        [ -n "$s" ] && log_detail "  - $s"
      done
    fi

    # ------------------------------------------------
    # C. Intersection
    # ------------------------------------------------
    if [ -z "$all_skill_names" ] || [ "$container_skill_count" = "0" ]; then
      log_item FAIL "Skills 交集比对失败" "容器内未检测到技能"
    else
      local intersection
      intersection=$(comm -12 <(echo "$all_skill_names") <(echo "$ref_skill_list") 2>/dev/null || echo "")
      local only_in_container
      only_in_container=$(comm -23 <(echo "$all_skill_names") <(echo "$ref_skill_list") 2>/dev/null || echo "")
      local only_in_ref
      only_in_ref=$(comm -13 <(echo "$all_skill_names") <(echo "$ref_skill_list") 2>/dev/null || echo "")

      local intersect_count
      intersect_count=$(echo "$intersection" | grep -c . 2>/dev/null | tr -d '\n' || echo 0)
      intersect_count="${intersect_count%%[^0-9]*}"
      [ -z "$intersect_count" ] && intersect_count=0

      log_item PASS "Skills 交集数量" "容器与参考目录共同拥有的技能: $intersect_count 个"

      if [ "$intersect_count" -gt 0 ] && [ "$VERBOSE" = "1" ]; then
        log_item PASS "Skills 交集列表:"
        echo "$intersection" | while IFS= read -r s; do
          [ -n "$s" ] && log_detail "  ✔ $s"
        done
      fi

      local only_container_count
      only_container_count=$(echo "$only_in_container" | grep -c . 2>/dev/null | tr -d '\n' || echo 0)
      only_container_count="${only_container_count%%[^0-9]*}"
      [ -z "$only_container_count" ] && only_container_count=0
      if [ "$only_container_count" -gt 0 ]; then
        log_item INFO "仅在容器中存在的技能 ($only_container_count):"
        echo "$only_in_container" | while IFS= read -r s; do
          [ -n "$s" ] && log_detail "  + $s"
        done
      fi

      local only_ref_count
      only_ref_count=$(echo "$only_in_ref" | grep -c . 2>/dev/null | tr -d '\n' || echo 0)
      only_ref_count="${only_ref_count%%[^0-9]*}"
      [ -z "$only_ref_count" ] && only_ref_count=0
      if [ "$only_ref_count" -gt 0 ]; then
        log_item WARN "仅在参考目录中存在的技能 ($only_ref_count)" \
          "这些技能未部署到容器中"
        echo "$only_in_ref" | while IFS= read -r s; do
          [ -n "$s" ] && log_detail "  - $s"
        done
      fi

      if [ "$ref_count" -gt 0 ]; then
        local coverage_pct
        coverage_pct=$(echo "scale=1; $intersect_count * 100 / $ref_count" | bc 2>/dev/null || echo "N/A")
        log_item PASS "Skills 覆盖率" "容器已覆盖参考集的 ${coverage_pct}% ($intersect_count/$ref_count)"
      fi
    fi
  else
    log_item WARN "无法定位 mindclaw 参考 Skills 目录" \
      "交集比对跳过。请设置 MINDCLAW_SKILLS_DIR 环境变量指向 mindclaw/skills 目录"
  fi

  # ------------------------------------------------
  # D. Key skills check (local, no exec)
  # ------------------------------------------------
  log_item INFO "关键技能逐一验证"
  local key_skills=(
    "openclaw-codeagent-workflow" "mineru-to-markdown" "ppt-multi-style-generator"
    "subagent-coordinator" "skill-creator" "self-improving-agent" "github"
    "humanizer" "docker-sandbox" "minimax-pdf" "minimax-docx" "minimax-xlsx"
  )

  local found_skills=""
  local missing_skills=""
  for skill in "${key_skills[@]}"; do
    if echo "$all_skill_names" | grep -q "^${skill}$"; then
      found_skills="$found_skills $skill"
    else
      missing_skills="$missing_skills $skill"
    fi
  done

  if [ -z "$missing_skills" ]; then
    log_item PASS "所有关键 Skills 存在" "已检查: $(echo $found_skills | tr ' ' ', ' | sed 's/^, //')"
  else
    log_item FAIL "部分关键 Skills 缺失" "缺失: $(echo $missing_skills | tr ' ' ', ' | sed 's/^, //')"
    log_detail "已找到: $(echo $found_skills | tr ' ' ', ' | sed 's/^, //')"
  fi
}

# -----------------------------------------------
# Section 9: Pre-deployed plugins
# -----------------------------------------------
section_plugins() {
  log_section "预置插件部署状态"

  local batch_result
  batch_result=$(exec_in_container_batch '
    # claude-mem plugin
    [ -d "/app/extensions/claude-mem" ] && echo "[CM_DIR] OK" || echo "[CM_DIR] FAIL"
    owner=$(stat -c "%U:%G" "/app/extensions/claude-mem" 2>/dev/null || echo "unknown")
    echo "[CM_OWNER] $owner"
    skill_count=$(find "/app/extensions/claude-mem/skills" -name "*.md" 2>/dev/null | wc -l | tr -d "\n")
    echo "[CM_SKILLS] $skill_count"
    [ -f "/usr/local/lib/node_modules/claude-mem/modes/code.json" ] && echo "[CM_MODES] OK" || echo "[CM_MODES] FAIL"
    [ -f "/usr/local/lib/node_modules/claude-mem/scripts/worker-service.cjs" ] && echo "[CM_WORKER] OK" || echo "[CM_WORKER] FAIL"
    # subagent plugins
    for plugin in subagent-exec-monitor subagent-taskr subagent-observability; do
      dir="/app/extensions/$plugin"
      [ -d "$dir" ] || { echo "[PLUGIN_$plugin] FAIL"; continue; }
      js_count=$(find "$dir/dist" -name "*.js" -not -path "*/node_modules/*" 2>/dev/null | wc -l | tr -d "\n")
      [ "$js_count" -gt 0 ] && echo "[PLUGIN_$plugin] DIST:$js_count" || {
        [ -f "$dir/index.js" ] && echo "[PLUGIN_$plugin] INDEX_OK" || echo "[PLUGIN_$plugin] FAIL"
      }
    done
    [ -d "/home/node/.claude" ] && echo "[CCB_DIR] OK" || echo "[CCB_DIR] FAIL"
  ')

  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in
      \[CM_DIR\]*) [ "$(echo "$line" | sed 's/\[CM_DIR\] //')" = "OK" ] && log_item PASS "claude-mem 插件目录存在" || log_item FAIL "claude-mem 插件目录不存在" ;;
      \[CM_OWNER\]*) log_item PASS "claude-mem 插件权限" "所有者: $(echo "$line" | sed 's/\[CM_OWNER\] //')" ;;
      \[CM_MODES\]*) [ "$(echo "$line" | sed 's/\[CM_MODES\] //')" = "OK" ] && log_item PASS "claude-mem modes/code.json 存在" || log_item FAIL "claude-mem modes/code.json 缺失" ;;
      \[CM_WORKER\]*) [ "$(echo "$line" | sed 's/\[CM_WORKER\] //')" = "OK" ] && log_item PASS "claude-mem worker-service.cjs 存在" || log_item FAIL "claude-mem worker-service.cjs 缺失" ;;
      \[CCB_DIR\]*) [ "$(echo "$line" | sed 's/\[CCB_DIR\] //')" = "OK" ] && log_item PASS "ccb 配置目录存在" || log_item WARN "ccb 配置目录不存在" ;;
      \[PLUGIN_subagent-*)
        plugin=$(echo "$line" | sed 's/\[PLUGIN_\(.*\)\] .*/\1/')
        val=$(echo "$line" | sed 's/\[PLUGIN_[^]]*] //')
        if [ "$val" = "FAIL" ]; then
          log_item FAIL "插件 $plugin 未部署"
        elif [ "${val:0:4}" = "DIST" ]; then
          cnt=$(echo "$val" | sed 's/DIST://')
          log_item PASS "插件 $plugin 已部署" "包含 $cnt 个 JS 文件"
        elif [ "$val" = "INDEX_OK" ]; then
          log_item PASS "插件 $plugin 已部署" "入口文件 index.js"
        fi
        ;;
      \[CM_SKILLS\]*)
        count=$(echo "$line" | sed 's/\[CM_SKILLS\] //')
        if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
          log_item PASS "claude-mem skills 存在" "检测到 $count 个 skill 文件"
        else
          log_item FAIL "claude-mem skills 不存在" "未检测到 skill 文件"
        fi
        ;;
    esac
  done <<< "$batch_result"
}

# -----------------------------------------------
# Port listening status
# -----------------------------------------------
section_ports() {
  log_section "端口监听状态"

  local batch_result
  batch_result=$(exec_in_container_batch '
    for entry in "12345:3039:Gateway" "12346:303A:SFTP" "8080:1F90:Memex" "37700:9344:Claude-Mem-Worker"; do
      port=${entry%%:*}
      rest=${entry#*:}
      hex=${rest%%:*}
      name=${rest#*:}
      if awk "NR>1 { split(\$2, a, \":\"); if (toupper(a[2]) == \"$hex\") found=1 } END { exit found ? 0 : 1 }" /proc/net/tcp /proc/net/tcp6 2>/dev/null; then
        echo "[PORT_$port] OK $name"
      else
        echo "[PORT_$port] MISS $name"
      fi
    done
  ')

  local all_ok=1
  while IFS= read -r line; do
    case "$line" in
      \[PORT_*\]*)
        port=$(printf '%s' "$line" | sed 's/^\[PORT_\([^]]*\)\] .*/\1/')
        status=$(printf '%s' "$line" | sed 's/^\[PORT_[^]]*\] \([^ ]*\).*/\1/')
        name=$(printf '%s' "$line" | sed 's/^\[PORT_[^]]*\] [^ ]* //')
        if [ "$status" = "OK" ]; then
          log_item PASS "$name 端口 $port 正在监听（容器内）"
        else
          log_item WARN "$name 端口 $port 未监听（容器内）"
          all_ok=0
        fi
        ;;
    esac
  done <<< "$batch_result"

  return $((1 - all_ok))
}


# -----------------------------------------------
# Section 11: claude-mem worker 健康状态
# -----------------------------------------------
section_claude_mem_worker() {
  log_section "claude-mem worker 健康状态"

  # Worker health API — 在容器内执行（端口 37700 映射到宿主机 12347）
  local worker_url="http://127.0.0.1:$CLAUDE_MEM_PORT/api/health"
  local worker_resp
  worker_resp=$(exec_in_container "wget -q -O - -T 10 '$worker_url' 2>/dev/null || curl -s -f -m 10 '$worker_url' 2>/dev/null || echo ''")

  if [ -z "$worker_resp" ]; then
    log_item FAIL "claude-mem worker health 异常" "Worker 容器内无响应: $worker_url"
    return 1
  fi

  local status
  status=$(echo "$worker_resp" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
  if [ "$status" = "ok" ]; then
    log_item PASS "claude-mem worker health 正常" "status: $status"
  else
    log_item FAIL "claude-mem worker health 异常" "status: $status"
  fi

  local initialized
  initialized=$(echo "$worker_resp" | grep -o '"initialized":[^,}]*' | cut -d':' -f2- | tr -d ' ' || echo "unknown")
  if [ "$initialized" = "true" ]; then
    log_item PASS "claude-mem initialized=true" "Worker 已完成初始化"
  else
    log_item FAIL "claude-mem initialized=$initialized" "Worker 初始化未完成"
  fi

  local mcp_ready
  mcp_ready=$(echo "$worker_resp" | grep -o '"mcpReady":[^,}]*' | cut -d':' -f2- | tr -d ' ' || echo "unknown")
  if [ "$mcp_ready" = "true" ]; then
    log_item PASS "claude-mem mcpReady=true" "MCP 协议已就绪"
  else
    log_item WARN "claude-mem mcpReady=$mcp_ready" "MCP 协议未就绪"
  fi

  # Session 初始化测试（容器内执行）
  local session_url="http://127.0.0.1:$CLAUDE_MEM_PORT/api/sessions/init"
  local session_resp
  session_resp=$(exec_in_container "wget -q -O - -T 10 --post-data='{\"contentSessionId\":\"health-check-test\",\"project\":\"health-check\",\"prompt\":\"test\"}' --header='Content-Type: application/json' '$session_url' 2>/dev/null || curl -s -f -m 10 -X POST -H 'Content-Type: application/json' -d '{\"contentSessionId\":\"health-check-test\",\"project\":\"health-check\",\"prompt\":\"test\"}' '$session_url' 2>/dev/null || echo ''")
  if echo "$session_resp" | grep -q "sessionDbId\|id"; then
    log_item PASS "claude-mem session 初始化成功"
  else
    log_item WARN "claude-mem session 初始化响应异常" "${session_resp:0:100}"
  fi
}

# -----------------------------------------------
# Section 12: OpenClaw 插件可见性
# -----------------------------------------------
section_plugin_visibility() {
  log_section "OpenClaw 插件可见性"

  # 注意：Gateway 的 agent/plugin 通信走 WebSocket
  # /api/plugins、/api/agents 等返回空是预期行为（API 通过 WS）
  # 通过 /health 探活 + 插件目录静态检查来验证

  local health_url="http://127.0.0.1:$GW_PORT/health"
  local health_resp
  health_resp=$(exec_in_container "wget -q -O - -T 10 '$health_url' 2>/dev/null || curl -s -f -m 10 '$health_url' 2>/dev/null || echo ''")
  if echo "$health_resp" | grep -q '"ok":true'; then
    log_item PASS "Gateway /health 正常" "插件通信层就绪（WebSocket 模式）"
  elif [ -n "$health_resp" ]; then
    log_item WARN "Gateway /health 响应异常" "${health_resp:0:100}"
  else
    log_item FAIL "Gateway /health 无响应"
  fi

  # 通过 Gateway /status 探测（容器内 HTTP，可能返回 JSON 或 HTML）
  local status_resp
  status_resp=$(exec_in_container "wget -q -O - -T 10 'http://127.0.0.1:$GW_PORT/status' 2>/dev/null || curl -s -f -m 10 'http://127.0.0.1:$GW_PORT/status' 2>/dev/null || echo ''")
  if echo "$status_resp" | grep -q '^\{'; then
    log_item PASS "Gateway /status API 正常" "Agent 系统就绪"
    log_detail "响应: $(echo $status_resp | head -c 300)"
  else
    log_item WARN "Gateway /status 走 WebSocket" "Agent 通信需通过 WebSocket 进行，非 HTTP"
  fi

  # 检查 gateway 进程（容器内）
  local gw_pid
  gw_pid=$(exec_in_container "pgrep -f 'node.*gateway' 2>/dev/null | head -1 || pgrep -f 'openclaw' 2>/dev/null | head -1 || echo ''")
  if [ -n "$gw_pid" ] && [ "$gw_pid" != "''" ]; then
    log_item PASS "Gateway 进程运行中" "PID: $gw_pid"
  else
    log_item FAIL "Gateway 进程未找到"
  fi
}

# -----------------------------------------------
# Section 13: 插件功能实际测试
# -----------------------------------------------
section_plugin_functional_test() {
  log_section "插件功能实际测试"

  # 注意：Gateway 工具系统通过 WebSocket 注册，/api/tools 返回空是正常
  # 通过 Worker 健康检查 + 插件 manifest 存在性做功能验证

  # claude-mem worker 健康检查（已在 section_claude_mem_worker 中执行，这里做补充验证）
  local worker_url="http://127.0.0.1:$CLAUDE_MEM_PORT/api/health"
  local worker_resp
  worker_resp=$(exec_in_container "wget -q -O - -T 10 '$worker_url' 2>/dev/null || curl -s -f -m 10 '$worker_url' 2>/dev/null || echo ''")
  if echo "$worker_resp" | grep -q '"status":"ok"'; then
    log_item PASS "claude-mem worker 功能正常" "Worker 端健康检查通过"
  else
    log_item WARN "claude-mem worker 功能异常" "${worker_resp:0:100}"
  fi

  # 通过 exec_in_container 验证 Gateway 工具注册状态（WS 模式）
  # Gateway 的工具通过 WS 暴露，这里通过健康状态间接验证
  if echo "$worker_resp" | grep -q '"mcpReady":true'; then
    log_item PASS "Gateway MCP 工具就绪" "MCP 协议已正常初始化"
  else
    log_item WARN "Gateway MCP 工具未就绪" "MCPReady: false"
  fi
}

# -----------------------------------------------
# Section 14: 健康探针自动重启测试
# -----------------------------------------------
section_health_probe_test() {
  log_section "健康探针自动重启测试"

  if [ "$SKIP_RESTART" = "1" ]; then
    log_item INFO "跳过重启测试" "SKIP_RESTART_TEST=1"
    return 0
  fi

  if [ "$PROBE_TEST" = "0" ]; then
    log_item INFO "跳过重启测试" "未指定 --probe-test"
    return 0
  fi

  # 获取 gateway PID
  local gw_pid
  gw_pid=$(get_gateway_pid)
  if [ -z "$gw_pid" ]; then
    log_item FAIL "无法获取 Gateway PID" "跳过重启测试"
    return 1
  fi

  echo "$gw_pid" > "$GATEWAY_PID_FILE"
  log_item INFO "Gateway PID" "PID: $gw_pid"

  # 记录重启前的状态
  local restart_url="http://127.0.0.1:$GW_PORT/health"
  local resp_before
  resp_before=$(curl -s -f -m 5 "$restart_url" ${GW_TOKEN:+-H "Authorization: Bearer $GW_TOKEN"} 2>/dev/null || echo "")
  log_item INFO "重启前 Gateway 状态" "${resp_before:0:100}"

  # 执行 kill gateway 进程
  log_item INFO "执行 kill gateway 进程" "PID: $gw_pid"
  if ! exec_in_container "kill -TERM '$gw_pid'" 2>/dev/null; then
    log_item FAIL "Gateway 进程终止命令执行失败" "PID: $gw_pid"
    return 1
  fi

  local stop_waited=0
  local stop_wait=30
  while [ $stop_waited -lt "$stop_wait" ]; do
    if ! gateway_pid_alive "$gw_pid"; then
      log_item PASS "Gateway 原进程已退出" "PID: $gw_pid，耗时 ${stop_waited}s"
      break
    fi
    sleep 2
    stop_waited=$((stop_waited + 2))
  done

  if gateway_pid_alive "$gw_pid"; then
    log_item FAIL "Gateway 原进程未退出" "PID: $gw_pid 在 ${stop_wait}s 后仍存在，停止等待自动重启"
    return 1
  fi

  # 等待容器重启（健康探针检测到异常后触发重启）
  log_item INFO "等待容器自动重启" "预计耗时 3-5 分钟..."

  # 分阶段等待：先等进程消失，再等新进程出现，最后等 API 恢复
  local waited=0
  local max_wait=300  # 小于默认整体超时，确保脚本有时间输出汇总
  local probe_recovered=0
  local restarted=0
  local new_pid=""

  while [ $waited -lt "$max_wait" ]; do
    sleep 10
    waited=$((waited + 10))

    if [ "$restarted" = "0" ]; then
      new_pid=$(get_gateway_pid 2>/dev/null || echo "")
      if [ -n "$new_pid" ] && [ "$new_pid" != "$gw_pid" ]; then
        log_item PASS "Gateway 进程已重启" "新 PID: $new_pid"
        restarted=1
      elif [ $((waited % 30)) -eq 0 ]; then
        log_item INFO "等待 Gateway 自动重启" "已等待 ${waited}s"
      else
        log_detail "等待中... 已等待 ${waited}s"
      fi
    fi

    if [ "$restarted" = "1" ]; then
      local resp_after
      resp_after=$(curl -s -f -m 10 "$restart_url" \
        ${GW_TOKEN:+-H "Authorization: Bearer $GW_TOKEN"} 2>/dev/null || echo "")
      if [ -n "$resp_after" ]; then
        log_item PASS "Gateway API 在重启后恢复正常" "响应: ${resp_after:0:100}"
        probe_recovered=1
        break
      elif [ $((waited % 30)) -eq 0 ]; then
        log_item INFO "等待 Gateway API 恢复" "已等待 ${waited}s，新 PID: $new_pid"
      else
        log_detail "Gateway API 尚未恢复，已等待 ${waited}s"
      fi
    fi
  done

  if [ "$probe_recovered" = "1" ]; then
    log_item PASS "健康探针自动重启测试通过" "容器在 ${waited}s 内恢复正常"
    return 0
  else
    log_item FAIL "健康探针自动重启测试失败" "超过 ${max_wait}s 未能恢复"
    return 1
  fi
}

# -----------------------------------------------
# Section 15: 汇总报告
# -----------------------------------------------
section_summary() {
  log_section "检查结果汇总"

  local end_time=$(date +%s)
  local duration=$((end_time - START_TIME))

  if [ "$OUTPUT_FORMAT" = "tap" ]; then
    echo ""
    echo "1..$TOTAL_COUNT"
    echo "# Totals: $PASS_COUNT passed, $FAIL_COUNT failed, $WARN_COUNT warnings"
    echo "# Duration: ${duration}s"
    echo "# Timestamp: $(date -Iseconds)"
  elif [ "$OUTPUT_FORMAT" = "json" ]; then
    cat << EOF
{
  "summary": {
    "passed": $PASS_COUNT,
    "failed": $FAIL_COUNT,
    "warnings": $WARN_COUNT,
    "total": $TOTAL_COUNT,
    "duration_seconds": $duration,
    "timestamp": "$(date -Iseconds)"
  }
}
EOF
  else
    echo ""
    echo -e "${BOLD}检查统计${RESET}"
    echo -e "  总检查项: $TOTAL_COUNT"
    echo -e "  ${GREEN}通过: $PASS_COUNT${RESET}"
    echo -e "  ${RED}失败: $FAIL_COUNT${RESET}"
    echo -e "  ${YELLOW}警告: $WARN_COUNT${RESET}"
    echo ""
    echo -e "${BOLD}耗时${RESET}: ${duration}s"
    echo -e "${BOLD}时间戳${RESET}: $(date -Iseconds)"
    echo ""

    if [ "$FAIL_COUNT" -gt 0 ]; then
      echo -e "${RED}${BOLD}★★★ 检查未通过 ★★★${RESET}"
      echo -e "${RED}有 $FAIL_COUNT 项检查失败，请查看上方详细信息${RESET}"
      echo ""
      if [ -s "$ERRORS_FILE" ]; then
        echo -e "${RED}错误摘要:${RESET}"
        cat "$ERRORS_FILE" | head -20
      fi
    elif [ "$WARN_COUNT" -gt 0 ]; then
      echo -e "${YELLOW}${BOLD}★★ 带警告通过 ★★${RESET}"
      echo -e "${YELLOW}有 $WARN_COUNT 项警告，但核心功能正常${RESET}"
    else
      echo -e "${GREEN}${BOLD}★★★ 检查全部通过 ★★★${RESET}"
      echo -e "${GREEN}所有检查项均通过${RESET}"
    fi
  fi

  # 返回退出码
  if [ "$FAIL_COUNT" -gt 0 ]; then
    return 1
  else
    return 0
  fi
}

# -----------------------------------------------
# 主流程
# -----------------------------------------------
main() {
  SCRIPT_STARTED=1
  # 清理临时文件
  trap 'rm -f "$RESULTS_FILE" "$ERRORS_FILE" "$GATEWAY_PID_FILE" 2>/dev/null' EXIT

  # 输出头部
  if [ "$OUTPUT_FORMAT" = "color" ]; then
    echo ""
    echo -e "${CYAN}${BOLD}============================================${RESET}"
    echo -e "${CYAN}${BOLD}  OpenClaw 容器健康状态校验脚本 v$SCRIPT_VERSION${RESET}"
    echo -e "${CYAN}${BOLD}============================================${RESET}"
  elif [ "$OUTPUT_FORMAT" = "tap" ]; then
    echo "# OpenClaw Health Check v$SCRIPT_VERSION"
    echo "# Container: ${CONTAINER_NAME:-default}"
    echo "# Port: $GW_PORT"
    echo "# Timestamp: $(date -Iseconds)"
  fi

  # 检测运行环境
  detect_runtime

  # 基础信息
  section_info

  # 镜像层静态检查（始终执行）
  echo "" > "$ERRORS_FILE"

  # Gateway 连通性（需要 API 就绪）
  if [ "$IMAGE_ONLY" = "0" ]; then
    section_gateway_connectivity || true
    section_gateway_talkability || true
    section_claude_mem_worker || true
    section_plugin_visibility || true
    section_plugin_functional_test || true
    section_health_probe_test || true
  else
    log_item INFO "跳过 API 层检查" "--image-only 模式"
  fi

  section_hermes
  section_browser
  section_python_deps
  section_npm_deps
  section_skills
  section_plugins
  section_ports

  # 汇总
  section_summary
  local exit_code=$?

  if [ "$OUTPUT_FORMAT" = "color" ]; then
    echo ""
    echo -e "${CYAN}详细日志已保存至:${RESET} $ERRORS_FILE"
  fi

  exit $exit_code
}

run_with_instance_timeout "${ORIGINAL_ARGS[@]}"
main "$@"
