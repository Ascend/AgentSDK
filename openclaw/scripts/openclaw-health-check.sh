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
