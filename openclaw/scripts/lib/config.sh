#!/bin/bash
# =============================================================================
# 配置生成引擎：envsubst 模板渲染 + 实例配置生成
# =============================================================================

# 模板目录
TEMPLATES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../templates" && pwd)"

# 渲染模板到目标文件
# 用法: render_template <模板文件> <输出文件> [变量列表]
# 如果输出文件已存在则跳过
render_template() {
    local tpl_file="$1"
    local out_file="$2"
    local var_list="${3:-}"

    if [ -f "$out_file" ]; then
        log_warn "跳过: $out_file (已存在)"
        return 0
    fi

    mkdir -p "$(dirname "$out_file")"

    if [ -n "$var_list" ]; then
        envsubst "$var_list" < "$tpl_file" > "$out_file"
    else
        # 不带变量列表时，使用 Python 进行可靠的变量替换
        local python_script="/tmp/render_template_$$.py"
        cat > "$python_script" << 'PYEOF'
import os
import re

tpl_file = os.environ.get('TPL_FILE', '')
out_file = os.environ.get('OUT_FILE', '')

with open(tpl_file, 'r') as f:
    content = f.read()

def replacer(match):
    var_expr = match.group(1)
    if ':-' in var_expr:
        var_name, default = var_expr.split(':-', 1)
        return os.environ.get(var_name.strip(), default.strip())
    else:
        return os.environ.get(var_expr.strip(), match.group(0))

result = re.sub(r'\$\{([^}]+)\}', replacer, content)
with open(out_file, 'w') as f:
    f.write(result)
PYEOF
        TPL_FILE="$tpl_file" OUT_FILE="$out_file" python3 "$python_script"
        rm -f "$python_script"
    fi
}

# 复制模板到目标文件（无变量替换）
# 用法: copy_template <模板文件> <输出文件>
copy_template() {
    local tpl_file="$1"
    local out_file="$2"

    if [ -f "$out_file" ]; then
        log_warn "跳过: $out_file (已存在)"
        return 0
    fi

    mkdir -p "$(dirname "$out_file")"
    cp "$tpl_file" "$out_file"
}

# 为单个实例生成全部配置
generate_instance_config() {
    local i=$1
    local config_dir="$CONFIG_BASE/instance-$i"

    # 计算端口
    export GW_PORT=$((BASE_PORT + (i - 1) * 4))
    export SFTP_PORT=$((GW_PORT + 1))
    export MDNS_PORT_HOST=$((GW_PORT + 2))
    export MEMEX_PORT=$((GW_PORT + 3))

    # 生成随机密码
    local sftp_password
    sftp_password=$(generate_random_password)

    # Gateway Token：若未指定全局 Token，则为本实例生成独立随机 Token
    if [ "$OPENCLAW_TOKEN_PER_INSTANCE" = "true" ]; then
        if [ -f "$config_dir/.gateway_token" ]; then
            # 重新部署时保留现有token
            OPENCLAW_TOKEN=$(cat "$config_dir/.gateway_token")
            log_info "实例 $i Gateway Token: (保留现有)"
        else
            # 首次部署生成新token
            OPENCLAW_TOKEN=$(openssl rand -hex 16)
            log_info "实例 $i Gateway Token: $OPENCLAW_TOKEN"
        fi
    fi

    # 时间戳
    export TIMESTAMP
    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%S.000Z 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)

    # 导出所有需要的变量
    export OPENCLAW_TOKEN MODEL_NAME MODEL_PROVIDER INFER_URL MDNS_PORT API_KEY
    export CLAUDE_MEM_WORKER_PORT CLAUDE_MEM_PROJECT
    export SUBAGENT_COORDINATOR_DIR SANDBOX_ENABLED
    # 非 local 供应商时，Claude 需要 /anthropic 后缀
    if [ "$MODEL_PROVIDER" != "local" ]; then
        export ANTHROPIC_SUFFIX="/anthropic"
    else
        export ANTHROPIC_SUFFIX=""
    fi

    # 设置沙箱模式
    if [ "$SANDBOX_ENABLED" = "true" ]; then
        export SANDBOX_MODE="all"
    else
        export SANDBOX_MODE="off"
    fi

    # 创建目录
    mkdir -p "$config_dir/agents/main/agent"
    mkdir -p "$config_dir/data"
    mkdir -p "$config_dir/ssh"

    # 渲染 openclaw.json
    render_template \
        "$TEMPLATES_DIR/openclaw.json.tpl" \
        "$config_dir/openclaw.json"

    # 渲染 models.json
    render_template \
        "$TEMPLATES_DIR/models.json.tpl" \
        "$config_dir/agents/main/agent/models.json" \
        '${INFER_URL} ${MODEL_NAME} ${MODEL_PROVIDER}'

    # 渲染 sshd_config
    render_template \
        "$TEMPLATES_DIR/ssh/sshd_config.tpl" \
        "$config_dir/ssh/sshd_config" \
        '${SFTP_PORT}'

    # 生成 Claude Code 配置
    mkdir -p "$config_dir/.claude"
    render_template         "$TEMPLATES_DIR/claude-settings.json.tpl"         "$config_dir/.claude/settings.json"         '${INFER_URL} ${MODEL_NAME} ${API_KEY} ${ANTHROPIC_SUFFIX}'
    # 生成 Hermes Agent 配置（hermes 已在镜像中，始终生成）
    mkdir -p "$config_dir/.hermes"
    render_template \
        "$TEMPLATES_DIR/hermes-config.yaml.tpl" \
        "$config_dir/.hermes/config.yaml" \
        '${INFER_URL} ${MODEL_NAME} ${API_KEY}'
    # 始终更新 .env 文件以确保 API_KEY 正确（如果指定了的话）
    if [ -n "$API_KEY" ]; then
        echo "OPENAI_API_KEY=${API_KEY}" > "$config_dir/.hermes/.env"
        echo "ANTHROPIC_API_KEY=${API_KEY}" >> "$config_dir/.hermes/.env"
    fi

    # 复制静态模板（无变量替换）
    copy_template "$TEMPLATES_DIR/ssh/passwd.tpl" "$config_dir/ssh/passwd"
    copy_template "$TEMPLATES_DIR/ssh/start_sshd.sh.tpl" "$config_dir/ssh/start_sshd.sh"
    copy_template "$TEMPLATES_DIR/health_monitor.sh.tpl" "$config_dir/health_monitor.sh"

    # 修复 Windows CRLF 换行符（确保 Linux 容器能执行）
    sed -i 's/\r$//' "$config_dir/ssh/start_sshd.sh" 2>/dev/null || true
    sed -i 's/\r$//' "$config_dir/health_monitor.sh" 2>/dev/null || true

    # 设置可执行权限
    chmod +x "$config_dir/ssh/start_sshd.sh" 2>/dev/null || true
    chmod +x "$config_dir/health_monitor.sh" 2>/dev/null || true

    # 生成 sftp 密码文件
    if [ ! -f "$config_dir/ssh/sftp_password" ]; then
        echo "$sftp_password" > "$config_dir/ssh/sftp_password"
    else
        log_warn "跳过: $config_dir/ssh/sftp_password (已存在)"
    fi

    # 保存 Gateway Token（供 compose 生成时读取）
    echo "$OPENCLAW_TOKEN" > "$config_dir/.gateway_token"

    # =============================================================================
    # claude-mem 插件补丁：为解决 workspaceDir 不可用问题，需要替换插件入口文件
    # 详见: https://github.com/thedotmack/claude-mem/issues/XXX
    # 补丁版本: v2026.4.11
    # 原始文件: /app/extensions/claude-mem/dist/index.js (在 Docker 镜像内)
    # 挂载后文件: /app/extensions/claude-mem/dist/index.js (覆盖镜像内文件)
    # =============================================================================
    CLAUDE_MEM_PATCH_SRC="$SCRIPT_DIR/patches/claude-mem/claude-mem-index.js.v2026.4.11.patched"
    if [ -f "$CLAUDE_MEM_PATCH_SRC" ]; then
        mkdir -p "$config_dir/plugins/claude-mem/dist"
        cp "$CLAUDE_MEM_PATCH_SRC" "$config_dir/plugins/claude-mem/dist/index.js"
        # 注意：权限会在后续 chown 阶段统一处理
        log_info "已应用 claude-mem 插件补丁 (workspaceDir fix)"
    else
        log_warn "未找到 claude-mem 插件补丁文件: $CLAUDE_MEM_PATCH_SRC"
    fi

    # 修复文件所有权
    chown 1000:1000 -R "$config_dir" 2>/dev/null || true

    local display_pass
    display_pass=$(cat "$config_dir/ssh/sftp_password" 2>/dev/null)
    log_ok "已配置实例 $i: Gateway $GW_PORT, SFTP $SFTP_PORT, SFTP密码: $display_pass"
}

# 生成所有实例的配置
generate_all_configs() {
    ensure_envsubst
    mkdir -p "$CONFIG_BASE"

    log_info "准备生成 $COUNT 个实例的配置..."
    log_info "基础端口: $BASE_PORT (Gateway: port, SFTP: port+1)"

    for i in $(seq "$START_INDEX" "$((START_INDEX + COUNT - 1))"); do
        generate_instance_config "$i"
    done

    log_ok "全部 $COUNT 个实例配置生成完成（编号: ${START_INDEX}-$((START_INDEX + COUNT - 1))）"
}
