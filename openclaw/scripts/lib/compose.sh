#!/bin/bash
# =============================================================================
# docker-compose.yml 生成与集群操作
# =============================================================================

# 生成 docker-compose.yml
generate_compose_file() {
    ensure_envsubst

    local compose_file="$CONFIG_BASE/docker-compose.yml"

    # 写入文件头
    cp "$TEMPLATES_DIR/docker-compose.header.tpl" "$compose_file"

    # 为每个实例追加 service 块
    for i in $(seq "$START_INDEX" "$((START_INDEX + COUNT - 1))"); do
        export INSTANCE_NUM="$i"
        export INSTANCE_PREFIX
        export IMAGE
        export GW_PORT=$((BASE_PORT + (i - 1) * 4))
        export SFTP_PORT=$((GW_PORT + 1))
        export MDNS_PORT_HOST=$((GW_PORT + 2))
        export MEMEX_PORT=$((GW_PORT + 3))
        export MDNS_PORT
        # 读取实例的 Gateway Token（per-instance 模式下从文件读取）
        if [ "$OPENCLAW_TOKEN_PER_INSTANCE" = "true" ]; then
            local token_file="$CONFIG_BASE/instance-$i/.gateway_token"
            if [ -f "$token_file" ]; then
                OPENCLAW_TOKEN=$(cat "$token_file")
            else
                OPENCLAW_TOKEN=$(openssl rand -hex 16)
                log_warn "未找到实例 $i 的 Token 文件，重新生成"
            fi
        fi
        export OPENCLAW_TOKEN

        # 计算配置目录绝对路径
        # Windows Git Bash 的 pwd 对中文路径可能输出 GBK 编码，
        # 而 docker-compose.yml 要求 UTF-8，所以用 readlink -f 或 realpath 代替
        local config_dir="$CONFIG_BASE/instance-$i"
        export CONFIG_DIR_ABS
        if command -v realpath &> /dev/null; then
            CONFIG_DIR_ABS="$(realpath "$config_dir" 2>/dev/null || echo "$config_dir")"
        elif command -v readlink &> /dev/null; then
            CONFIG_DIR_ABS="$(readlink -f "$config_dir" 2>/dev/null || echo "$config_dir")"
        else
            CONFIG_DIR_ABS="$(cd "$config_dir" 2>/dev/null && pwd || echo "$config_dir")"
        fi

        export INSTALL_HERMES
        export GUARDIAN_PORT
        export MEMEX_KB_VOLUME
        envsubst '${INSTANCE_PREFIX} ${INSTANCE_NUM} ${IMAGE} ${GW_PORT} ${SFTP_PORT} ${MDNS_PORT_HOST} ${MDNS_PORT} ${CONFIG_DIR_ABS} ${OPENCLAW_TOKEN} ${INSTALL_HERMES} ${GUARDIAN_PORT} ${MEMEX_PORT} ${MEMEX_KB_VOLUME}' \
            < "$TEMPLATES_DIR/docker-compose.service.tpl" >> "$compose_file"
        # macOS (BSD) sed requires '' after -i, GNU sed doesn't
        if sed --version &> /dev/null; then
            SED_I=(sed -i)
        else
            SED_I=(sed -i '')
        fi
        # 不挂载 skills 时移除 skills volume 行
        if [ "$MOUNT_SKILLS" != "true" ]; then
            "${SED_I[@]}" "/openclaw-skills-${INSTANCE_NUM}/d" "$compose_file"
        fi
        # 未指定 guardian port 时移除相关行
        if [ -z "$GUARDIAN_PORT" ]; then
            "${SED_I[@]}" "/GUARDIAN_PORT/d" "$compose_file"
            "${SED_I[@]}" '/"":""$/d' "$compose_file"
            "${SED_I[@]}" '/- ":"$/d' "$compose_file"
        fi
        done

    # 修复编码：Windows Git Bash 下 envsubst 可能输出 GBK，docker compose 要求 UTF-8
    if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
        if command -v iconv &> /dev/null; then
            iconv -f GBK -t UTF-8 "$compose_file" > "${compose_file}.tmp" 2>/dev/null && \
                mv "${compose_file}.tmp" "$compose_file" || \
                rm -f "${compose_file}.tmp"
        fi
    fi


    # 追加 volumes 声明块（--skills 时需要，hermes 已改为 bind mount 不需要）
    if [ "$MOUNT_SKILLS" = "true" ]; then
        {
            echo ''
            echo 'volumes:'
            for i in $(seq "$START_INDEX" "$((START_INDEX + COUNT - 1))"); do
                echo "  openclaw-skills-${i}:"
            done
        } >> "$compose_file"
    fi

    {
    echo ''
    echo 'networks:'
    echo '  default:'
    echo '    driver: bridge'
    echo '    ipam:'
    echo '      config:'
    echo "        - subnet: ${SUBNET}"
    } >> "$compose_file"

    log_ok "docker-compose.yml 生成完成"
}

# 显示操作提示
show_compose_hints() {
    echo ""
    echo "docker compose 命令参考"
    echo "=============================================="
    echo "启动集群:"
    echo "  ${DOCKER_COMPOSE_CMD} -f $CONFIG_BASE/docker-compose.yml up -d"
    echo ""
    echo "查看日志:"
    echo "  ${DOCKER_COMPOSE_CMD} -f $CONFIG_BASE/docker-compose.yml logs -f"
    echo ""
    echo "停止集群:"
    echo "  ${DOCKER_COMPOSE_CMD} -f $CONFIG_BASE/docker-compose.yml down"
    echo "=============================================="
}

# 启动集群
cluster_up() {
    detect_docker_compose
    show_compose_hints

    echo ""
    log_info "开始启动集群..."

    # 先关闭已有集群（--no-down 扩容模式下跳过）
    if [ "$SKIP_DOWN" = "true" ]; then
        log_info "扩容模式：跳过 down，已有实例不受影响"
    else
        log_info "执行 ${DOCKER_COMPOSE_CMD} -f $CONFIG_BASE/docker-compose.yml down 关闭已有集群并准备更新"
        ${DOCKER_COMPOSE_CMD} -f "$CONFIG_BASE/docker-compose.yml" down 2>/dev/null || true
        sleep 5
    fi

    # 启动
    log_info "执行 ${DOCKER_COMPOSE_CMD} -f $CONFIG_BASE/docker-compose.yml up -d 集群启动中"
    ${DOCKER_COMPOSE_CMD} -f "$CONFIG_BASE/docker-compose.yml" up -d

    # 记录当前版本
    echo "$IMAGE" > "$CONFIG_BASE/.current_version"

    # 同步 hermes 配置（hermes 已在镜像中，始终同步配置）
    for i in $(seq "$START_INDEX" "$((START_INDEX + COUNT - 1))"); do
        local cname="${INSTANCE_PREFIX}-${i}"
        local hermes_cfg="$CONFIG_BASE/instance-$i/.hermes"
        if [ -d "$hermes_cfg" ]; then
            docker cp "$hermes_cfg/.env" "$cname:/home/node/.hermes/.env" 2>/dev/null || true
            docker exec "$cname" chown -R node:node /home/node/.hermes 2>/dev/null || true
        fi
    done

    # 同步 skills 到 named volume（volume 初始为空，需要从宿主机拷入）
    if [ "$MOUNT_SKILLS" = "true" ] && [ -d "skills" ]; then
        local skill_count=0
        local skill_list=""
        for i in $(seq "$START_INDEX" "$((START_INDEX + COUNT - 1))"); do
            local cname="${INSTANCE_PREFIX}-${i}"
            skill_count=0
            skill_list=""
            for skill_dir in skills/*/; do
                [ -d "$skill_dir" ] || continue
                local skill_name
                skill_name="$(basename "$skill_dir")"
                docker exec "$cname" mkdir -p "/home/node/.claude/skills/${skill_name}" 2>/dev/null || true
                docker cp "${skill_dir}/." "$cname:/home/node/.claude/skills/${skill_name}/" 2>/dev/null || true
                skill_count=$((skill_count + 1))
                skill_list="${skill_list} ${skill_name}"
            done
            docker exec "$cname" bash -c 'find /home/node/.claude/skills -name "*.sh" -exec chmod +x {} + 2>/dev/null; find /home/node/.claude/skills -name "*.py" -exec chmod +x {} + 2>/dev/null' || true
            log_info "同步 ${skill_count} 个 skills 到 ${cname}:${skill_list}"
            # 反向拷贝：把 volume 中的完整 skills 快照回宿主机，方便查看
            mkdir -p "$CONFIG_BASE/instance-$i/skills"
            docker cp "$cname:/home/node/.claude/skills/." "$CONFIG_BASE/instance-$i/skills/" 2>/dev/null || true
        done
        log_ok "skills 同步完成（共 ${skill_count} 个），宿主机快照: $CONFIG_BASE/instance-*/skills/"
    fi

    log_ok "集群启动完成"
}

# 停止集群
cluster_down() {
    detect_docker_compose
    log_info "正在停止集群..."
    ${DOCKER_COMPOSE_CMD} -f "$CONFIG_BASE/docker-compose.yml" down 2>/dev/null || true
    log_ok "集群已停止"
}
