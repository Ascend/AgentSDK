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

    # ── skills 合并目录（始终从镜像提取 skills-shared）─────────────
    # 镜像 skills-shared 与宿主机 skills 取并集，通过 bind mount 挂载
    local skills_merged_dir="$CONFIG_BASE/skills-merged"
    # 确保能删除旧目录（WSL/NTFS 下容器写入的文件可能属于其他用户）
    [ -d "$skills_merged_dir" ] && find "$skills_merged_dir" -type d -exec chmod u+w {} + 2>/dev/null || true
    rm -rf "$skills_merged_dir" 2>/dev/null || true
    mkdir -p "$skills_merged_dir"

    # 1) 从镜像中提取 skills-shared
    local image_skills_dir="$CONFIG_BASE/.image-skills-tmp"
    rm -rf "$image_skills_dir" 2>/dev/null || true
    mkdir -p "$image_skills_dir"

    # 通过 tar 管道提取，避免 docker cp 在 WSL/Windows/Linux 间路径兼容问题
    docker rm -f skills-extract-tmp 2>/dev/null || true
    local _create_output
    _create_output=$(docker create --name skills-extract-tmp "$IMAGE" 2>&1) || {
        log_warn "无法从镜像创建临时容器（镜像: $IMAGE, 错误: $_create_output）"
    }
    if [ -n "$_create_output" ] && docker ps -a --filter name=skills-extract-tmp --format '{{.ID}}' 2>/dev/null | grep -q .; then
        docker cp skills-extract-tmp:/home/node/.openclaw/skills-shared/. - 2>/dev/null | tar xf - -C "$image_skills_dir/" 2>/dev/null || true
        docker rm -f skills-extract-tmp 2>/dev/null || true
    fi

    # 2) 先把镜像中的 skills 复制到合并目录
    if [ -d "$image_skills_dir" ] && [ "$(ls -A "$image_skills_dir" 2>/dev/null)" ]; then
        cp -r "$image_skills_dir/." "$skills_merged_dir/"
        log_info "从镜像中提取 skills 到合并目录"
    else
        log_warn "镜像中未找到 skills-shared 目录，skills 将为空"
    fi
    rm -rf "$image_skills_dir" 2>/dev/null || true

    # 3) --skills 时合并宿主机技能（宿主机优先覆盖同名技能）
    if [ "$MOUNT_SKILLS" = "true" ]; then
        local host_skills_dir
        host_skills_dir="$(cd "$SCRIPT_DIR/../../openclaw/skills" 2>/dev/null && pwd || echo "")"
        if [ -n "$host_skills_dir" ] && [ -d "$host_skills_dir" ]; then
            for skill_dir in "$host_skills_dir"/*/; do
                [ -d "$skill_dir" ] || continue
                local skill_name
                skill_name="$(basename "$skill_dir")"
                rm -rf "$skills_merged_dir/$skill_name"
                cp -r "$skill_dir" "$skills_merged_dir/$skill_name/"
                log_info "合并宿主机技能: $skill_name"
            done
        else
            log_warn "未找到宿主机 skills 目录，仅使用镜像内置技能"
        fi
    fi

    # 4) 设置可执行权限
    find "$skills_merged_dir" -name "*.sh" -exec chmod +x {} + 2>/dev/null || true
    find "$skills_merged_dir" -name "*.py" -exec chmod +x {} + 2>/dev/null || true

    local skill_count
    skill_count=$(ls -1 "$skills_merged_dir/" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$MOUNT_SKILLS" = "true" ]; then
        log_ok "skills 合并完成（共 ${skill_count} 项技能），目录: $skills_merged_dir"
    else
        log_info "skills 提取完成（共 ${skill_count} 项镜像内置技能），目录: $skills_merged_dir"
    fi

    # 为每个实例追加 service 块
    for i in $(seq "$START_INDEX" "$((START_INDEX + COUNT - 1))"); do
        export INSTANCE_NUM="$i"
        export INSTANCE_PREFIX
        export IMAGE
        export GW_PORT=$((BASE_PORT + (i - 1) * 4))
        export SFTP_PORT=$((GW_PORT + 1))
        export MDNS_PORT_HOST=$((GW_PORT + 2))
        export MDNS_PORT
        # 读取 per-instance Gateway Token
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

        # 计算配置目录绝对路径（Windows Git Bash 中文路径可能输出 GBK）
        local config_dir="$CONFIG_BASE/instance-$i"
        export CONFIG_DIR_ABS
        if command -v realpath &> /dev/null; then
            CONFIG_DIR_ABS="$(realpath "$config_dir" 2>/dev/null || echo "$config_dir")"
        elif command -v readlink &> /dev/null; then
            CONFIG_DIR_ABS="$(readlink -f "$config_dir" 2>/dev/null || echo "$config_dir")"
        else
            CONFIG_DIR_ABS="$(cd "$config_dir" 2>/dev/null && pwd || echo "$config_dir")"
        fi

        # skills 合并目录绝对路径（始终用于 bind mount）
        export SKILLS_MERGED_DIR_ABS
        if command -v realpath &> /dev/null; then
            SKILLS_MERGED_DIR_ABS="$(realpath "$skills_merged_dir" 2>/dev/null || echo "$skills_merged_dir")"
        elif command -v readlink &> /dev/null; then
            SKILLS_MERGED_DIR_ABS="$(readlink -f "$skills_merged_dir" 2>/dev/null || echo "$skills_merged_dir")"
        else
            SKILLS_MERGED_DIR_ABS="$(cd "$skills_merged_dir" 2>/dev/null && pwd || echo "$skills_merged_dir")"
        fi

        export INSTALL_HERMES
        export GUARDIAN_PORT
        # envsubst 变量列表：始终包含 SKILLS_MERGED_DIR_ABS（skills 现已始终提取）
        local envsubst_vars='${INSTANCE_PREFIX} ${INSTANCE_NUM} ${IMAGE} ${GW_PORT} ${SFTP_PORT} ${MDNS_PORT_HOST} ${MDNS_PORT} ${CONFIG_DIR_ABS} ${OPENCLAW_TOKEN} ${INSTALL_HERMES} ${GUARDIAN_PORT} ${SKILLS_MERGED_DIR_ABS}'
        envsubst "$envsubst_vars" \
            < "$TEMPLATES_DIR/docker-compose.service.tpl" >> "$compose_file"
        # macOS (BSD) sed requires '' after -i, GNU sed doesn't
        if sed --version &> /dev/null; then
            SED_I=(sed -i)
        else
            SED_I=(sed -i '')
        fi
        # 未指定 guardian port 时移除相关行
        if [ -z "$GUARDIAN_PORT" ]; then
            "${SED_I[@]}" "/GUARDIAN_PORT/d" "$compose_file"
            "${SED_I[@]}" '/"":""$/d' "$compose_file"
            "${SED_I[@]}" '/- ":"$/d' "$compose_file"
        fi
        # 禁用沙箱时移除 docker socket 挂载
        if [ "$SANDBOX_ENABLED" != "true" ]; then
            "${SED_I[@]}" "/docker.sock/d" "$compose_file"
        fi
    done

    # Windows Git Bash envsubst 可能输出 GBK，docker compose 要求 UTF-8
    if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
        if command -v iconv &> /dev/null; then
            iconv -f GBK -t UTF-8 "$compose_file" > "${compose_file}.tmp" 2>/dev/null && \
                mv "${compose_file}.tmp" "$compose_file" || \
                rm -f "${compose_file}.tmp"
        fi
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

    # ── hermes 配置检查（bind mount，无需 docker cp 同步）─────────
    for i in $(seq "$START_INDEX" "$((START_INDEX + COUNT - 1))"); do
        local hermes_cfg_dir="$CONFIG_BASE/instance-$i/.hermes"
        if [ -d "$hermes_cfg_dir" ]; then
            local cfg_ok=true
            [ -f "$hermes_cfg_dir/config.yaml" ] || { log_warn "实例 $i hermes config.yaml 缺失"; cfg_ok=false; }
            [ -f "$hermes_cfg_dir/.env" ] || { log_warn "实例 $i hermes .env 缺失"; cfg_ok=false; }
            if [ "$cfg_ok" = "true" ]; then
                log_info "实例 $i hermes 配置已就绪（bind mount: $hermes_cfg_dir）"
            fi
        else
            log_warn "实例 $i hermes 配置目录不存在: $hermes_cfg_dir"
        fi
    done

    log_ok "集群启动完成"
}

# 停止集群
cluster_down() {
    detect_docker_compose
    log_info "正在停止集群..."
    ${DOCKER_COMPOSE_CMD} -f "$CONFIG_BASE/docker-compose.yml" down 2>/dev/null || true
    log_ok "集群已停止"
}
