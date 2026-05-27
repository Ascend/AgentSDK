#!/bin/bash
# =============================================================================
# 版本管理：镜像列表、切换、回滚
# =============================================================================

# 列出本地所有 OpenClaw 相关镜像
version_list() {
    log_info "本地 OpenClaw 镜像列表:"
    echo ""

    local current=""
    if [ -f "$CONFIG_BASE/.current_version" ]; then
        current=$(cat "$CONFIG_BASE/.current_version")
    fi

    # 获取镜像列表
    local images
    images=$(docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" \
        | grep -iE "openclaw|mindclaw" 2>/dev/null || true)

    if [ -z "$images" ]; then
        log_warn "未找到 OpenClaw 相关镜像"
        log_info "加载镜像: docker load -i <镜像文件.tar>"
        return
    fi

    printf "  %-55s %-10s %s\n" "IMAGE" "SIZE" "CREATED"
    printf "  %-55s %-10s %s\n" "-----" "----" "-------"

    while IFS=$'\t' read -r image size created; do
        local marker="  "
        if [ "$image" = "$current" ]; then
            marker="* "
        fi
        printf "  %s%-53s %-10s %s\n" "$marker" "$image" "$size" "$created"
    done <<< "$images"

    echo ""
    if [ -n "$current" ]; then
        log_info "当前使用: $current（* 标记）"
    fi
}

# 显示当前使用的镜像版本
version_current() {
    if [ ! -f "$CONFIG_BASE/.current_version" ]; then
        log_warn "未找到版本记录（尚未部署过？）"
        log_info "配置目录: $CONFIG_BASE"
        return 1
    fi

    local current
    current=$(cat "$CONFIG_BASE/.current_version")
    log_info "当前版本: $current"
}

# 切换到指定镜像版本
version_switch() {
    local target="$1"

    if [ -z "$target" ]; then
        log_error "请指定目标镜像，例如: deploy.sh version switch openclaw:2026.4.1"
        return 1
    fi

    # 检查镜像是否存在
    if ! docker image inspect "$target" &> /dev/null; then
        log_error "镜像不存在: $target"
        log_info "使用 'docker load -i <文件.tar>' 加载镜像"
        log_info "使用 'deploy.sh version list' 查看可用镜像"
        return 1
    fi

    # 保存当前版本为 previous
    if [ -f "$CONFIG_BASE/.current_version" ]; then
        cp "$CONFIG_BASE/.current_version" "$CONFIG_BASE/.previous_version"
    fi

    # 更新镜像变量
    IMAGE="$target"
    echo "$IMAGE" > "$CONFIG_BASE/.current_version"

    # 重新生成 compose 文件并重启
    log_info "切换镜像到: $target"
    generate_compose_file
    detect_docker_compose

    log_info "重启集群..."
    ${DOCKER_COMPOSE_CMD} down 2>/dev/null || true
    sleep 3
    ${DOCKER_COMPOSE_CMD} up -d

    log_ok "版本切换完成: $target"
}

# 回滚到上一个版本
version_rollback() {
    if [ ! -f "$CONFIG_BASE/.previous_version" ]; then
        log_error "没有可回滚的版本（未找到 .previous_version）"
        return 1
    fi

    local previous
    previous=$(cat "$CONFIG_BASE/.previous_version")
    log_info "回滚到: $previous"

    version_switch "$previous"
}
