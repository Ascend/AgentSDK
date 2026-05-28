
  ${INSTANCE_PREFIX}-${INSTANCE_NUM}:
    image: ${IMAGE}
    container_name: ${INSTANCE_PREFIX}-${INSTANCE_NUM}
    user: root
    ports:
      - "${GW_PORT}:${GW_PORT}"
      - "${SFTP_PORT}:${SFTP_PORT}"
      - "${MDNS_PORT_HOST}:${MDNS_PORT}"
      - "${GUARDIAN_PORT}:${GUARDIAN_PORT}"
      - "${MEMEX_PORT}:8080"
    volumes:
      # =============================================================================
      # claude-mem 插件补丁: workspaceDir 不可用修复
      # 补丁文件由 config.sh 在部署时自动生成到 instance-X/plugins/claude-mem/dist/
      # 挂载到容器内 /tmp/openclaw-claude-mem-patch/ 作为源文件
      # health_monitor.sh 会在 Gateway 启动前复制到 overlay 层并设置正确权限
      # 版本: v2026.4.11
      # =============================================================================
      - ${CONFIG_DIR_ABS}/plugins/claude-mem/dist/index.js:/tmp/openclaw-claude-mem-patch/index.js:ro
      - ${CONFIG_DIR_ABS}:/home/node/.openclaw
      - openclaw-skills-${INSTANCE_NUM}:/home/node/.openclaw/skills
      - openclaw-skills-${INSTANCE_NUM}:/home/node/.claude/skills
      - ${CONFIG_DIR_ABS}/.claude/settings.json:/home/node/.claude/settings.json
      - ${CONFIG_DIR_ABS}/.hermes/config.yaml:/home/node/.hermes/config.yaml
      - ${CONFIG_DIR_ABS}/.hermes/.env:/home/node/.hermes/.env
      # =============================================================================
      # self-improvement 插件挂载
      # 将宿主的 plugins/self-improvement-monitor 挂载为只读源
      # =============================================================================
      - ${CONFIG_DIR_ABS}/../../plugins/self-improvement-monitor:/tmp/openclaw-self-improvement-monitor:ro
      # =============================================================================
      # Memex 知识库挂载
      # MEMEX_KB_VOLUME 指向宿主机的知识库 Markdown 文件目录
      # =============================================================================
      - ${MEMEX_KB_VOLUME:-/tmp/memex-empty}:/home/node/wiki
    environment:
      - HOME=/home/node
      - OPENCLAW_HOME=/home/node/.openclaw
      - OPENCLAW_STATE_DIR=/home/node/.openclaw
      - OPENCLAW_CONFIG_PATH=/home/node/.openclaw/openclaw.json
      - CLAUDE_CONFIG_DIR=/home/node/.claude
      - OPENCLAW_GATEWAY_PORT=${GW_PORT}
      - HEALTH_CHECK_PORT=${GW_PORT}
      - HEALTH_CHECK_INTERVAL=30
      - HEALTH_CHECK_FAILURE_THRESHOLD=3
      - ANTHROPIC_API_KEY=${API_KEY:-dummy-key}
      - OPENAI_API_KEY=${API_KEY:-dummy-key}
      - GUARDIAN_PORT=${GUARDIAN_PORT}
      - MEMEX_SERVER_PORT=8080
      - MEMEX_KB_ROOT=/home/node/wiki
    command: /home/node/.openclaw/health_monitor.sh
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:${GW_PORT}/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    restart: unless-stopped
