
  ${INSTANCE_PREFIX}-${INSTANCE_NUM}:
    image: ${IMAGE}
    container_name: ${INSTANCE_PREFIX}-${INSTANCE_NUM}
    user: root
    ports:
      - "${GW_PORT}:${GW_PORT}"
      - "${SFTP_PORT}:${SFTP_PORT}"
      - "${MDNS_PORT_HOST}:${MDNS_PORT}"
      - "${GUARDIAN_PORT}:${GUARDIAN_PORT}"
    volumes:
      # Docker socket and binary (required for sandbox)
      - /var/run/docker.sock:/var/run/docker.sock
      - /usr/bin/docker:/usr/bin/docker:ro
      - ${CONFIG_DIR_ABS}:/home/node/.openclaw
      - ${SKILLS_MERGED_DIR_ABS}:/home/node/.openclaw/skills
      - ${SKILLS_MERGED_DIR_ABS}:/home/node/.claude/skills
      - ${CONFIG_DIR_ABS}/.claude/settings.json:/home/node/.claude/settings.json
      - ${CONFIG_DIR_ABS}/.hermes/config.yaml:/home/node/.hermes/config.yaml
      - ${CONFIG_DIR_ABS}/.hermes/.env:/home/node/.hermes/.env
      - ${CONFIG_DIR_ABS}/../../plugins/self-improvement-monitor:/tmp/openclaw-self-improvement-monitor:ro
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
      - SANDBOX_ENABLED=${SANDBOX_ENABLED:-false}
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
          cpus: '2'
          memory: 4G
    restart: unless-stopped
