{
  "env": {
    "ANTHROPIC_BASE_URL": "${INFER_URL}${ANTHROPIC_SUFFIX}",
    "ANTHROPIC_AUTH_TOKEN": "${API_KEY}",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "${MODEL_NAME}",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "${MODEL_NAME}",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "${MODEL_NAME}"
  },
  "modelType": "anthropic",
  "enabledPlugins": {
    "pr-review-toolkit@claude-plugins-official": true,
    "code-review@claude-plugins-official": true,
    "feature-dev@claude-plugins-official": true,
    "code-simplifier@claude-plugins-official": true
  },
  "skipDangerousModePermissionPrompt": true
}
