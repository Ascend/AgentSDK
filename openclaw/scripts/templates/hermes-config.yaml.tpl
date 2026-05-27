model:
  default: "${MODEL_NAME}"
  provider: "custom"
  base_url: "${INFER_URL}/v1"
  api_key: "${API_KEY}"
terminal:
  backend: "local"
  timeout: 180
agent:
  max_turns: 90
  reasoning_effort: "medium"
memory:
  memory_enabled: true
