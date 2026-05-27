{
  "providers": {
    "${MODEL_PROVIDER}": {
      "baseUrl": "${INFER_URL}/v1",
      "apiKey": "${API_KEY:-dummy-key}",
      "api": "openai-completions",
      "models": [
        {
          "id": "${MODEL_NAME}",
          "name": "${MODEL_NAME}",
          "api": "openai-completions",
          "reasoning": false,
          "input": ["text"],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
