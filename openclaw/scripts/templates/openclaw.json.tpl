{
  "meta": {
    "lastTouchedVersion": "2026.4.1",
    "lastTouchedAt": "${TIMESTAMP}"
  },
  "wizard": {
    "lastRunAt": "${TIMESTAMP}",
    "lastRunVersion": "2026.3.31",
    "lastRunCommand": "onboard",
    "lastRunMode": "local"
  },
  "messages": {
    "ackReactionScope": "group-mentions"
  },
  "gateway": {
    "mode": "local",
    "auth": {
      "mode": "token",
      "token": "${OPENCLAW_TOKEN}",
      "rateLimit": {
        "maxAttempts": 60
      }
    },
    "port": ${GW_PORT},
    "bind": "lan",
    "controlUi": {
      "allowedOrigins": ["*"],
      "dangerouslyDisableDeviceAuth": true,
      "dangerouslyAllowHostHeaderOriginFallback": true,
      "allowInsecureAuth": true
    },
    "tailscale": {
      "mode": "off"
    }
  },
  "acp": {
    "enabled": true,
    "backend": "acpx",
    "defaultAgent": "openclaw",
    "allowedAgents": [
      "ccb",
      "codex",
      "copilot",
      "cursor",
      "droid",
      "gemini",
      "iflow",
      "kilocode",
      "kimi",
      "kiro",
      "openclaw",
      "opencode",
      "pi",
      "qwen"
    ],
    "maxConcurrentSessions": 8,
    "runtime": {
      "ttlMinutes": 120
    }
  },
  "models": {
    "providers": {
      "${MODEL_PROVIDER}": {
        "baseUrl": "${INFER_URL}/v1",
        "apiKey": "${API_KEY}",
        "auth": "api-key",
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
            "contextWindow": 200000,
            "maxTokens": 8192,
            "compat": {
              "supportsUsageInStreaming": true
            }
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "${MODEL_PROVIDER}/${MODEL_NAME}"
      },
      "models": {
        "${MODEL_PROVIDER}/${MODEL_NAME}": {
          "alias": "${MODEL_NAME}"
        }
      },
      "userTimezone": "Asia/Shanghai",
      "envelopeTimezone": "Asia/Shanghai",
      "maxConcurrent": 4,
      "subagents": {
        "maxConcurrent": 8,
        "announceTimeoutMs": 120000,
        "runTimeoutSeconds": 1200
      },
      "compaction": {
        "mode": "default"
      },
      "contextPruning": {
        "mode": "cache-ttl",
        "ttl": "5m",
        "keepLastAssistants": 3,
        "softTrimRatio": 0.2,
        "softTrim": {
          "maxChars": 4000
        }
      },
      "workspace": "/home/node/.openclaw/workspace",
      "heartbeat": {
        "activeHours": {
          "start": "00:00",
          "end": "23:59",
          "timezone": "Asia/Shanghai"
        }
      },
      "timeoutSeconds": 1200,
      "llm": {
        "idleTimeoutSeconds": 1200
      },
      "memorySearch": {
        "enabled": true,
        "provider": "none",
        "sync": {
          "watch": true,
          "watchDebounceMs": 1500
        },
        "query": {
          "maxResults": 8,
          "hybrid": {
            "enabled": true,
            "vectorWeight": 0,
            "textWeight": 1,
            "candidateMultiplier": 4,
            "mmr": {
              "enabled": false,
              "lambda": 0.7
            },
            "temporalDecay": {
              "enabled": true,
              "halfLifeDays": 30
            }
          }
        },
        "cache": {
          "enabled": true,
          "maxEntries": 50000
        },
        "store": {
          "path": "/home/node/.openclaw/memory/{agentId}.sqlite"
        }
      },
      "sandbox": {
        "mode": "${SANDBOX_MODE:-all}",
        "scope": "agent",
        "workspaceAccess": "ro",
        "workspaceRoot": "/tmp/openclaw-sandboxes",
        "docker": {
          "image": "openclaw-sandbox:bookworm-slim",
          "containerPrefix": "openclaw-sbx-",
          "workdir": "/workspace",
          "network": "bridge",
          "user": "sandbox",
          "tmpfs": ["/tmp", "/var/tmp", "/run"],
          "readOnlyRoot": true,
          "capDrop": ["ALL"],
          "memory": "512m",
          "cpus": 2,
          "pidsLimit": 256
        },
        "prune": {
          "maxAgeDays": 7
        }
      }
    }
  },
  "cron": {
    "store": "/home/node/.openclaw/cron/jobs.json"
  },
  "browser": {
    "enabled": true,
    "executablePath": "/home/node/.cache/ms-playwright/chromium-latest/chrome-linux/chrome",
    "headless": true,
    "noSandbox": true,
    "extraArgs": [
      "--proxy-server=http://127.0.0.1:${MDNS_PORT}",
      "--proxy-bypass-list=localhost,127.0.0.1",
      "--ignore-certificate-errors"
    ],
    "ssrfPolicy": {
      "dangerouslyAllowPrivateNetwork": true
    }
  },
  "memory": {
    "backend": "builtin"
  },
  "plugins": {
    "enabled": true,
    "load": {
      "paths": [
        "/app/extensions/subagent-exec-monitor",
        "/app/extensions/subagent-observability",
        "/app/extensions/subagent-taskr",
        "/app/extensions/self-improvement-monitor"
      ]
    },
    "slots": {
      "memory": "memory-core"
    },
    "entries": {
      "acpx": {
        "enabled": true,
        "config": {
          "permissionMode": "approve-reads",
          "nonInteractivePermissions": "fail",
          "timeoutSeconds": 300
        }
      },
      "memory-core": {
        "enabled": true,
        "config": {
          "dreaming": {
            "enabled": true,
            "frequency": "0 3 * * *",
            "timezone": "Asia/Shanghai",
            "phases": {
              "deep": {
                "enabled": true,
                "limit": 10,
                "minScore": 0.8,
                "minRecallCount": 3,
                "minUniqueQueries": 3,
                "recencyHalfLifeDays": 14,
                "maxAgeDays": 30
              }
            }
          }
        }
      },
      "memory-wiki": {
        "enabled": true,
        "config": {
          "vaultMode": "bridge",
          "vault": {
            "path": "/home/node/.openclaw/wiki/main",
            "renderMode": "native"
          },
          "bridge": {
            "enabled": true,
            "readMemoryArtifacts": true,
            "indexDreamReports": true,
            "indexDailyNotes": true,
            "indexMemoryRoot": true,
            "followMemoryEvents": true
          },
          "search": {
            "backend": "shared",
            "corpus": "all"
          },
          "render": {
            "preserveHumanBlocks": true,
            "createBacklinks": true,
            "createDashboards": true
          },
          "ingest": {
            "autoCompile": true,
            "maxConcurrentJobs": 1,
            "allowUrlIngest": true
          },
          "context": {
            "includeCompiledDigestPrompt": false
          }
        }
      }
    }
  }
}
