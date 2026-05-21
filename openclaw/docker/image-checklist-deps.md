# 镜像依赖项待检查列表

> 用途：每次构建新镜像时，逐项确认以下依赖已正确安装。
> 版本：2026-04-11

---

## 🔴 必须确认项（缺失会导致功能不可用）

### 1. Chromium 运行时依赖

| # | 依赖项 | 包名（Debian） | 用途 | 检查命令 | 通过标准 |
|---|--------|---------------|------|----------|----------|
| ☐ | libnspr4 | `libnspr4` | NSS 基础库 | `ldconfig -p \| grep libnspr4` | 有输出 |
| ☐ | libnss3 | `libnss3` | SSL/TLS 证书 | `ldconfig -p \| grep libnss3` | 有输出 |
| ☐ | libatk-1.0 | `libatk1.0-0` | 无障碍功能 | `ldconfig -p \| grep libatk-1.0` | 有输出 |
| ☐ | libatk-bridge-2.0 | `libatk-bridge2.0-0` | GTK 辅助功能桥接 | `ldconfig -p \| grep libatk-bridge-2.0` | 有输出 |
| ☐ | libcups | `libcups2` | 打印支持 | `ldconfig -p \| grep libcups` | 有输出 |
| ☐ | libdrm | `libdrm2` | GPU 加速 | `ldconfig -p \| grep libdrm.so` | 有输出 |
| ☐ | libxkbcommon | `libxkbcommon0` | 键盘布局 | `ldconfig -p \| grep libxkbcommon` | 有输出 |
| ☐ | libXcomposite | `libxcomposite1` | X11 复合扩展 | `ldconfig -p \| grep libXcomposite` | 有输出 |
| ☐ | libXdamage | `libxdamage1` | X11 损坏检测 | `ldconfig -p \| grep libXdamage` | 有输出 |
| ☐ | libXrandr | `libxrandr2` | 显示配置 | `ldconfig -p \| grep libXrandr` | 有输出 |
| ☐ | libgbm | `libgbm1` | GPU 帧缓冲 | `ldconfig -p \| grep libgbm` | 有输出 |
| ☐ | libasound | `libasound2` | ALSA 音频 | `ldconfig -p \| grep libasound` | 有输出 |
| ☐ | libdbus-1 | `libdbus-1-3` | D-Bus 消息总线 | `ldconfig -p \| grep libdbus-1` | 有输出 |
| ☐ | libxfixes | `libxfixes3` | X11 修复扩展 | `ldconfig -p \| grep libXfixes` | 有输出（注意：库名为 `libXfixes`，grep 需区分大小写） |
| ☐ | libwayland-client | `libwayland-client0` | Wayland 支持 | `ldconfig -p \| grep libwayland-client` | 有输出 |
| ☐ | libpango | `libpango-1.0-0` | 文本渲染 | `ldconfig -p \| grep libpango-1.0` | 有输出 |
| ☐ | libcairo | `libcairo2` | 2D 图形 | `ldconfig -p \| grep libcairo` | 有输出 |
| ☐ | libxcb | `libxcb1` | X11 协议 | `ldconfig -p \| grep libxcb` | 有输出 |

### 2. Chromium 二进制文件

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | Chromium 可执行文件存在 | `ls /home/node/.cache/ms-playwright/chromium-*/chrome-linux64/chrome` | 文件存在 |
| ☐ | Chromium Headless Shell 存在 | `ls /home/node/.cache/ms-playwright/chromium_headless_shell-*/` | 目录存在 |
| ☐ | FFmpeg 存在 | `ls /home/node/.cache/ms-playwright/ffmpeg-*/ffmpeg-linux/ffmpeg` | 文件存在 |
| ☐ | Chromium 能启动（无库依赖错误） | `/home/node/.cache/ms-playwright/chromium-*/chrome-linux64/chrome --headless --no-sandbox --version` | 输出版本号，无 `cannot open shared object file` |

### 3. Playwright 功能验证

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | Playwright CLI 可用 | `npx playwright --version` | 输出版本号 |
| ☐ | Chromium 能打开网页 | `/home/node/.cache/ms-playwright/chromium-*/chrome-linux64/chrome --headless --no-sandbox --dump-dom https://example.com` | 输出 HTML 内容 |

---

## 🟡 建议确认项（缺失可能影响部分功能）

### 4. 核心运行时

| # | 依赖项 | 包名 | 检查命令 | 通过标准 |
|---|--------|------|----------|----------|
| ☐ | Node.js | 内置 | `node --version` | 输出版本号 |
| ☐ | Bun | npm 全局包 | `bun --version` | 输出版本号 |
| ☐ | pnpm | npm 全局包 | `pnpm --version` | 输出版本号 |
| ☐ | Python3 | `python3` | `python3 --version` | 输出版本号 |
| ☐ | gosu | `gosu` | `gosu --version` 或 `gosu node whoami` | 正确降权执行 |
| ☐ | SSH 服务 | `openssh-server` | `sshd -t` | 无语法错误 |
| ☐ | git | `git` | `git --version` | 输出版本号 |

### 5. CLI 命令行工具

| # | 命令 | 检查命令 | 通过标准 |
|---|------|----------|----------|
| ☐ | ccb (claude-code-best) | `ccb --help` | 输出帮助信息，无 `command not found` |
| ☐ | ccb (clawcode 别名) | `clawcode --help` | 与 ccb 输出一致 |
| ☐ | openclaw (OpenClaw 主 CLI) | `openclaw --help` | 输出 OpenClaw 帮助信息 |
| ☐ | claude-mem | `node /usr/local/lib/node_modules/claude-mem/npx-cli/index.js --help` | 输出 claude-mem 帮助信息 |

### 6. Python pip 包

| # | 包名 | 检查命令 | 通过标准 |
|---|------|----------|----------|
| ☐ | markitdown[pptx] | `python3 -m pip show markitdown` | 包已安装 |
| ☐ | python-pptx | `python3 -m pip show python-pptx` | 包已安装 |
| ☐ | python-docx | `python3 -m pip show python-docx` | 包已安装 |
| ☐ | lxml | `python3 -m pip show lxml` | 包已安装 |
| ☐ | openpyxl | `python3 -m pip show openpyxl` | 包已安装 |
| ☐ | pillow | `python3 -m pip show pillow` | 包已安装 |
| ☐ | pdf2image | `python3 -m pip show pdf2image` | 包已安装 |
| ☐ | pdfminer.six | `python3 -m pip show pdfminer.six` | 包已安装 |

### 7. npm 全局包

| # | 包名 | 检查命令 | 通过标准 |
|---|------|----------|----------|
| ☐ | pptxgenjs | `npm list -g pptxgenjs` | 已安装 |
| ☐ | sharp | `npm list -g sharp` | 已安装 |
| ☐ | docx | `npm list -g docx` | 已安装 |
| ☐ | xlsx | `npm list -g xlsx` | 已安装 |
| ☐ | react / react-dom | `npm list -g react` | 已安装 |
| ☐ | react-icons | `npm list -g react-icons` | 已安装 |

---

## 🟢 构建时一次性检查项

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | SSH 目录 `/run/sshd` 存在 | `ls /run/sshd` | 目录存在 |
| ☐ | SSH 配置文件端口为 2222 | `grep "^Port" /etc/ssh/sshd_config` | 输出 `Port 2222` |
| ☐ | SSH root 登录已禁用 | `grep "^PermitRootLogin" /etc/ssh/sshd_config` | 输出 `PermitRootLogin no` |
| ☐ | `node` 用户已创建 | `id node` | uid=1000 |
| ☐ | bun 软链接到 `/root/.bun/bin/bun` | `ls -la /root/.bun/bin/bun` | 链接有效 |
| ☐ | pnpm store 目录 `/app/.pnpm-store` 存在 | `ls /app/.pnpm-store` | 目录存在 |
| ☐ | Node 内存限制已设置 | `echo $NODE_OPTIONS` | 包含 `--max-old-space-size` |
| ☐ | 暴露端口正确 | `grep EXPOSE Dockerfile` | 包含 `2222` 和 `18792` |

---

## 🔵 插件检查项（plugins/）

### 8. subagent-coordinator 插件（SC全家桶）

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | subagent-exec-monitor 插件存在 | `ls /app/extensions/subagent-exec-monitor/dist/` | 目录存在，有 .js 文件 |
| ☐ | subagent-taskr 插件存在 | `ls /app/extensions/subagent-taskr/dist/` | 目录存在，有 .js 文件 |
| ☐ | subagent-observability 插件存在 | `ls /app/extensions/subagent-observability/dist/` | 目录存在，有 .js 文件 |

### 9. claude-mem 插件

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | claude-mem 插件目录存在 | `ls /app/extensions/claude-mem/` | 目录存在 |
| ☐ | claude-mem skills 存在 | `ls /app/extensions/claude-mem/skills/` | 目录存在，有 skill 文件 |
| ☐ | claude-mem modes 目录存在 | `ls /usr/local/lib/node_modules/claude-mem/modes/` | 目录存在，含有 code.json |
| ☐ | claude-mem modes/code.json 存在 | `cat /usr/local/lib/node_modules/claude-mem/modes/code.json` | JSON 文件存在且包含 observation_types |
| ☐ | claude-mem worker-service.cjs 存在 | `ls /usr/local/lib/node_modules/claude-mem/scripts/worker-service.cjs` | 文件存在且可执行 |
| ☐ | claude-mem CLI 可用 | `claude-mem --help` 或 `node /usr/local/lib/node_modules/claude-mem/dist/npx-cli/index.js --help` | 输出版本或帮助信息 |

> **注意**：claude-mem worker 不会随镜像默认 CMD 启动（镜像默认启动 OpenClaw gateway）。如需检查 worker 健康状态，需手动启动：`gosu node bun /usr/local/lib/node_modules/claude-mem/scripts/worker-service.cjs &`

### 10. ccb 插件

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | ccb 配置存在 | `ls /home/node/.claude/` | 目录存在 |

### 11. OpenClaw 项目技能（/home/node/.openclaw/skills/）

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | skills 目录存在 | `ls /home/node/.openclaw/skills/` | 目录存在 |
| ☐ | openclaw-codeagent-workflow skill 存在 | `ls /home/node/.openclaw/skills/openclaw-codeagent-workflow/SKILL.md` | 文件存在 |
| ☐ | mineru-to-markdown skill 存在 | `ls /home/node/.openclaw/skills/mineru-to-markdown/SKILL.md` | 文件存在 |
| ☐ | ppt-multi-style-generator skill 存在 | `ls /home/node/.openclaw/skills/ppt-multi-style-generator/SKILL.md` | 文件存在 |
| ☐ | ascend-download skill 存在 | `ls /home/node/.openclaw/skills/ascend-download-skill/SKILL.md` | 文件存在 |
| ☐ | skill 文件总数不少于 1 | `ls /home/node/.openclaw/skills/ \| wc -l` | 大于 0 |

### 12. Hermes Agent

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | Hermes CLI 符号链接存在 | `ls -la /usr/local/bin/hermes` | 指向 `/home/node/.hermes/hermes-agent/venv/bin/hermes` |
| ☐ | Hermes venv 目录存在 | `ls /home/node/.hermes/hermes-agent/venv/` | 目录存在，含 bin/include/lib |
| ☐ | Hermes venv Python 可执行 | `/home/node/.hermes/hermes-agent/venv/bin/python --version` | 输出版本号 |
| ☐ | Hermes CLI 可用 | `hermes --help` | 输出帮助信息，无 `command not found` |
| ☐ | hermes-agent Python 包已安装 | `/home/node/.hermes/hermes-agent/venv/bin/pip show hermes-agent` | 包已安装 |
| ☐ | Hermes 工作目录所有权为 node | `ls -la /home/node/.hermes/` | 所有者为 node:node |

---

## 🟠 容器启动检查（运行时 — 容器运行后才检查）

> claude-mem worker 由 OpenClaw gateway 启动，非镜像默认 CMD。以下检查项在镜像构建检查时跳过，在容器运行时检查。

### 13. claude-mem worker（运行时必查）

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | claude-mem worker health 正常 | `curl -s http://127.0.0.1:37700/api/health` | 返回 `{"status":"ok","initialized":true}` |
| ☐ | claude-mem worker 端口可连接 | `nc -z 127.0.0.1 37700 && echo "端口开放"` 或 `ss -tlnp \| grep 37700` | 端口处于 LISTEN 状态 |
| ☐ | claude-mem session 初始化正常 | `curl -s -X POST http://127.0.0.1:37700/api/sessions/init -H "Content-Type: application/json" -d '{"contentSessionId":"test","project":"test","prompt":"test"}'` | 返回包含 sessionDbId 的 JSON |
| ☐ | claude-mem MCP Ready 状态 | `curl -s http://127.0.0.1:37700/api/health \| jq '.mcpReady'` | 返回 `true` |
| ☐ | claude-mem initialized 状态 | `curl -s http://127.0.0.1:37700/api/health \| jq '.initialized'` | 返回 `true`（表示 modes 和 DB 初始化成功） |

### 14. Hermes Agent 配置（运行时必查）

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | Hermes volume 挂载正常 | `ls /home/node/.hermes/` | 目录存在且非空 |
| ☐ | INSTALL_HERMES 环境变量已设置 | `echo $INSTALL_HERMES` | 有输出（true/false） |
| ☐ | Hermes CLI 可正常执行 chat | `hermes chat -q "Hello"` | 有正常返回（需预先配置 API keys） |

### 15. CCB 运行时验证（运行时必查）

| # | 检查项 | 检查命令 | 通过标准 |
|---|--------|----------|----------|
| ☐ | CCB CLI 可正常执行 | `ccb -p "Hello"` | 有正常返回（需预先配置 API keys） |

## 一键批量检查脚本

```bash
# 在镜像构建后运行，快速确认所有关键依赖
docker run --rm openclaw:2026.4.11-sftp-docx-browser-ccb sh -c '
echo "=== Chromium 依赖检查 ==="
for lib in libnspr4 libnss3 libatk-1.0 libatk-bridge-2.0 libcups libdrm libxkbcommon libXcomposite libXdamage libXrandr libgbm libasound libdbus-1 libXfixes libwayland-client; do
  if ldconfig -p | grep -q "$lib"; then
    echo "✅ $lib"
  else
    echo "❌ $lib 缺失"
  fi
done

echo ""
echo "=== Chromium 启动测试 ==="
/home/node/.cache/ms-playwright/chromium-*/chrome-linux64/chrome --headless --no-sandbox --version 2>&1 && echo "✅ Chromium 可用" || echo "❌ Chromium 不可用"

echo ""
echo "=== Playwright 测试 ==="
npx playwright --version && echo "✅ Playwright 可用"

echo ""
echo "=== CLI 命令 ==="
ccb --help 2>&1 | grep -q "Usage:" && echo "✅ ccb 可用" || echo "❌ ccb 不可用"
clawcode --help 2>&1 | grep -q "Usage:" && echo "✅ clawcode 可用" || echo "❌ clawcode 不可用"
openclaw --help 2>&1 | grep -q "Usage:" && echo "✅ openclaw 可用" || echo "❌ openclaw 不可用"
node /usr/local/lib/node_modules/claude-mem/npx-cli/index.js --help 2>&1 | grep -q "claude-mem" && echo "✅ claude-mem CLI 可用" || echo "❌ claude-mem CLI 不可用"

echo ""
echo "=== 插件检查 ==="
echo "--- claude-mem ---"
ls -la /app/extensions/claude-mem/ 2>/dev/null && echo "✅ claude-mem 已安装" || echo "❌ claude-mem 未安装"
ls -la /app/extensions/claude-mem/skills/ 2>/dev/null && echo "✅ claude-mem skills 已安装" || echo "❌ claude-mem skills 未安装"
ls /usr/local/lib/node_modules/claude-mem/modes/code.json 2>/dev/null && echo "✅ claude-mem modes/code.json 存在" || echo "❌ claude-mem modes/code.json 缺失"
ls /usr/local/lib/node_modules/claude-mem/scripts/worker-service.cjs 2>/dev/null && echo "✅ claude-mem worker-service.cjs 存在" || echo "❌ claude-mem worker-service.cjs 缺失"

echo ""
echo "--- claude-mem worker 运行时检查 ---"
HEALTH_RESP=$(curl -s http://127.0.0.1:37700/api/health 2>/dev/null)
if echo "$HEALTH_RESP" | grep -q '"status":"ok"'; then
    echo "✅ claude-mem worker health 正常"
    echo "$HEALTH_RESP" | jq -r '.initialized' 2>/dev/null | grep -q "true" && echo "✅ claude-mem initialized=true" || echo "❌ claude-mem initialized=false"
    echo "$HEALTH_RESP" | jq -r '.mcpReady' 2>/dev/null | grep -q "true" && echo "✅ claude-mem mcpReady=true" || echo "❌ claude-mem mcpReady=false"
else
    echo "⚠️  claude-mem worker health 异常或未启动（端口 37700）"
fi

echo ""
echo "--- claude-mem session 初始化测试 ---"
SESSION_RESP=$(curl -s -X POST http://127.0.0.1:37700/api/sessions/init -H "Content-Type: application/json" -d '{"contentSessionId":"test-check","project":"test","prompt":"test"}' 2>/dev/null)
echo "$SESSION_RESP" | grep -q "sessionDbId" && echo "✅ claude-mem session 初始化成功" || echo "❌ claude-mem session 初始化失败"

echo ""
echo "--- subagent-coordinator SC全家桶 ---"
ls -la /app/extensions/subagent-exec-monitor/ 2>/dev/null && echo "✅ subagent-exec-monitor 已安装" || echo "❌ subagent-exec-monitor 未安装"
ls -la /app/extensions/subagent-taskr/ 2>/dev/null && echo "✅ subagent-taskr 已安装" || echo "❌ subagent-taskr 未安装"
ls -la /app/extensions/subagent-observability/ 2>/dev/null && echo "✅ subagent-observability 已安装" || echo "❌ subagent-observability 未安装"

echo ""
echo "--- ccb ---"
ls -la /home/node/.claude/ 2>/dev/null && echo "✅ ccb 配置已安装" || echo "❌ ccb 配置未安装"

echo ""
echo "--- OpenClaw 项目技能 ---"
ls /home/node/.openclaw/skills/ 2>/dev/null | head -1 > /dev/null && echo "✅ skills 目录存在" || echo "❌ skills 目录不存在"
ls /home/node/.openclaw/skills/openclaw-codeagent-workflow/SKILL.md 2>/dev/null && echo "✅ openclaw-codeagent-workflow skill 存在" || echo "⚠️  openclaw-codeagent-workflow skill 不存在"
ls /home/node/.openclaw/skills/mineru-to-markdown/SKILL.md 2>/dev/null && echo "✅ mineru-to-markdown skill 存在" || echo "⚠️  mineru-to-markdown skill 不存在"
ls /home/node/.openclaw/skills/ppt-multi-style-generator/SKILL.md 2>/dev/null && echo "✅ ppt-multi-style-generator skill 存在" || echo "⚠️  ppt-multi-style-generator skill 不存在"

echo ""
echo "--- Hermes Agent ---"
ls -la /usr/local/bin/hermes 2>/dev/null && echo "✅ Hermes CLI 符号链接存在" || echo "❌ Hermes CLI 符号链接缺失"
ls -d /home/node/.hermes/hermes-agent/venv 2>/dev/null && echo "✅ Hermes venv 存在" || echo "❌ Hermes venv 缺失"
hermes --help 2>&1 | grep -q "Usage:" && echo "✅ Hermes CLI 可用" || echo "❌ Hermes CLI 不可用"

echo ""
echo "--- Hermes 运行时验证 ---"
timeout 30 hermes chat -q "Hello" 2>&1 | head -5 && echo "✅ Hermes chat 可用" || echo "⚠️  Hermes chat 需要 API keys 配置"

echo ""
echo "--- CCB 运行时验证 ---"
timeout 30 ccb -p "Hello" 2>&1 | head -5 && echo "✅ CCB 可用" || echo "⚠️  CCB 需要 API keys 配置"
'
```

---

## 使用说明

1. **每次构建新镜像版本前**：打印本清单，逐项确认
2. **发现问题项**：记录到对应项备注栏，说明修复方案
3. **CI/CD 集成**：将一键检查脚本加入构建流程，失败则阻止发布
4. **问题追溯**：记录每次构建的检查结果，便于定位线上问题
