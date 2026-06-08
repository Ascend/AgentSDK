# SkillHub CLI

去中心化的技能管理命令行工具，支持从 Git 托管平台（GitHub、Gitee、GitCode）或本地路径发现、安装和管理 AI 技能。

## 特性

- **多平台支持**：兼容 GitHub、Gitee、GitCode 三大平台
- **技能发现**：跨多个源搜索和发现技能
- **依赖管理**：自动依赖解析和安装
- **安全认证**：使用系统 keyring 安全存储令牌
- **智能缓存**：缓存元数据以提升性能
- **美观输出**：进度条和表格美化终端显示
- **完整性验证**：SHA256/SHA512 校验和与 GPG 签名验证
- **审计日志**：记录所有安装事件

## 安装

### 从源码安装

SkillHub 是 AgentSDK 项目的子模块，位于 `openclaw/skillhub` 目录下。

```bash
# 在 AgentSDK 仓库根目录下执行
cd openclaw/skillhub
pip install -e .
```

或使用完整路径：

```bash
# 克隆 AgentSDK 仓库后
cd <AgentSDK-root>/openclaw/skillhub
pip install -e .
```

验证安装：

```bash
skillhub --help
```

### 初次环境诊断

```bash
skillhub doctor doctor --fix
```

创建必要的目录结构并修复常见问题。

## 常用命令速查表

| 操作 | 命令 |
|------|------|
| 环境诊断 | `skillhub doctor doctor --fix` |
| 添加源 | `skillhub source add <name> <url> --type <platform>` |
| 查看源列表 | `skillhub source list` |
| 登录认证 | `skillhub auth login <platform> --token <token>` |
| 搜索技能 | `skillhub search "<query>"` |
| 安装技能 | `skillhub skill install <skill-name>` |
| 安装指定版本 | `skillhub skill install <skill-name>@<version>` |
| 列出已安装 | `skillhub skill list` |
| 查看技能详情 | `skillhub skill info <skill-name>` |
| 升级技能 | `skillhub skill upgrade --skill <skill-name>` |
| 卸载技能 | `skillhub skill uninstall <skill-name>` |
| 查看配置 | `skillhub config list` |
| 清理缓存 | `skillhub cache clear --force` |

## 快速开始

> **注意**：由于 CLI 的嵌套结构，部分命令需要多层调用：
>
> - `skillhub doctor doctor --fix` （双层 doctor）
> - `skillhub skill upgrade --skill <name>` （upgrade 需用 `--skill` 参数）
> - 推荐优先使用 `skillhub skill <command>` 系列命令

### 1. 认证配置

**重要提示：** 部分平台（如 GitCode）可能要求 API 访问时必须携带认证令牌，即使是公开仓库。建议在配置源之前先完成认证。

```bash
# 查看认证状态
skillhub auth status

# 登录（交互式或命令行参数）
skillhub auth login <platform>
skillhub auth login <platform> --token <token>

# 登出
skillhub auth logout <platform>
```

**获取平台访问令牌：**

- **GitHub**：`https://github.com/settings/tokens`（需要 `repo` 或 `public_repo` 权限）
- **Gitee**：`https://gitee.com/profile/personal_access_tokens`（需要 `projects` 权限）
- **GitCode**：`https://gitcode.com/-/profile/personal_access_tokens`（需要 `read_repository` 权限）

### 2. 配置技能源

SkillHub 支持从多个 Git 托管平台获取技能，支持的源类型：

- `github`：GitHub 平台
- `gitee`：Gitee 平台（国内）
- `gitcode`：GitCode 平台

#### 源管理命令

**查看已配置的源：**

```bash
skillhub source list
skillhub source list --verbose  # 详细模式
```

**添加新源：**

```bash
# 参数说明：
# <source-name>：源的名称标识（自定义，如 my-github、my-gitee）
# <repo-url>：仓库 URL 地址（如 https://github.com/my-org/skills）
# --type <platform>：平台类型（可选值：github/gitee/gitcode，默认 github）
# --subpath <subpath>：技能所在子目录（可选），默认为 skills/

# 完整命令
skillhub source add <source-name> <repo-url> --type <platform> --subpath <subpath>

# 参考样例：添加昇腾GitCode 源
skillhub source add ascend-skills https://gitcode.com/Ascend/agent-skills --type gitcode --subpath official/Common
```

**测试源连接：**

```bash
skillhub source test <source-id>  # 测试指定源是否可访问
```

**移除源：**

```bash
skillhub source remove <source-id>
```

### 3. 搜索与安装

```bash
# 全局搜索
skillhub search "<query>"    #如需全量搜索可将"<query>"替换为""

# 在特定源中搜索
skillhub search "<query>" --source <source-name>

# 限制结果数量
skillhub search "<query>" -n 5

# JSON 格式输出
skillhub search "<query>" --json

# 安装技能（推荐使用 skill 子命令）
skillhub skill install <skill-name>

# 安装特定版本（使用 @ 语法）
skillhub skill install <skill-name>@<version>

# 安装选项
skillhub skill install <skill-name> --force      # 强制重新安装
skillhub skill install <skill-name> --no-deps    # 跳过依赖
skillhub skill install <skill-name> --dry-run    # 模拟安装
skillhub skill install <skill-name> --target <custom-path>  # 指定路径
skillhub skill install <skill-name> --source <source-name>     # 指定源
```

### 4. 本地技能管理（可选）

SkillHub 支持从本地目录直接安装技能，适用于本地开发和测试场景。

#### 安装本地技能

将本地开发的技能目录直接安装：

```bash
# 本地目录需包含 SKILL.md 文件
skillhub skill install <local-path>

# 示例：安装当前目录下的技能
skillhub skill install ./my-skill

# 强制覆盖已存在的技能
skillhub skill install <local-path> --force

# 安装到指定路径
skillhub skill install <local-path> --target <custom-path>
```

#### SKILL.md 文件格式

要从本地目录安装技能，该目录必须包含 `SKILL.md` 文件，格式如下：

```markdown
---
name: my-local-skill
version: 1.0.0
description: A skill for local testing
author: your-name
tags: ["test", "local"]
dependencies:
  other-skill: ">=1.0.0"
---

# My Local Skill

技能说明文档内容...
```

**必需字段**：

- `name`：技能名称（如果缺失，使用目录名）
- `version`：版本号（如果缺失，默认为 `latest`）

**可选字段**：

- `description`：技能描述
- `author`：作者名称
- `tags`：标签列表
- `dependencies`：依赖声明（本地安装时不会自动安装依赖）
- `license`：许可证类型

#### 本地技能与远程技能的差异

本地安装的技能会记录到 `installed.json`，后续管理方式与远程技能一致（详见"管理已安装技能"章节）。

| 功能 | 本地技能 | 远程技能 |
|------|---------|---------|
| 安装 | ✓ 支持 | ✓ 支持 |
| 卸载 | ✓ 支持 | ✓ 支持 |
| 依赖管理 | ✗ 跳过 | ✓ 自动解析安装 |
| 升级 | ✗ 不支持 | ✓ 支持 |
| 版本控制 | ✗ 手动维护 | ✓ 自动获取 |
| 校验和 | ✓ 自动计算 | ✓ 自动计算 |
| 来源标识 | `local` | `github/gitee/gitcode` |

**注意**：本地技能不支持升级操作，因为无法从远程源获取新版本。如需更新，请使用 `--force` 重新安装。

### 5. 管理已安装技能

```bash
# 列出已安装技能
skillhub skill list
skillhub skill list --verbose
skillhub skill list --json

# 查看技能详情
skillhub skill info <skill-name>
skillhub skill info <skill-name> --version <version>
skillhub skill info <skill-name> --json

# 升级技能（⚠️ 注意：需要显式指定 --skill 参数）
skillhub skill upgrade --skill <skill-name>
skillhub skill upgrade --skill <skill-name> --version <version>
skillhub skill upgrade --skill <skill-name> --force
skillhub skill upgrade  # 升级所有已安装技能（不指定 skill 参数）

# 卸载技能
skillhub skill uninstall <skill-name>
skillhub skill uninstall <skill-name> --yes
skillhub skill uninstall <skill-name> --force
```

### 6. 缓存管理

```bash
# 查看缓存信息
skillhub cache info

# 清理过期缓存
skillhub cache clean

# 清空全部缓存
skillhub cache clear --force
```

### 7. 配置管理

```bash
# 列出所有配置
skillhub config list

# 查看特定配置项
skillhub config get <config-key>

# 设置配置值
skillhub config set <config-key> <value>

# 重置配置
skillhub config reset --force
```

### 配置参数说明

#### 目录配置

| 参数 | 说明 |
|------|------|
| `config_dir` | 配置文件目录，存储 config.json、sources.json 等 |
| `data_dir` | 数据目录，存储技能、审计日志等 |
| `cache_dir` | 缓存目录，存储元数据、搜索结果缓存 |
| `skills_dir` | 技能安装目录，已安装的技能存放位置 |

#### 缓存配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cache.enabled` | `True` | 是否启用缓存 |
| `cache.ttl_metadata` | `7200` | 技能元数据缓存有效期（秒），默认 2 小时 |
| `cache.ttl_search` | `1800` | 搜索结果缓存有效期（秒），默认 30 分钟 |
| `cache.ttl_releases` | `7200` | 发布版本缓存有效期（秒），默认 2 小时 |
| `cache.max_size_mb` | `1024` | 缓存最大大小（MB），超过时自动清理过期条目 |

#### 平台配置（github/gitee/gitcode）

| 参数 | 说明 |
|------|------|
| `<platform>.api_url` | 平台 API 地址（如 `https://api.github.com`） |
| `<platform>.auth_type` | 认证类型：`pat`（个人访问令牌）、`oauth`（OAuth） |
| `<platform>.rate_limit` | API 速率限制（每小时请求次数） |
| `<platform>.timeout` | API 请求超时时间（秒） |

#### 安全配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `security.allow_unsigned` | `False` | 是否允许安装未签名技能（建议保持关闭） |
| `security.sandbox_installs` | `True` | 是否在沙箱环境执行安装脚本 |
| `security.strict_permissions` | `True` | 是否严格限制安装文件权限（600/700） |
| `security.trusted_authors` | `[]` | 可信作者白名单列表，跳过签名验证 |
| `security.trusted_sources` | `[]` | 可信源白名单列表，跳过部分安全检查 |

#### 发现配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `discovery.default_sources` | `[]` | 默认搜索源列表（未指定源时使用） |
| `discovery.search_timeout` | `30` | 搜索超时时间（秒） |
| `discovery.max_results` | `100` | 最大搜索结果数量 |

#### 日志配置

| 参数 | 说明 |
|------|------|
| `log_level` | 日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `log_file` | 日志文件路径（None 表示不记录到文件） |

#### 配置修改示例

```bash
# 延长元数据缓存有效期到 4 小时
skillhub config set cache.ttl_metadata 14400

# 禁用缓存（调试时使用）
skillhub config set cache.enabled false

# 调整日志级别为 DEBUG
skillhub config set log_level DEBUG

# 设置日志文件路径
skillhub config set log_file /var/log/skillhub.log

# 设置技能安装路径
skillhub config set skills_dir /var/skillhub/skills

# 设置默认搜索源
skillhub config set discovery.default_sources '["github", "gitee"]'
```

## 目录结构

```text
~/.local/share/skillhub/
├── skills/           # 已安装技能
│   └── <skill-name>/ # 技能目录
│       └── SKILL.md
├── cache/            # 缓存数据
├── installed.json    # 安装记录（含校验和）
└── audit.json        # 审计日志

~/.config/skillhub/
└── config.json       # 配置文件
```

## 完整性验证

SkillHub 在安装技能时自动进行完整性验证，通过校验和确保文件未被篡改。

### 验证机制

**自动验证流程：**

1. 安装技能时计算文件哈希（默认 SHA256）
2. 校验和存储到 `installed.json` 和 `audit.json`
3. 后续可通过命令查看或手动验证

**支持的校验和算法：**

- SHA256（默认，安全性高）
- SHA512（安全性最高）
- MD5（仅用于兼容旧系统）

**GPG 签名验证（可选）：**
如果技能清单包含 GPG 签名，SkillHub 在安装时会自动验证签名有效性。

### 验证操作

#### 1. 查看技能校验和信息

```bash
# 查看技能详情（含 checksum 字段）
skillhub skill info <skill-name> --json
```

输出示例：

```json
{
  "installed": {
    "name": "<skill-name>",
    "checksum": "sha256:<hash>",
    "install_path": "/path/to/skill"
  }
}
```

#### 2. 查看安装记录

```bash
# 查看所有已安装技能的校验和
cat ~/.local/share/skillhub/installed.json
```

#### 3. 查看审计日志

```bash
# 查看安装历史（含校验和、来源仓库、Git引用）
cat ~/.local/share/skillhub/audit.json
```

#### 4. 手动验证文件完整性

```bash
# 获取技能安装路径
skillhub skill info <skill-name> --json | grep install_path

# 计算文件哈希并对比 checksum 字段
# Linux（项目部署环境）：
sha256sum ~/.local/share/skillhub/skills/<skill-name>/SKILL.md

# macOS（参考）：
shasum -a 256 ~/.local/share/skillhub/skills/<skill-name>/SKILL.md

# Windows PowerShell（参考）：
Get-FileHash -Algorithm SHA256 ~/.local/share/skillhub/skills/<skill-name>/SKILL.md
```

> **说明**：本项目推荐在 Linux 环境下部署，其他系统命令仅供参考。

### 安全配置

控制完整性验证行为的安全配置：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `security.allow_unsigned` | `false` | 是否允许未签名技能（建议保持关闭） |
| `security.sandbox_installs` | `true` | 是否在沙箱环境执行安装脚本 |
| `security.strict_permissions` | `true` | 是否严格限制文件权限 |
| `security.trusted_authors` | `[]` | 可信作者白名单（跳过签名验证） |
| `security.trusted_sources` | `[]` | 可信源白名单（跳过部分安全检查） |

修改安全配置：

```bash
# 查看当前安全配置
skillhub config get security

# 允许未签名技能（不推荐）
skillhub config set security.allow_unsigned true

# 添加可信作者
skillhub config set security.trusted_authors '["<author-name>"]'

# 添加可信源
skillhub config set security.trusted_sources '["<source-name>"]'
```

## 开发

### 设置开发环境

```bash
# 安装 Poetry
pip install poetry

# 安装依赖
cd <AgentSDK-root>/openclaw/skillhub
poetry install

# 运行测试
pytest tests/

# 运行 lint
ruff check .

# 运行类型检查
mypy .
```

### 项目结构

```text
skillhub/
├── adapters/          # 平台适配器（GitHub, Gitee, GitCode）
├── commands/          # CLI 命令
├── interfaces/        # 抽象基类
├── models/           # Pydantic 模型
├── services/         # 业务逻辑实现
├── utils/            # 工具函数（checksum, archive, semver）
├── cli.py            # CLI 入口点
├── config.py         # 配置管理
├── exceptions/       # 异常定义
└── README.md         # 本文件
```

### 架构说明

SkillHub CLI 采用分层架构：

1. **CLI 层**：基于 Typer 的命令行接口，使用 Rich 实现美观输出
2. **Service 层**：源管理、技能解析、安装引擎等业务逻辑
3. **Adapter 层**：平台特定实现（GitHub、Gitee、GitCode API 适配）
4. **Model 层**：Pydantic 模型实现类型安全数据处理

## 许可证

MIT 许可证 - 详情见 LICENSE 文件。

## 贡献

欢迎贡献！请阅读 [贡献指南](https://gitcode.com/Ascend/AgentSDK/blob/master/contributing.md) 了解如何开始。

## 支持

- **问题反馈**: [GitCode Issues](https://gitcode.com/Ascend/AgentSDK/issues)
