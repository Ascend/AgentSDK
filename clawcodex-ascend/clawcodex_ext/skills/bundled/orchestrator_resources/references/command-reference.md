# 编排器命令速查

> 所有命令的前缀为 `clawcodex-dev orchestrator`。按场景分类。
>
> ℹ️ **本速查表中的命令是正确、可信的，可直接使用。** 仅当对某个子命令的具体参数/用法拿不准时，再用 `-h` 查询核实：
>
> - 顶层：`clawcodex-dev orchestrator -h`
> - 子命令：`clawcodex-dev orchestrator <command> -h`（如 `clawcodex-dev orchestrator issue -h`、`clawcodex-dev orchestrator issue retry -h`）
>
> 无需每次执行前都跑 `-h`——那样太慢。速查表已覆盖常用场景，正常执行即可。

## 前置准备

如果用户使用了虚拟环境，在执行编排器命令前先激活：

```bash
cd /path/to/project
source .venv/bin/activate
clawcodex-dev orchestrator server status
```

也可以直接使用虚拟环境内的 cli，无需手动 activate：

```bash
cd /path/to/project && .venv/bin/clawcodex-dev orchestrator server status
```

**注意**：虚拟环境不是必须的。如果未使用虚拟环境，直接执行 `clawcodex-dev` 即可。

## 守护进程管理

> `server start` 是长驻前台进程，直接运行会阻塞当前 shell。需后台运行（`nohup ... &` 或 `... &`）否则命令无法返回。

```bash
# 启动（后台运行，日志重定向）
nohup clawcodex-dev orchestrator server start --workflow ./workflow.md > /tmp/orchestrator.log 2>&1 &
clawcodex-dev orchestrator server start --workflow ./workflow.md --gateway           # 带 IM 网关
clawcodex-dev orchestrator server start --workflow ./workflow.md --workflow-yaml ./.clawcodex/workflow.yaml

# 查看状态
clawcodex-dev orchestrator server status

# 停止
clawcodex-dev orchestrator server stop --all
clawcodex-dev orchestrator server stop --workspace /tmp/workspaces
```

## Issue 管理

```bash
# 列表与详情
clawcodex-dev orchestrator issue list                                          # 所有 Issue
clawcodex-dev orchestrator issue list --status running                         # 筛选
clawcodex-dev orchestrator issue show --id <issue-id>                          # 详情

# 跟踪
clawcodex-dev orchestrator issue tail --id <issue-id>                          # 实时跟踪
clawcodex-dev orchestrator issue tail --id <issue-id> --turn 5                 # 特定 turn
clawcodex-dev orchestrator issue transcript --id <issue-id>                    # 完整对话
clawcodex-dev orchestrator issue transcript --id <issue-id> --role assistant   # 仅 assistant
clawcodex-dev orchestrator issue diff --id <issue-id>                          # 代码变更
clawcodex-dev orchestrator issue diff --id <issue-id> --full                   # 完整 diff

# 控制
clawcodex-dev orchestrator issue stop --id <issue-id>                          # 终止
clawcodex-dev orchestrator issue pause --id <issue-id> --reason "..."          # 暂停
clawcodex-dev orchestrator issue resume --id <issue-id>                        # 恢复
clawcodex-dev orchestrator issue retry --id <issue-id> --mode reset            # 重置重试
clawcodex-dev orchestrator issue retry --id <issue-id> --mode followup         # 追加 commit
clawcodex-dev orchestrator issue retry --id <issue-id> --mode unblock          # 解封
clawcodex-dev orchestrator issue retry --id <issue-id> --mode reset --force    # 强制重试
clawcodex-dev orchestrator issue rebase --id <issue-id>                        # 变基
clawcodex-dev orchestrator issue rebase --id <issue-id> --force                # 强制变基

# 注入与控制
clawcodex-dev orchestrator issue inject --id <issue-id> "提示内容"              # 注入提示
clawcodex-dev orchestrator issue inject --id <issue-id> --list                 # 列出提示
clawcodex-dev orchestrator issue inject --id <issue-id> --remove 1             # 删除提示
clawcodex-dev orchestrator issue attach --id <issue-id>                        # 实时 TUI 连接
clawcodex-dev orchestrator issue takeover --id <issue-id>                      # 接管为交互式 REPL
clawcodex-dev orchestrator issue resume-session --id <issue-id>                # 恢复 LLM 上下文

# 澄清
clawcodex-dev orchestrator issue clarify --id <issue-id> --answer "..."        # 回答澄清
clawcodex-dev orchestrator issue clarify --id <issue-id> --forward-to-author   # 转发给作者
clawcodex-dev orchestrator issue clarify --id <issue-id> --resolve             # 标记已解决
clawcodex-dev orchestrator issue clarify --id <issue-id> --recheck             # 重新分析

# 审查反馈
clawcodex-dev orchestrator issue feedback --id <issue-id> --list               # 列出待处理反馈
clawcodex-dev orchestrator issue feedback --id <issue-id> --approve            # 批准
clawcodex-dev orchestrator issue feedback --id <issue-id> --dismiss            # 驳回
clawcodex-dev orchestrator issue review --id <issue-id> --approve              # 审查批准
clawcodex-dev orchestrator issue review --id <issue-id> --reject --feedback "..."  # 审查驳回

# 工作区文件
clawcodex-dev orchestrator issue workspace --id <issue-id> --ls                # 文件列表
clawcodex-dev orchestrator issue workspace --id <issue-id> --cat <file>        # 查看文件
clawcodex-dev orchestrator issue workspace --id <issue-id> --edit <file> --with "..."  # 编辑文件

# 初始化
clawcodex-dev orchestrator issue init                                           # 交互式创建 Issue 卡片
clawcodex-dev orchestrator issue init --id F-37.1 --title "..." --non-interactive  # 非交互式
```

## 工作流管理

```bash
clawcodex-dev orchestrator workflow init                                        # 初始化 workflow.md
clawcodex-dev orchestrator workflow init --template workflow-local              # 本地跟踪器模板
clawcodex-dev orchestrator workflow init --non-interactive                      # 非交互式
clawcodex-dev orchestrator workflow list-templates                              # 列出可用模板
```

## 工作区管理

```bash
clawcodex-dev orchestrator workspace list                                       # 列出保留的工作区
clawcodex-dev orchestrator workspace list --status completed                    # 筛选
clawcodex-dev orchestrator workspace show --id <issue-id>                       # 详情
clawcodex-dev orchestrator workspace cd --id <issue-id>                         # 输出路径
clawcodex-dev orchestrator workspace cleanup --id <issue-id>                    # 清理单个
clawcodex-dev orchestrator workspace cleanup --all-completed                    # 清理所有已完成
clawcodex-dev orchestrator workspace cleanup --all-completed --force            # 跳过确认
clawcodex-dev orchestrator workspace verify --id <issue-id>                     # 运行验证
```

## 规则管理

```bash
clawcodex-dev orchestrator rules list                                           # 列出所有规则
clawcodex-dev orchestrator rules review --id 1                                  # 查看规则详情
clawcodex-dev orchestrator rules delete --id 1                                  # 删除规则
clawcodex-dev orchestrator rules extract                                        # 提取规则
clawcodex-dev orchestrator rules extract --dry-run                              # 预览模式
clawcodex-dev orchestrator rules extract --limit 5                              # 限制处理数量
clawcodex-dev orchestrator rules stats                                          # 规则统计
```

## 仪表盘

```bash
clawcodex-dev orchestrator dashboard                                            # 启动 Web 仪表盘
clawcodex-dev orchestrator dashboard --port 8080                                # 指定端口
```

## IM 网关

```bash
clawcodex-dev orchestrator server connect-gateway    # 连接 IM 网关
clawcodex-dev orchestrator server disconnect-gateway  # 断开 IM 网关
```
