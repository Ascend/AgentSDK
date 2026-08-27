# LKB Plan Graph

当启用 `LKB_PLAN_GRAPH` 功能开关时，LKB（Logical Kanban）是支持 ClawCodex Task-v2 的持久化、经过校验的 Plan Graph 权威存储与项目级逻辑协作图。

支持的运行时链路为：

```text
TaskCreate / TaskGet / TaskList / TaskUpdate
    → clawcodex_task_adapter
    → LkbApplicationService
    → plan_graph 处理程序 + 确定性规则
    → JsonFileLkbRepository / JsonBoardStore
    → 读取模型 + 斜杠命令/工具/TUI 投影
```

该包专注于宿主集成，不提供独立的拆解 CLI、MCP 服务、传统 Evidence 工作流、外部求解器流水线或旧版 TodoWrite sidecar。

***

## 1. 架构设计与核心理念

### 1.1 双图共用一致性内核

LKB 采用“双图共用一个一致性内核”的长期架构设计：

- **Plan Graph（本期实现）**：管理 Task、Agent、Claim、依赖关系、执行状态、验收条件和拉取调度。
- **Artifact Consistency Graph（预留扩展）**：未来用于源码、文档、小说事实与冲突管理。
- **Consistency Kernel（一致性内核）**：两者共用通用节点引用（NodeRef）、断言 IR（Canonical Assertion IR）、规则/求解器、Truth Maintenance System (TMS)、证据（Evidence）、审计与解释能力。

### 1.2 四层分层架构

1. **Host Protocol Layer（宿主协议层）**：接收 Task-v2 工具调用，保持兼容；LKB 关闭时退回原生 Task-v2；将 LKB Read Model 投影给 REPL/TUI。
2. **Plan Graph Domain（计划图领域层）**：处理 Task 生命周期、Owner/Claim、Task 依赖无环校验、可执行性判定、验收门禁与拉取调度。
3. **Consistency Kernel（一致性内核层）**：提供通用 NodeRef、断言记录、规则包注册、TMS 依赖推演、证据校验、Validation Run 与审计解释。
4. **Graph Store（图存储层）**：提供 Board 跨进程锁、基于 Snapshot 的版本号（RevisionVector）、原子提交、幂等记录与崩溃恢复。

```text
                       +---------------------------+
                       |   ClawCodex Agent / UI    |
                       +-------------+-------------+
                                     |
                         Task-v2 Tool Protocol
                                     |
                       +-------------v-------------+
                       |      Task-v2 Adapter       |
                       | create/get/list/update     |
                       +-------------+-------------+
                                     |
                    +----------------v----------------+
                    |            Plan Graph           |
                    | Task / Agent / Claim / Depends  |
                    +----------------+----------------+
                                     |
             +-----------------------v-----------------------+
             |                Consistency Kernel             |
             | NodeRef / Assertion IR / Rules / Solver       |
             | Evidence / TMS / Validation / Audit / Explain |
             +-----------------------+-----------------------+
                                     |
                       +-------------v-------------+
                       |        Graph Store         |
                       | revision / tx / events     |
                       +-------------+-------------+
```

### 1.3 基础状态与派生状态分离

LKB 将任务状态解耦为：

- **Base Status（基础状态）**：与 Task-v2 兼容，包含 `pending`、`in_progress`、`completed`。
- **Derived Status（派生状态）**：由当前依赖图、Claim、断言及证据动态推导，包含：
  - `ready`：前置依赖全完成且就绪。
  - `blocked`：存在未完成的上游阻塞任务。
  - `needs_recheck`：历史已完成，但上游依赖或假设发生变更，需重新验证。
  - `needs_review`：需要人工或上层审视。

***

## 2. 核心领域模型与标识契约

### 2.1 Board 与 Plan 作用域区分

- **Board（看板容器）**：按工作区解析、跨会话与进程长期存在的项目级存储容器。同一工作区共享一个 Board。
- **Plan Graph（执行计划）**：Board 内互相隔离的独立执行图。系统不在 Board 级设置全局“当前 Plan”，新会话默认创建或恢复私有 Plan；子 Agent 自动继承父会话 Plan，非父子会话需显式绑定。

### 2.2 NodeRef 规范引用

所有图节点使用规范化的 `NodeRef` 进行引用：
`graph:kind:id`

- 例如：`plan:task:T-001`、`plan:agent:agent-a`、`artifact:file:src/auth.py`。
- 保障全 Board 唯一、支持安全序列化并防止路径穿越。

### 2.3 Claim 原子所有权与并发保护

- 任务领取使用原子的 `Claim` 契约，支持 `active`、`released`、`completed`、`overridden` 状态。
- **并发锁保护**：两个 Agent 同时 Claim 同一任务时，在 Board 锁内重读最新状态，确保恰好仅一个成功，失败者收到 `already_claimed`。
- **Transfer / Override**：非 Owner 释放或转移任务需提供特权角色与显式 Reason，并保留完整审计。

### 2.4 Evidence 证据与失效传播

- 任务完成由结构化 `Evidence` 支撑，包含 `source`（位置与内容 Hash）、`valid_for_revision` 与信任等级。
- **失效传播（Invalidation Propagation）**：当上游已完成任务重新打开（Reopen）或修改关键契约时，受影响的下游完成状态将被标为 `needs_recheck`，其证据标记为 `stale`，并沿依赖闭包传播。

### 2.5 RevisionVector 与命令幂等

- 存储引擎维护 `storeRevision` 以及针对各个 Graph 的 `RevisionVector`。
- 每次写命令均校验 `command_id` 与 `request_hash`：相同请求重写返回首次提交结果，避免重复创建任务或事件；重用相同 `command_id` 但参数不同时拒绝（`idempotency_key_reused`）。

***

## 3. Task-v2 工具适配与映射规则

### 3.1 命令映射与 PatchTask

| Task-v2 操作                          | LKB 领域处理                              |
| ----------------------------------- | ------------------------------------- |
| TaskCreate                          | `CreateTask`                          |
| TaskGet                             | `GetTaskProjection`                   |
| TaskList                            | `ListTaskProjections`（仅列出当前 Plan 内任务） |
| TaskUpdate (subject/description)    | `UpdateTaskFields`                    |
| TaskUpdate (owner 从空到有)             | `ClaimTask`                           |
| TaskUpdate (status=in\_progress)    | `StartTask`                           |
| TaskUpdate (status=completed)       | `CompleteTask`                        |
| TaskUpdate (status=deleted)         | `DeleteTask`                          |
| TaskUpdate (addBlockedBy/addBlocks) | `AddDependency`                       |

当一个 TaskUpdate 包含多个字段修改时，适配器将其封装为原子的 `PatchTask`：基于同一候选 Snapshot 进行整体校验，在一次 Board 锁中原子提交，全成或全败。

### 3.2 依赖无环校验（Cycle Detection）

添加依赖边（`AddDependency`）时执行双向一致性处理及环检测（`dependency_cycle`）。如果候选边会导致图构建闭环，直接拒绝并回滚，不产生任何副作用。

### 3.3 轻量拉取调度接口

系统提供原子化任务发现与领取接口，供 Agent 自动或手动拉取任务：

- `list_runnable(board_id, agent_id, limit)`：根据优先级、关键路径及创建时间返回最先可执行的任务列表。
- `claim_next(board_id, agent_id, command_id)`：在单个 Board 锁临界区内原子执行“寻找下一个 Runnable 任务 + 执行 Claim”。

***

## 4. 功能开关与会话管理

### 4.1 功能开关

LKB Plan Graph 由**唯一一个功能开关**控制，默认**关闭**：

| Flag             | 默认值 | 用途                                |
| ---------------- | --- | --------------------------------- |
| `LKB_PLAN_GRAPH` | off | 让持久化 Graph Store 成为 Task-v2 的权威存储 |

开启 `LKB_PLAN_GRAPH` 后，Graph Store 是唯一权威数据源；`ToolContext.tasks` 是从 Store 水合（hydrate）的兼容 Projection/Cache。关闭时走原生 Task-v2 路径，行为不变。

在 Headless / 非交互会话中，开启 `LKB_PLAN_GRAPH` 会**自动同时启用 Task V2 工具面**（无需另行设置 `CLAUDE_CODE_ENABLE_TASKS=1`）。

#### 开启与关闭命令

```bash
# 持久化到配置
clawcodex-dev feature set LKB_PLAN_GRAPH --on
clawcodex-dev feature set LKB_PLAN_GRAPH --off

# 环境变量单次启动
export CLAWCODEX_FEATURE_LKB_PLAN_GRAPH=1
clawcodex-dev
```

交互式会话中输入 `/lkb` 可打开选择菜单进行开关切换。

### 4.2 Board 身份 5 级解析优先级

1. 显式 `board_id`
2. 环境变量 / 启动参数
3. 项目 LKB 配置中的 Board ID
4. 从去凭据的 Git origin + 仓库内相对路径派生；非 Git 工作区回退到规范化工作区根目录
5. 找不到工作区时使用会话级 Board（标记为非项目）

### 4.3 Plan 身份与会话绑定

运维及开发人员可通过斜杠命令查看与管理 Plan：

```text
/lkb plan current
/lkb plan list
/lkb plan new [title]
/lkb plan use <plan_id>
/lkb plan suspend|complete|abandon|archive
/lkb plan reopen <plan_id>
```

Plan 状态机为：

```text
active -> suspended|completed|abandoned|archived
   ^             |
   +-------------+  reopen
```

离开 `active` 状态时会释放该 Plan 内所有活动 Claim 并重置 owner。

***

## 5. 数据目录、原子提交与运维

### 5.1 数据目录布局

默认根目录为 `~/.clawcodex/lkb/`（可通过 `CLAWCODEX_HOME` 修改）：

```text
~/.clawcodex/lkb/
├── boards/
│   └── <safe-board-id>/
│       ├── board.json          # 权威的活动状态 Snapshot
│       ├── board.json.bak      # 上一个已知完好的修订版
│       ├── .lock               # 持久的跨进程锁锚点
│       ├── .lock.owner.json    # 锁持有者诊断信息（PID/host/cmd）
│       ├── .tmp/               # 未提交的候选文件
│       ├── history/            # 压缩后的不可变审计段
│       └── quarantine/         # 待修复的损坏文件
├── archives/<safe-board-id>/   # 显式归档快照
├── tombstones/                 # 已 purge 的 Board 标记
└── .catalog.lock               # 目录列表锁
```

- `board.json` 是唯一权威的活动 Snapshot；不支持按任务拆分的分布式 JSON 写入。
- `.lock` 为持久锁锚点，通过 POSIX `fcntl` / Windows `msvcrt` 实现跨进程互斥。

### 5.2 原子 Snapshot 提交协议

每次状态更新遵循以下严格步骤：

1. 获取 Board 独占跨进程锁及进程内 `RLock`；
2. 重新读取并校验 `board.json` 权威信封；
3. 在内存中构造候选 Envelope，计算 payloadHash 与校验不变量；
4. 写入 `.tmp/<unique>` 文件，执行 `flush` 与 `fsync`；
5. 将当前 `board.json` 复制为 `board.json.bak`（fsync、原子替换）；
6. 使用 `os.replace` 将候选临时文件原子替换为 `board.json`；
7. 目录 fsync 并回读 Header/hash 校验后释放锁；通知进程内 watcher。

### 5.3 损坏检测与自动恢复（Doctor）

加载 Board 时：

1. 校验 `board.json` 的 JSON 格式、Schema、Board ID 与 Payload Hash。
2. 若主文件损坏，校验 `board.json.bak`。若备份有效且属于同一 Board，则在锁内自动恢复备份，并将损坏的主文件移入 `quarantine/`。
3. 若两者均无效，返回 `board_store_corrupt` 错误（绝不返回空 Board）。
4. 运行 `doctor(board_id, repair=True)` 可安全自动修复（从 `.bak` 恢复、清理孤儿临时文件）。

### 5.4 Board 生命周期与文件 GC

- **Board 状态**：`active` → `closed` → `archived` → `trashed` → `purged`。
- **文件 GC 清理规则**：

| 文件路径           | 清理规则                                  |
| -------------- | ------------------------------------- |
| `.tmp/*`       | 持锁状态下 24 小时后 GC；可疑新修订版移入 `quarantine` |
| 会话孤儿 Board     | 7 天后 GC（仅清理崩溃留下的临时孤儿）                 |
| `quarantine/*` | 30 天；删除前先报告                           |
| `tombstones/*` | 90 天；仍被引用时自动续期                        |
| 项目 Board / 归档  | **永不自动删除**                            |
| 用户导出文件         | LKB GC **永不删除**                       |

***

## 6. 展示与交互面

系统采用三层展示原则，避免过多调试信息污染常规视图：

1. **TaskList 摘要**：仅在兼容任务对象中附带紧凑的 `lkb` 派生字段（`derivedStatus`、`claimable`、`activeBlockers` 等）。
2. **详情与解释（`/lkb explain <task_id>`）**：查看特定任务的阻塞原因、失效推导过程与建议恢复动作。
3. **专用 ASCII 看板（`/lkb board`）**：在终端提供统一的排版看板，直观查看所有任务状态、所有者及活动问题。

### 看板 Badge 优先级

REPL / 看板中的 Badge 优先级固定为：
`validation_failed` > `needs_review` > `needs_recheck` > `blocked` > `running` > `ready` > `verified`

***

## 7. 已知限制与验证命令

### 7.1 已知限制

- 每次写入均需序列化完整活跃 Snapshot，开销随单 Board 任务规模增长。
- Board 级锁牺牲了同 Board 内的并发写入并行度，以换取强一致性与可审计性。
- 不支持网络/同步文件系统上的多主并发写入（fail-closed）。
- LKB 仅管理 Plan、Task、依赖、Claim、状态、失效与恢复；不提供 Evidence 自动提取器、文档/代码内容扫描或外部 Solver 运行时。
- `board.json` 中的 `assertions` / `evidence` 槽位仅作 Schema 兼容保存，不参与写命令构造或校验。
- 审计历史段在 POSIX 上强制目录 `0700`、文件 `0600`；Windows 依赖父目录继承 ACL，并会记录一次不含审计数据的安全告警，不执行等价 ACL 验证。

### 7.2 验证命令

在 WSL 或 Linux 环境中运行以下验证命令：

```bash
# 激活开发虚拟环境
. .venv/bin/activate

# 运行单元、UI、集成、存储库与并发测试
python -m pytest \
  extensions/lkb/tests/unit \
  extensions/lkb/tests/ui \
  extensions/lkb/tests/integration \
  extensions/lkb/tests/repository \
  extensions/lkb/tests/concurrency \
  tests/command_system/test_lkb_board_command.py -q

# 代码静态检查与格式校验
python -m ruff check extensions/lkb clawcodex_ext
python -m ruff format --check extensions/lkb clawcodex_ext

# 完整发布演练（包含 AgentLoop 自动化冒烟测试）
python -m pytest \
  extensions/lkb/tests/smoke/test_agent_loop_drill.py -q -s
```
