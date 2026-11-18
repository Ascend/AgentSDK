# Rollout 统一管理与会话生命周期使用指南

## 简介

Rollout 统一管理（rollout_manager）是 Aura 框架提供的一层 Rollout 侧管理能力：三种工作模式共享同一套入口契约，以会话（session）为单位跟踪一次 Agent 多轮交互，提供会话状态查询、训练停止时的优雅排空，以及 Rollout 资源的统一申请登记与回收。

该能力默认关闭（`enabled: false`）。关闭时不产生任何新增日志，行为与改造前一致；开启后通过统一前缀 `[AURA_ROLLOUT]` 输出可检索分析的生命周期运行流程。

### 能力概览

| 能力 | 说明 |
|------|------|
| 统一入口 | 三种模式统一使用 `create / run / shutdown / status`，入口层不再复制 rollout 组装逻辑 |
| 会话生命周期 | 每条 rollout 任务对应一个 session，成功 `finalize`、失败 `abort`，单条会话失败不影响同批其他会话 |
| 状态观测 | 训练主节点提供 `GET /session_status`，并每轮输出 `sessionmgr.status` 汇总 |
| 优雅关闭 | 停止训练后先等待在途会话结束，超过排空预算则安全终止，再释放资源 |
| 资源回收 | 资源按申请登记顺序释放 RolloutWorker、命名 actor、HTTP 端口与子进程，并输出回收结果 |

### 运行过程

- 入口按 `work_mode` 创建 RolloutManager，完成组装与就绪校验后开始运行
- Rollout 侧为每条任务创建 session，正常完成时 `finalize`，异常/超时/取消时 `abort`
- 每轮结束输出 `sessionmgr.status`，训练主节点可通过 HTTP 查询同一份计数
- 训练正常结束或收到 `SIGINT/SIGTERM` 时，先排空在途会话，再释放资源，最后停止状态端点

### 适用场景

- 需要统一观测三种模式的 Rollout 运行状态与在途会话数
- 需要训练停止时等待在途轨迹自然完成，而不是直接杀进程
- 需要确认 Rollout 占用的 actor、端口、子进程已完整回收
- 需要定位“某轮 rollout 是否发生失败/占位补齐”等数据完整性问题

## 使用方法

### 步骤 1：开启 rollout_manager

配置文件位置：训练 YAML 中 `train_instances[].executor_kwargs.rollout_config` 下。

```yaml
train_instances:
  - name: RL-QWEN3-8B-WITH-MATH
    executor_kwargs:
      work_mode: one_step_off
      rollout_config:
        # ... 其他 rollout 配置 ...
        rollout_manager:        # Rollout 统一管理
          enabled: true
          status_port: 18080
          session_drain_timeout: 240
          shutdown_hard_limit: 1800
```

说明：

- [共卡训练配置文件](../../../../aura/configs/train/verl_train_hybrid_A3_t16_qwen3_8b_math_fsdp_rollout_manager.yaml) 与 [分离训练配置文件](../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_8b_math_fsdp_rollout_manager.yaml) 已预置上述配置块（`enabled: true`）
- 其余配置文件未预置，需要手动在 `rollout_config` 下新增
- 不配置或 `enabled: false` 时，能力关闭，行为与改造前一致

### 步骤 2：配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rollout_manager.enabled` | `false` | 总开关。开启后才输出生命周期日志、启动状态端点并执行显式资源释放 |
| `rollout_manager.status_port` | `18080` | 状态端点起始端口，仅在训练主节点绑定；被占用时自动顺延，最多尝试 10 个端口 |
| `rollout_manager.session_drain_timeout` | `240` | 关闭时等待在途会话结束的秒数，应不小于单条会话耗时 P99 |
| `rollout_manager.shutdown_hard_limit` | `1800` | 关闭等待的防挂死上限，仅用于进程卡死时兜底，不参与业务排空 |

> [!NOTE]
> `session_drain_timeout` 是唯一的业务排空预算。RolloutWorker 端会预留 30 秒回程余量用于自身收尾，因此实际等待在途会话的时间略小于配置值；外层 driver 侧会在此基础上再多等 10 秒，用于吸收 RPC 往返与结果回传延迟。

### 步骤 3：启动训练

与对应模式的启动方式完全一致，无需新增启动参数：

```bash
cd /home/work/AgentSDK/aura
bash scripts/start_rl_with_verl_vllm.sh
```

启动后可在训练主节点日志中确认入口已生效：

```text
[AURA_ROLLOUT] manager.create mode=one_step_off enabled=true
[AURA_ROLLOUT] manager.run mode=one_step_off driver=executor blocking=true
[AURA_ROLLOUT] status.server.start addr=0.0.0.0:18080 path=/session_status
```

one_step_off 模式的特征是 `driver=executor` 且 `blocking=true`：

```text
[AURA_ROLLOUT] manager.create mode=one_step_off enabled=true
[AURA_ROLLOUT] manager.run mode=one_step_off driver=executor blocking=true
```

### 步骤 4：查看会话状态

在训练主节点执行：

```bash
curl http://0.0.0.0:18080/session_status
```

单步异步 / 全异步模式返回示例：

```json
{
  "mode": "one_step_off",
  "enabled": true,
  "running": true,
  "source": "worker",
  "status_port": 18080,
  "sessions": {"active": 3, "finished": 125, "aborted": 0},
  "placeholder": 0
}
```

| 字段 | 说明 |
|------|------|
| `mode` | 当前工作模式 |
| `enabled` | `rollout_manager.enabled` 实际取值 |
| `running` | 入口是否处于运行状态 |
| `source` | 会话计数来源：`worker`（worker actor）/ `local`（入口进程内）/ `trainer`（hybrid trainer 进程） |
| `status_port` | 配置的状态端点端口 |
| `sessions` | `{active, finished, aborted}`；`null` 表示该进程不承载会话 |
| `placeholder` | 因会话失败而插入的零奖励占位数量 |

每轮结束还会输出汇总日志：

```text
[AURA_ROLLOUT] sessionmgr.status active=0 finished=128 aborted=0 placeholder=0
```

计数关系：`create = finalize + abort`，`active` 在每轮结束应归零。

### 步骤 5：正常训练结束

训练达到 `total_train_steps` 后，`executor.fit()` 返回，`start_rollout` 的 `finally` 调用 `RolloutManager.shutdown()`，入口按“先排空、后释放”的顺序自动退出，无需人工干预。

单步异步 / 全异步模式的正常关闭日志顺序：

```text
[AURA_ROLLOUT] manager.shutdown.start mode=one_step_off timeout=240.0
[AURA_ROLLOUT] sessionmgr.drain.start active=2
[AURA_ROLLOUT] sessionmgr.drain.done active=0 elapsed=12.3
[AURA_ROLLOUT] manager.shutdown.drain.done active=0 elapsed=12.4
[AURA_ROLLOUT] manager.shutdown.status active=0 finished=64 aborted=0 placeholder=0
[AURA_ROLLOUT] manager.release.item kind=actor name=RolloutWorker status=released elapsed=0.8
[AURA_ROLLOUT] manager.release.item kind=process name=rollout_weight_manager status=released elapsed=1.2
[AURA_ROLLOUT] manager.release.done owned=6 released=6 failed=0 elapsed=3.1
[AURA_ROLLOUT] manager.shutdown.release.done owned=6 released=6 failed=0 elapsed=15.5
[AURA_ROLLOUT] status.server.stop port=18080 mode=executor
```

hybrid 模式同样在训练结束后的 trainer 进程内执行排空，并在结束后交接资源：

```text
[AURA_ROLLOUT] manager.shutdown.drain.start mode=hybrid timeout=240.0s
[AURA_ROLLOUT] sessionmgr.drain.start active=5
[AURA_ROLLOUT] sessionmgr.drain.done active=0 elapsed=95.1
[AURA_ROLLOUT] manager.shutdown.drain.done mode=hybrid elapsed=95.2s
[AURA_ROLLOUT] manager.shutdown.status active=0 finished=1024 aborted=0 placeholder=0
[AURA_ROLLOUT] status.server.stop port=18080 mode=hybrid
[AURA_ROLLOUT] manager.release.delegated owner=verl status=deferred
```

`shutdown_hard_limit` 仅用于信号中断场景等待训练任务退出，正常训练结束不会触发该上限。

## 源码实现原理

### 组件与源码位置

| 组件 | 源码位置 | 说明 |
|------|---------|------|
| 统一入口 | [rollout_manager.py](../../../../aura/aura/trainer/rollout/rollout_manager.py) | `RolloutManager.create/run/shutdown/status`，三模式组装与关闭编排 |
| 会话管理 | [session_manager.py](../../../../aura/aura/trainer/rollout/session_manager.py) | `SessionManager` / `NullSessionManager`，会话注册、计数与排空 |
| 状态端点 | [session_status_server.py](../../../../aura/aura/trainer/rollout/session_status_server.py) | `GET /session_status`，仅在训练主节点启动 |
| 资源托管 | [resource_ledger.py](../../../../aura/aura/trainer/rollout/resource_ledger.py) | `ResourceEntry` 资源申请登记与按序释放，单项超时隔离 |
| hybrid 注入类 | [agent_loop_manager.py](../../../../aura/aura/trainer/train_adapter/verl/hybrid/agent_loop_manager.py) | `RolloutAgentLoopManager`，trainer 进程内的会话与端点 |
| RolloutWorker | [rollout_worker.py](../../../../aura/aura/trainer/rollout/rollout_worker.py) | 会话创建/结束插槽、`drain_sessions`、`release`、`stop` |
| 资源持有方 | [rollout_controller.py](../../../../aura/aura/controllers/rollout_controller/rollout_controller.py) | 权重管理器、命名队列、RolloutServer 的关闭逻辑 |
| 信号入口 | [start.py](../../../../aura/aura/start.py) | 注册 SIGINT/SIGTERM，路由到 `TrainRouter.cancel_active` |
| driver 排空 | [train_router.py](../../../../aura/aura/trainer/train_router.py) | 排空等待、rollout 任务取消、driver 侧资源释放 |
| 训练侧接入 | [rollout_service.py](../../../../aura/aura/trainer/rollout/rollout_service.py) / [full_async/train_main.py](../../../../aura/aura/trainer/train_adapter/verl/full_async/train_main.py) / [hybrid/train_main.py](../../../../aura/aura/trainer/train_adapter/verl/hybrid/train_main.py) | 创建入口、申请登记 rollout 句柄、异常时取消与等待 |

### 启动调用链

1. 入口脚本 [start_rl_with_verl_vllm.sh](../../../../aura/scripts/start_rl_with_verl_vllm.sh) 按 `work_mode` 启动训练进程
2. 单步异步 / 全异步：`start_rollout` / `start_fully_async_rollout` 调用 `RolloutManager.create(...)`，随后 `run()` 阻塞在 `executor.fit()`，`finally` 中执行 `shutdown()`
3. hybrid：`start_train` 调用 `RolloutManager.create("hybrid", ...)`，只向 verl 注入 `agent_loop_manager_class`，`run()` 立即返回；真正驱动由 `HybridTaskRunner.run()` 完成
4. 每次 `create` 成功后会输出一次 `manager.status` 自检行，便于在没有 HTTP 端点时 grep

### 会话与失败隔离

- `session_id` 复用 `task.task_id`，同一 prompt 的组内序号通过 `slot` 记录
- 单步异步 / 全异步：`stream_generate_trajectories` 为每条任务创建会话，`await` 完成时 `finalize`；超时、取消、重试耗尽分别以 `reason=timeout / cancelled / retry_exhausted` 结束
- hybrid：`async_generate_sequences` 用 `create_task + gather(return_exceptions=True)` 隔离单条失败，并按 `retry_limit` 重试
- 失败会话按同一 prompt 组补零奖励占位，保证 `n_samples_per_prompt` 组内样本数不变
- 晚到的回调只记录 `session.finalize.late` / `session.abort.late`，不重复计数

### 关闭与释放链路

- 非 hybrid 正常路径：`RolloutManager.shutdown` 固定 `drain → release → 停止状态端点` 三段；排空通过 `worker.drain_sessions(timeout)` 完成
- hybrid 正常路径：`HybridTaskRunner.run` 的 `finally` 调用 `RolloutAgentLoopManager.drain_sessions`，随后输出状态快照并停止端点；Actor/PG/NPU 由 verl 持有，只申请登记为 delegated
- 信号路径：`start.py` 捕获信号后，先向 rollout 提交排空，再取消训练任务；非 hybrid 使用 `ray.cancel(force=False)`，hybrid 使用 `AURA_STOP_FLAG` 停止标志
- driver 侧以 `session_drain_timeout + 10s` 等待 rollout 任务完成，超时后强制取消；再以 `shutdown_hard_limit` 等待训练任务退出
- 释放顺序与申请登记一致：RolloutWorker → rollout_weight_manager → rollout_queue → rollout_server → 状态端点；每项独立线程释放，单项失败不阻断其他项

### 日志说明

| 日志事件 | 含义 | 级别 |
|---------|------|------|
| `manager.create` | 入口创建成功，包含 mode 与 enabled | INFO |
| `manager.run` | 入口开始运行，包含 driver 与是否阻塞 | INFO |
| `status.server.start` / `status.server.stop` | 状态端点启动/停止及实际端口 | INFO |
| `status.port.fallback` | 配置端口被占用，自动顺延到下一个端口 | WARNING |
| `manager.status` | 状态自检或查询结果，含 running/source/sessions | INFO |
| `session.create` / `session.finalize` | 会话创建与正常结束 | INFO |
| `session.abort` | 会话异常结束，含 reason 与耗时 | WARNING |
| `session.placeholder` | 失败会话的零奖励占位 | WARNING |
| `session.finalize.late` / `session.abort.late` | 重复或晚到的回调，不计数 | INFO |
| `sessionmgr.status` | 每轮会话计数汇总 | INFO |
| `sessionmgr.drain.start` / `drain.done` / `drain.timeout` | 排空开始、完成、超时剩余数 | INFO/WARNING |
| `manager.shutdown.start` / `drain.done` / `status` | 关闭开始、排空完成、关闭前状态快照 | INFO |
| `manager.release.item` / `manager.release.done` | 单项释放结果与汇总（owned/released/failed） | INFO/WARNING |
| `manager.shutdown.release.deferred` | 任务侧被取消，释放移交 driver | WARNING |
| `manager.release.delegated` | hybrid 资源交接给 verl | INFO |
| `driver.signal.received` | driver 收到 SIGINT/SIGTERM | WARNING |
| `router.signal.cancel.drain.begin` / `done` | driver 侧排空开始/完成 | WARNING |
| `router.wait_rollout.begin` / `done` / `timeout` | 等待 rollout 任务自身关闭 | WARNING |
| `runner.fit.interrupted` / `runner.cancel.sent` | hybrid 训练被打断、trainer 任务被取消 | WARNING |

## 常见问题

### Q1：开启后启动失败，提示端口被占用或缺少依赖

常见原因与处理：

1. `rollout status port is already in use: 18080`：端口被其他进程占用，修改 `status_port` 或释放端口；同一节点并行跑多个训练任务时必须使用不同端口
2. `hybrid rollout manager requires actor_rollout_ref.hybrid_engine=true`：hybrid 模式需保证 `verl_conf.actor_rollout_ref.hybrid_engine: true`
3. `requires a non-empty infer_service`：`executor_kwargs.infer_service` 未配置或引用的服务名不存在

就绪校验失败会直接终止启动，不会进入半启动状态。

### Q2：`/session_status` 返回 `sessions: null` 是什么意思

表示当前进程不承载会话，不是“没有活跃会话”。hybrid 模式下列表中的会话在 trainer 进程内，driver 只做调度，因此返回 `null`；请在 trainer 日志中查看 `sessionmgr.status`。非 hybrid 模式返回 `null` 通常说明 worker 尚未创建或状态查询失败，可检查是否有 `manager.status.worker.failed` 告警。

### Q3：停止训练很慢，是否卡死

先看日志停在哪一段：

- 停在 `sessionmgr.drain.done` 之前：在等长尾会话，属于预期；`session_drain_timeout` 应设置为不小于单条会话 P99
- 停在 `manager.release.item`：某项释放超时（默认 5 秒），会记录 `status=failed`，不影响后续项
- 停在训练任务等待：`shutdown_hard_limit` 只用于防挂死，说明训练侧同步代码未响应取消（如初始化、存档阶段），会等到上限才退出
- 没有出现 `sessionmgr.drain.done`：可能被 `kill -9` 或进程终止，Python 无清理机会，只能依赖 `ray stop` 兜底

### Q4：开启后长尾轨迹不再被超时打断

这是设计行为：开启统一管理后，rollout 侧不设置轨迹超时，以保证排空期间会话可以自然完成。若需要强超时，请通过 agent engine 的 `env_args.trajectory_timeout`（缺省 7200 秒）、`max_steps`、`tool_timeout` 控制，或关闭 `rollout_manager.enabled` 恢复原超时策略。

### Q5：关闭 `rollout_manager.enabled` 会有什么影响

- 不输出任何 `[AURA_ROLLOUT]` 新增日志，不启动状态端点
- 关闭流程与改造前一致，入口不执行显式资源释放，依赖进程退出回收
- 训练数据、权重同步、rollout 生成逻辑均不受影响
