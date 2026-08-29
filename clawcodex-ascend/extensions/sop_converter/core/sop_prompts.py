#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Prompt fragments for SOP-to-agent runtime behavior."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SOP_INTERACTIVE_TERMINAL_STOP_LOSS = """\
## 交互式终端停损（阻塞）

当用户要求 **交互式终端 / TUI / REPL / 对话式 CLI / 等待 stdin 的程序**，且已通过
Skill → ToolSearch 调用过**相应 SDK 工具**后，若出现以下任一情况：

- 工具 **超时**（timed out / Command timed out）
- 日志含 **等待输入**、**stdin**、**TTY**、**prompt**、**interactive**、**readline**
- 工具无 error 但长时间无返回（阻塞在交互循环）

则 **立即停损**，归类为 **环境不兼容（Agent 上下文无交互终端）**——**不可**在本会话内完成操作。

### 必须做（一次即可）

1. **Glob/Read workspace** 内用户配置（``spec.yaml``、``*.yaml``、``.clawcodex/`` 等）——确定**用户侧**配置路径；**勿**到 SDK 源码树的 ``tests/``、``fixtures/`` 等目录找示例配置
2. **向用户报告**：本 Agent 环境无真实 TTY，无法代用户完成交互；须在**用户自己的终端**运行
3. **给出可复制命令**，依据（按优先级）：
   - Skill「任务指南」与 ToolSearch 已确认的**公共 API 工具**及其参数
   - 已 Read 的 wrapper ``_SOURCE_DIR`` + 对应模块的 docstring/入口函数
   - manifest 中的 **SDK 源码根** + workspace 内配置文件的**绝对路径**
   - **禁止**照搬 SDK ``tests/``、``examples/`` 下的 launcher/样例脚本路径，除非任务指南明确要求

### 禁止（停损后）

- **禁止** 在 SDK 源码树 ``tests/``、``test/``、``fixtures/`` 等目录 Glob/Grep/Bash/Read **寻找**用户配置或 launcher
- **禁止** background Bash、``script``/``pty`` 伪终端、加长 timeout **重试同一交互式工具**
- **禁止** 再次 ``Skill`` / ``ToolSearch`` / 委派子 Agent **重复同一交互式启动任务**
- **禁止** 宣称「已成功启动交互终端」——超时或阻塞即未成功

交互式程序的正确入口以 **Skill 任务指南 + ToolSearch 匹配的 SDK 工具**为准，不是测试目录里的 wrapper 脚本。"""

# Default: Skill + ToolSearch path. Source exploration is a gated fallback only.
SOP_SOURCE_EXPLORATION_POLICY = """\
## 源码探索策略

**默认（主路径）**
- 通过 Skill + ToolSearch + SDK 工具完成任务
- **允许** Read/Glob 读取 workspace 内用户配置（``spec.yaml``、``*.yaml`` 等）——读完再委派或调用 SDK
- **允许** 开放式任务在 **SDK 源码根**（manifest 绝对路径）下 Grep/Read 理解行为；禁止无边界广搜
- 禁止用 Read/Grep/Glob **搜索 kebab 工具名**来**替代** Skill → ToolSearch → 工具调用
- **禁止**把 Skill ``description`` 里的逻辑路径标签当作相对于当前 workspace 的文件系统路径去 Grep/Glob

**备选项（仅工具失败诊断 — 按顺序，够用即停）**

当且仅当**同时**满足：
1. 已按任务指南完成 Skill → ToolSearch → **至少一次**工具调用
2. 工具返回 error（非用户取消）
3. 同一工具未因「改参数」成功，且未对同一目标重复 Skill/ToolSearch 超过 1 次

允许**有限**诊断（禁止广搜 SDK 树）：
1. Read 该工具的 AgentToolSpec JSON（``~/.clawcodex/agent-tools/`` 或 bundle 内 ``agent-tools/``）——确认 ``input_schema``
2. Read 该工具对应的 wrapper 脚本（``agent-tools/scripts/`` 下 ``*_<hash>.py``）——确认 ``_SOURCE_DIR`` 与 wrapper 实例化 ``cls(...)`` 问题
3. 若需 Grep/Read **SDK 源码**：**必须先**完成步骤 2 取得 ``_SOURCE_DIR``，或使用 system prompt 中的 **SDK 源码根**绝对路径；**禁止**用 Skill description 拼接 workspace
4. 若错误含类/函数名（如 ``SharedMemoryManager.__init__``），仅在上述 SDK 源码根下 Grep/Read **该符号**的定义

诊断结论须明确区分并向用户报告：
- **调用方参数问题**：schema 要求某字段，应补传（如上一步工具返回的路径）
- **工具桥接/实例化问题**：schema 无该字段但 ``__init__``/wrapper 需要 — agent 无法通过补参修复，需维护者修 pos convert
- **跨工具依赖**：需先调用其他工具（说明涉及环节）
- **环境不兼容（交互式终端 / 无 TTY）**：见「交互式终端停损」——**不得**进入备选项源码广搜

得出根因后：**停止 Skill/ToolSearch 重试循环**，用自然语言汇报；禁止启动无边界源码探索。"""

SOP_NO_SOURCE_EXPLORATION = (
    """\
- 禁止用 Grep/Glob/Bash **搜索 kebab 工具名或 wrapper** 来**替代** Skill → ToolSearch
- **允许** overview 先 Glob/Read workspace 配置（``spec.yaml``）；**允许**在 SDK 源码根做有目标的源码阅读
- 工具**未失败**时禁止 Read wrapper / tool spec 代替 Skill「任务指南」

"""
    + SOP_SOURCE_EXPLORATION_POLICY
)

SOP_TOOL_FAILURE_RECOVERY = """\
工具调用失败时的恢复规则（阻塞）：
- **禁止**把报错里的 ``msg`` / ``suggestions`` 当成 SDK 工具的 input 字段——那是权限/框架层信息，不是工具 schema
- 第一次失败：核对 Skill「任务指南」中的参数；跨域编排时 overview 须把上一步结果写入子代理 prompt
- 参数已按任务指南填写仍失败：**先**按「源码探索策略 → 备选项」做有限诊断（Read tool spec → Read wrapper 取 ``_SOURCE_DIR``），**再**向用户报告工具名 + 实际入参 + 根因分类
- **交互式终端超时/无 TTY**：不适用备选项广搜——立即执行「交互式终端停损」
- 禁止因一次失败就无限重复 Skill/ToolSearch；禁止派 ``Explore`` / ``general-purpose`` 广搜源码代替诊断步骤"""

SOP_TOOLSEARCH_GUIDANCE = """\
ToolSearch 用法（用户无需提供工具名）：
1. **优先读 Skill 正文中的「任务指南」表**——用表中的「搜索建议」或用户原话作为 query
2. 第一次：`ToolSearch(query="<用户任务描述或任务指南中的搜索建议>")`
3. 若无匹配或结果不对：**换同义词改写后再搜**（例如用 Skill 任务指南中的搜索建议、或 docstring 里的动词短语）
4. 已知完整 kebab 工具名时可用：`ToolSearch(query="select:<工具名>")`
5. 返回 matches 后**立即调用**该工具；**不要**重复 Skill，**不要**对同一目标反复 ToolSearch 超过 2–3 次
6. ToolSearch 成功后**禁止**再 Read/Grep/Glob/Bash 查工具定义——除非工具调用失败且进入「源码探索策略 → 备选项」"""

SOP_OVERVIEW_ROUTING = f"""\
## SOP 路由（必须遵守）

主循环（总览 / overview）负责**路由与汇总**，**不得**代替域子代理执行 SDK 任务。

### 委派规则（阻塞）

1. 用户点名 ``@<domain>-agent`` 或任务落在某个 SDK 域时，**必须**调用：
   ``Agent(subagent_type="<domain>-agent", prompt="...")``
2. **禁止**用 ``Agent(subagent_type="general-purpose")`` 或 ``Agent(subagent_type="Explore")`` 完成本应由 ``*-agent`` 承担的 SDK 调用
3. **禁止**主循环自己调用域 ``Skill`` / ``ToolSearch`` / 域 SDK 工具（除非用户明确要求总览代劳）
4. 子代理内部顺序固定：**Skill → ToolSearch → SDK 工具**；工具失败后才可进入有限诊断（见源码探索策略）
5. 用户已给出工作流或任务指南中的示例参数时，**直接执行**，不要反复向用户确认

### Skill 正文（阻塞）

- Overview **禁止**臆造 skill 路径并用 Read/Glob 查阅（例如 ``skills/annotated/``）
- **委派** ``Agent(subagent_type="…-agent")`` 后，由**子代理**调用 ``Skill`` 加载任务指南，再 ToolSearch → 工具

### 宏工具意图（阻塞 — 高于源码探索）

若用户话术命中下方 **「宏工具意图」** 表（例如「用手写宏处理文本数据」）：
**立即**按表委派对应域 Agent，**禁止** Explore / 广搜 SDK / 阅读 macro 源码。

### 意图不明确时（阻塞 — 优先于源码探索）

当用户请求模糊（例如「初始化团队对话」「启动 JiuwenAgent」「跑一个 agent」）且无法从
**宏工具意图表**、**SDK 模块总览**、子 Agent 描述或 workflow 表唯一确定目标时：

1. 先查下方 **宏工具意图** 表；未命中再 **Read** bundle 内 ``SDK_OVERVIEW.md``（overview 正文只放路径指针，不内嵌全文）
2. **禁止** Glob/Grep 广搜 SDK 源码树；**禁止**派 ``Explore`` / ``general-purpose`` 做工具发现
3. **禁止** overview 自己调用域 ``Skill`` / ``ToolSearch`` 试探（应委派子 Agent）
4. 若仍有 **2 个及以上**合理候选（域 Agent 或入口 API），**向用户确认**：
   列出选项、各自一句话差异、建议的调用顺序（含前置工具链）
5. 用户确认后再 ``Agent(subagent_type="...", prompt="...")``；prompt 中写明 Skill、
   ToolSearch 查询、以及上一步应产出的对象/路径

### 用户已指定 @agent / 工具名时（阻塞 — 禁止搜索）

当用户消息已包含 ``@<domain>-agent``、kebab 工具名、或任务指南中的步骤时：

1. **禁止**用 Read / Grep / Glob / Bash 查找工具定义、参数 schema 或 SDK 源码
2. **禁止**派 ``Explore`` / ``general-purpose`` 做工具发现
3. **立即** ``Agent(subagent_type="<domain>-agent", prompt="...")``；子代理 prompt 须写明 Skill 名、ToolSearch 查询、参数

### 跨域编排（sop convert 生成）

跨域任务按步骤委派对应域 Agent；需要路由表时 **Read** bundle 内 ``ORCHESTRATION_ROUTES.md``
（overview 正文只放路径指针，不内嵌全文）。
Overview prompt 中须把上一步工具返回的路径/对象写入下一步子代理 prompt。
具体编排路径以当前 bundle 的 Available Skills / workflow.yaml 为准。

### 交互式终端任务

| 阶段 | 行为 |
|------|------|
| 启动 | 委派域 Agent：``Skill`` → ``ToolSearch`` → **调用一次**；prompt 含 workspace 用户配置绝对路径 |
| 超时/阻塞后 | overview **汇总停损**；按「交互式终端停损」给用户终端命令；**禁止**再委派或搜 SDK tests/fixtures |

{SOP_INTERACTIVE_TERMINAL_STOP_LOSS}

{SOP_TOOLSEARCH_GUIDANCE}

{SOP_TOOL_FAILURE_RECOVERY}

子 Agent 与 Skill 一一对应：``<domain>-agent`` ↔ ``<domain>-skill``（以当前 bundle 的 Available Skills 为准）。
"""


def format_sdk_source_dir_block(sdk_source_dir: str | Path | None) -> str:
    """System prompt block with the absolute SDK source root from bundle manifest."""
    if not sdk_source_dir:
        return ""
    path = str(Path(sdk_source_dir).expanduser().resolve())
    path_label = f"{Path(path).name}/openjiuwen"
    return f"""\
## SDK 源码根（pos convert bundle manifest）

- **SDK 源码根目录**：``{path}``
- ``{path_label}`` 是 Skill description 中可能出现的路径标签；不要把它拼到当前 workspace 下当作真实 SDK 路径
- **勿**将 Skill ``description`` 中的路径标签拼到当前 workspace 下当作路径（该字段仅为分组标签）
- 需要 Grep/Read SDK 源码时：优先使用上述绝对路径；若无此块，须先 Read wrapper 脚本中的 ``_SOURCE_DIR``"""


def agent_type_to_skill_name(agent_type: str, *, project_prefix: str | None = None) -> str:
    """Map agent type to flat skill name.

    ``harness_merged-agent`` → ``harness_merged-skill``
    ``AutoResearchClaw-topic-init-agent`` → ``topic-init-skill`` (when prefix given)
    """
    if not agent_type.endswith("-agent"):
        return f"{agent_type}-skill"
    base = agent_type[: -len("-agent")]
    if project_prefix and base.startswith(f"{project_prefix}-"):
        base = base[len(project_prefix) + 1 :]
    return f"{base}-skill"


def pick_pipeline_execute_tool(tools: list[str]) -> str | None:
    """Return the pipeline execute-stage tool from a stage agent tool list, if any."""
    for tool in tools:
        lowered = tool.lower()
        if "execute" in lowered and "stage" in lowered and "pipeline" in lowered:
            return tool
    for tool in tools:
        lowered = tool.lower()
        if lowered.endswith("-execute-stage") or lowered.endswith("execute-stage"):
            return tool
    return None


def infer_stage_label_from_skill(skill_name: str) -> str:
    """``topic-init-skill`` → ``TOPIC_INIT``."""
    base = skill_name[: -len("-skill")] if skill_name.endswith("-skill") else skill_name
    return base.replace("-", "_").upper()


def stage_agent_sop_body(
    *,
    agent_type: str,
    skill_name: str,
    stage_label: str,
    pipeline_tool: str | None = None,
    bridge_tool: str | None = None,
    sdk_source_dir: str | Path | None = None,
    stage_id: int | None = None,
    output_files: list[str] | None = None,
    contract_dod: str | None = None,
) -> str:
    """Condensed SOP body prepended to stage agents (keeps output contracts below)."""
    sdk_block = format_sdk_source_dir_block(sdk_source_dir)
    sdk_section = f"\n\n{sdk_block}" if sdk_block else ""
    pipeline_line = (
        f"- Pipeline 主工具：`{pipeline_tool}` — `stage`=`{stage_label}`，`run_dir`=<绝对路径>；"
        "`config`/`adapters`/`run_id` 可省略（从 run_dir/config.yaml 加载）"
        if pipeline_tool
        else (
            f"- Pipeline 主工具：Skill 任务指南中的 execute-stage 工具（ToolSearch 返回），"
            f"`stage`=`{stage_label}`，`run_dir`=<绝对路径>"
        )
    )
    bridge_line = (
        f"\n- Bridge（仅 Hybrid/Wrapper 且 Skill 任务指南明确指向时）：`{bridge_tool}`，"
        "`stage_id`（int）+ `project_dir`/`run_dir` — **不可**作为 pipeline 超时/失败的替代"
        if bridge_tool
        else ""
    )
    stage_dir_hint = (
        f"`run_dir/stage-{stage_id:02d}/`"
        if stage_id is not None
        else "`run_dir/stage-XX/`（本阶段编号见 workflow 或下方阶段目标）"
    )
    outputs_block = ""
    if output_files:
        lines = "\n".join(f"  - `{f}`" for f in output_files)
        dod_line = f"\n- **DoD:** {contract_dod}" if contract_dod else ""
        outputs_block = f"""
## 执行后必须验证

- Read {stage_dir_hint}`decision.json`（`status`=done 且 `decision`=proceed 为通过）
- 确认以下产出存在且非空：{dod_line}
{lines}
- 汇总：主路径是否成功、耗时（若有 stage_health.json）、产出路径
"""
    return f"""\
## 默认用户指令（无需长篇 prompt）

用户只需 `@` 本 agent 并提供 **`run_dir`（或 `project_dir`）** 请求执行本阶段即可；**不要**要求用户写出 Skill 名、ToolSearch 查询、工具名或 fallback 禁令——由本段 SOP 自动执行。

**最短有效示例：**
```
@agent-{agent_type} 在 run_dir=<绝对路径> 执行本阶段
```

若用户已给出 `run_dir` 与阶段意图，**立即**按下方流程执行，勿反复确认。

## SOP 工作流（阻塞 — 必须按顺序）

1. **第一步（必须，仅一次）**：`Skill(skill="{skill_name}")`
2. **第二步**：`ToolSearch(query="execute pipeline stage {stage_label}")` 或 Skill 任务指南中的搜索建议
3. **第三步**：调用 pipeline 主工具
{pipeline_line}{bridge_line}
4. **第四步**：验证输出契约与 gate 结果（见下方「输出契约」及「执行后必须验证」）
5. **主路径失败**：报告工具名 + error，**停止** — 禁止因超时/一次失败就改调 Bridge、Bash 脚本或其他 SDK 工具

## 长耗时说明

Pipeline 主工具可能运行 **1–10+ 分钟**（外部 API、批处理、LLM）。等待主工具完成；**禁止**因超时就切换 Bridge 或未在 Skill 中列出的工具。

{SOP_TOOLSEARCH_GUIDANCE}

{SOP_TOOL_FAILURE_RECOVERY}

## 禁止（阻塞）

- 禁止委派 coarse/domain agent 代劳本 stage
- 禁止跳过 Skill → ToolSearch → pipeline 主工具，直接 Grep/Glob SDK 源码
- 禁止 pipeline 主路径失败后静默 fallback；须向用户明确报告
{outputs_block}
## Agent: {agent_type}
{sdk_section}"""


def domain_agent_sop_body(
    *,
    agent_type: str,
    description: str,
    skill_name: str,
    sdk_source_dir: str | Path | None = None,
    bundle: str | Path | None = None,
) -> str:
    """System prompt body for a SOP domain sub-agent.

    Args:
        agent_type: Display name for the agent (e.g. ``"AutoResearch"``).
        description: Free-form description of the agent's purpose.
        skill_name: The SOP skill the agent must load first.
        sdk_source_dir: Optional SDK root path; surfaces a fenced block
            with absolute paths so the agent can Read SDK source on
            demand.
        bundle: Optional bundle directory.  When set, the body picks
            up the L3 lifecycle block if the bundle contains a
            ``tool-dependencies.yaml``.
    """
    sdk_block = format_sdk_source_dir_block(sdk_source_dir)
    sdk_section = f"\n\n{sdk_block}" if sdk_block else ""
    lifecycle_block = _lifecycle_prompt_block(bundle)
    lifecycle_section = f"\n\n{lifecycle_block}" if lifecycle_block else ""
    catalog_section = ""
    # Only inject catalog locations when a concrete bundle is provided —
    # otherwise the block leaks host home paths (CLAWCODEX_HOME) into a
    # generic domain-agent prompt and breaks SDK-agnostic checks.
    if bundle is not None:
        try:
            from extensions.sop_converter.resource_catalog import (
                format_resource_catalog_locations_block,
            )

            catalog_block = format_resource_catalog_locations_block(bundle)
            if catalog_block.strip():
                catalog_section = f"\n\n{catalog_block.strip()}"
        except (ImportError, OSError) as exc:
            logger.warning("resource catalog locations block skipped: %s", exc)
    return f"""\
# Agent: {agent_type}

{description}

## SOP 工作流（阻塞要求 — 必须按顺序执行）

1. **第一步（必须，仅一次）**：调用 `Skill(skill="{skill_name}")` 加载本域工具与说明
2. **第二步**：阅读 Skill 中的 **任务指南**（若有），用 **用户任务描述** 或指南中的「搜索建议」调用 `ToolSearch`
   - 示例：`ToolSearch(query="<把用户原话或同义改写填在这里>")`
   - 第一次无匹配时，换同义词再搜（不要用 kebab 工具名逼用户）
   - 仅在已从任务指南或前次搜索确认工具名时：`ToolSearch(query="select:<完整工具名>")`
3. **第三步**：**立即调用** ToolSearch 返回的 SDK 工具；不要写 wrapper 脚本
4. 参数以 Skill「任务指南」与 ToolSearch schema 为准；跨域编排时 overview 会在 prompt 中给出上一步结果
5. 任务指南已给出示例参数时直接使用，不要向用户重复确认
6. **第四步（仅工具失败）**：按「源码探索策略 → 备选项」做有限诊断后向用户报告根因，停止 Skill 重试
7. **交互式终端超时/阻塞**：立即执行「交互式终端停损」——引导用户到真实终端，禁止搜 SDK tests/fixtures

{SOP_TOOLSEARCH_GUIDANCE}

{SOP_TOOL_FAILURE_RECOVERY}

{SOP_INTERACTIVE_TERMINAL_STOP_LOSS}
{sdk_section}
{lifecycle_section}
{catalog_section}

## 禁止

{SOP_NO_SOURCE_EXPLORATION}
- 禁止 Skill 成功后再次调用 Skill（除非用户明确要求另一个 skill）
- 禁止要求用户提供具体工具名或 kebab 名称
- ToolSearch 已有合适 matches 时禁止继续 ToolSearch 同一任务
- 禁止调用 ``Agent(subagent_type="Explore")`` 或 ``Agent(subagent_type="general-purpose")`` 代替本域 SDK 工具调用
"""


def _lifecycle_prompt_block(bundle: str | Path | None) -> str:
    """L3 — render the tool lifecycle hint for the system prompt.

    The block is **only** emitted when:

    1. ``bundle`` is not ``None``, AND
    2. ``<bundle>/.clawcodex/tool-dependencies.yaml`` exists, AND
    3. The file parses to a non-empty :class:`ToolDependencyGraph`.

    The content is intentionally compact — a short table of the
    top ``min(N, 3)`` dependencies plus an explicit recovery hint
    for the create→invoke failure mode the lifecycle doc calls out.  The
    block is appended after the SDK source block so it never
    interferes with the existing prompt structure.
    """
    if bundle is None:
        return ""
    try:
        from extensions.sop_converter.core.dependency import (
            load_tool_dependencies,
        )
    except ImportError:  # pragma: no cover — defensive
        return ""
    graph = load_tool_dependencies(bundle)
    if graph is None or graph.is_empty():
        return ""
    deps = graph.dependencies[:3]
    if not deps:
        return ""
    rows: list[str] = []
    for dep in deps:
        shared = ", ".join(dep.shared_params) if dep.shared_params else "—"
        lifecycle = dep.lifecycle or "—"
        rows.append(f"| `{dep.from_tool}` | `{dep.to_tool}` | {shared} | {lifecycle} |")
    table = "\n".join(rows)
    return f"""\
## 工具生命周期提示（来自 bundle 元数据）

本 bundle 检测到以下 create→invoke 依赖链，请按顺序调用：

| 前置工具 | 后置工具 | 共享参数 | 阶段 |
|----------|----------|----------|------|
{table}

- 如果后置工具返回 ``not found`` / ``not exist`` 类错误，**先检查**前置工具是否已调用；不要换工具名继续空转
- 已标记为「自动完成」的中间步骤由 runtime 隐式处理，无需手动调用
- 若 tool-dependencies.yaml 缺失导致本节为空，属于 bundle 元数据问题，**报告用户**而不是改用其他 invoke 工具兜底"""


def format_overview_stage_pipeline_block(bundle_path: Path | None) -> str:
    """Generic overview prompt for chaining FWA stage agents via workflow.yaml."""
    if bundle_path is None:
        return ""
    from extensions.sop_converter.workflow_project import read_workflow_stage_pipeline

    rows = read_workflow_stage_pipeline(bundle_path)
    if not rows:
        return ""

    table_lines = [
        "| # | Stage | Agent | 产出 | 备注 |",
        "|---|-------|-------|------|------|",
    ]
    for row in rows:
        stage_id = row.get("id")
        name = str(row.get("name") or "")
        kind = str(row.get("kind") or "agent")
        agent = row.get("agent")
        outputs = row.get("output_files") or []
        output_text = ", ".join(f"`{f}`" for f in outputs) if outputs else "—"
        if kind == "gate":
            gate_mode = row.get("gate_mode") or "manual"
            note = f"GATE ({gate_mode})"
            agent_text = "—"
        elif kind == "decision":
            note = "DECISION"
            agent_text = "—"
        else:
            note = "—"
            agent_text = f"`{agent}`" if agent else "—"
        id_cell = str(stage_id) if stage_id is not None else "—"
        table_lines.append(f"| {id_cell} | {name} | {agent_text} | {output_text} | {note} |")

    table = "\n".join(table_lines)
    return f"""\
## 流水线 Stage 编排（Overview 默认行为）

用户只需提供 **`run_dir`**（绝对路径）并说明 **从哪一 stage 开始 / 跑到哪一 stage**，Overview **按 workflow 顺序委派 stage agent**，**禁止**替子 agent 执行 Skill / ToolSearch / pipeline 工具。

### 用户最短有效示例

```
在 run_dir=/path/to/run 从 Stage 4 做到 Stage 6
```

```
继续 run_dir=/path/to/run 的流水线（从上次 proceed 的下一阶段开始）
```

```
在 run_dir=/path/to/run 执行到 LITERATURE_SCREEN 为止
```

### Overview 委派规则（阻塞）

1. **只路由**：``Agent(subagent_type="<stage-agent>", prompt="...")`` — 禁止 overview 自己调用 pipeline / 域 SDK
2. **传给 stage agent 的 prompt 保持简短**（子 agent 自带完整 SOP）：
   ```
   在 run_dir=<绝对路径> 执行本阶段
   ```
3. **顺序执行**：每阶段完成后 Read ``run_dir/stage-NN/decision.json``；仅当 ``decision=proceed`` 才委派下一阶段
4. **失败即停**：子 agent 失败时汇总 error，**禁止**改用 coarse agent / bridge / general-purpose 代跑
5. **GATE / DECISION**：见下表「备注」列；gate 未通过时停止并报告，不要跳过
6. **长耗时**：部分 stage（外部 API、实验）可能 **1–10+ 分钟**；等待子 agent 完成，勿因慢而换 agent

### Workflow 阶段表（本 bundle）

{table}
"""


def append_sop_overview_routing(
    body: str,
    *,
    sdk_source_dir: str | Path | None = None,
    bundle_path: str | Path | None = None,
    component_agents: list | None = None,
) -> str:
    body = (body or "").strip()
    parts: list[str] = []
    if body:
        parts.append(body)
    pipeline_block = format_overview_stage_pipeline_block(Path(bundle_path) if bundle_path is not None else None)
    if pipeline_block and pipeline_block.strip() not in body:
        parts.append(pipeline_block.strip())
    routing = SOP_OVERVIEW_ROUTING.strip()
    if routing not in body:
        parts.append(routing)
    from extensions.sop_converter.cross_domain_orchestration import (
        format_orchestration_routes_block,
    )
    from extensions.sop_converter.runtime.macros.overview_intent import (
        format_overview_macro_intent_block,
    )
    from extensions.sop_converter.sdk_overview import format_sdk_overview_block

    # Macro intent table must sit above SDK_OVERVIEW pointers so overview
    # routes handwritten-macro tasks without falling back to source search.
    macro_block = format_overview_macro_intent_block(
        Path(bundle_path) if bundle_path is not None else None,
        component_agents=component_agents,
    )
    if macro_block and macro_block.strip() not in body:
        parts.append(macro_block.strip())

    overview_block = format_sdk_overview_block(Path(bundle_path) if bundle_path is not None else None)
    if overview_block and overview_block.strip() not in body:
        parts.append(overview_block.strip())
    orch_block = format_orchestration_routes_block(
        Path(bundle_path) if bundle_path is not None else None,
    )
    if orch_block and orch_block.strip() not in body:
        parts.append(orch_block.strip())
    sdk_block = format_sdk_source_dir_block(sdk_source_dir)
    if sdk_block and sdk_block.strip() not in body:
        parts.append(sdk_block.strip())
    try:
        from extensions.sop_converter.resource_catalog import (
            format_resource_catalog_locations_block,
        )

        catalog_block = format_resource_catalog_locations_block(
            Path(bundle_path) if bundle_path is not None else None,
        )
        if catalog_block and catalog_block.strip() not in body:
            parts.append(catalog_block.strip())
    except (ImportError, OSError) as exc:
        logger.warning("resource catalog locations block skipped: %s", exc)
    return "\n\n".join(parts).strip() if parts else SOP_OVERVIEW_ROUTING.strip()
