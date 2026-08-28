#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""AgentMarkdownWriter — generate CLI-loadable agent / skill markdown files.

Generates from AgentDefinition + SkillSpec:
  - ``.claude/agents/<name>.md`` — loadable via CLI ``@agent-name``
  - ``.atomcode/skills/<name>/SKILL.md`` — skill definition + operation reference
  - ``.atomcode/skills/<name>/reference/...`` — embedded source snippets
  - ``clawcodex-overview.md`` — overview agent (knows every sub-agent's
    responsibilities and the invocation chain)

Design decisions:
  - Rendered with Jinja2 templates; templates can be customized via templates_dir
  - The overview agent is always generated (no extra arguments)
  - ``.claude/agents/*.md`` is the default output format for this path
  - Skill reference scripts are embedded from files instead of rendered
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Template

from extensions.capabilities.agent_definition_protocol import AgentToolConstants
from extensions.sop_converter.sop_prompts import (
    agent_type_to_skill_name,
    append_sop_overview_routing,
    domain_agent_sop_body,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source language → reference file extension mapping
# ---------------------------------------------------------------------------

_LANGUAGE_EXTENSIONS = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "java": ".java",
    "go": ".go",
    "rust": ".rs",
    "ruby": ".rb",
    "shell": ".sh",
    "bash": ".sh",
    "sh": ".sh",
    "powershell": ".ps1",
    "c": ".c",
    "c++": ".cpp",
    "cpp": ".cpp",
    "csharp": ".cs",
    "html": ".html",
    "css": ".css",
    "json": ".json",
    "yaml": ".yaml",
    "yml": ".yml",
    "sql": ".sql",
    "markdown": ".md",
}

_DEFAULT_SOURCE_LANGUAGE = "python"
_DEFAULT_SOURCE_EXTENSION = ".py"


def _source_language_for(skill: dict[str, Any]) -> str:
    """Return the explicit ``source_language`` or a content-inferred language."""
    language = skill.get("source_language")
    if language and str(language).strip():
        return str(language).strip().lower()
    inferred = _infer_source_language(skill.get("source_code", ""))
    return inferred or _DEFAULT_SOURCE_LANGUAGE


def _infer_source_language(source_code: str) -> str | None:
    """Heuristically infer the language from source code content.

    Uses the first markdown code-fence language tag when present; otherwise
    falls back to common shebang / syntax markers.
    """
    first_line = (source_code.splitlines() or [""])[0].strip()
    if first_line.startswith("#!"):
        if "python" in first_line:
            return "python"
        if "bash" in first_line or "sh" in first_line:
            return "bash"
    match = re.match(r"```([A-Za-z0-9_+-]+)", source_code.strip())
    if match:
        return match.group(1).lower()
    return None


def _source_extension_for(language: str) -> str:
    """Map a source language name to a reference file extension."""
    return _LANGUAGE_EXTENSIONS.get(language, _DEFAULT_SOURCE_EXTENSION)


# ---------------------------------------------------------------------------
# Data containers for overview agent
# ---------------------------------------------------------------------------


@dataclass
class AgentComponentInfo:
    """Information about a sub-agent, used by the overview agent for routing."""

    name: str  # "video-ops-agent"
    description: str  # "video operator processing: transcode, slice, watermark"
    capabilities: list[str] = field(default_factory=list)
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    invoke_pattern: str = ""  # '@video-ops-agent transcode input.mp4 to HLS'


@dataclass
class WorkflowStage:
    """A single stage of a workflow, describing how stages connect."""

    name: str  # "video preprocessing"
    order: int  # stage sequence number
    description: str = ""  # "detect and normalize the format of the raw video"
    responsible_agent: str = ""  # "video-ops-agent"
    depends_on: list[str] | None = None  # prerequisite stages
    output_type: str = ""  # "normalized video"


# ---------------------------------------------------------------------------
# Default Jinja2 templates
# ---------------------------------------------------------------------------

_AGENT_MD_TEMPLATE_SRC = """\
---
name: {{ name }}
description: '{{ description | replace("'", "''") }}'
{% if model %}model: {{ model }}{% endif %}
tools:
{% for tool in tools %}
  - {{ tool }}
{% endfor %}
{% if skills %}skills:
{% for skill in skills %}
  - {{ skill }}
{% endfor %}{% endif %}
{% if when_to_use %}when_to_use: '{{ when_to_use | replace("'", "''") }}'{% endif %}
---

# Agent: {{ name }}

{{ description }}

## 使用说明

This agent is auto-generated by SOP Converter. Use `@{{ name }}` to invoke it.

{% if use_lazy_tools %}
## Tool Loading

This agent has **{{ domain_tool_count }} domain tools** that are not loaded upfront.
To work with a domain:

1. Invoke the matching skill via the Skill tool (listed in frontmatter `skills:`)
2. Or use ToolSearch to discover a specific tool by name
3. After Skill invocation, only that skill's tools are visible to the model
{% endif %}
"""

_SKILL_MD_TEMPLATE_SRC = """\
---
name: {{ name }}
description: '{{ description | replace("'", "''") }}'
user-invocable: true
{% if allowed_tools %}allowed-tools:
{% for tool in allowed_tools %}
  - {{ tool }}
{% endfor %}{% endif %}
{% if lifecycle_deps %}lifecycle-deps: {{ lifecycle_deps }}{% endif %}
---

# Skill: {{ name }}

{{ description }}

{% if task_guide %}{{ task_guide }}{% endif %}

## 参数说明

{% if parameters %}
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
{% for param in parameters %}| `{{ param.name }}` | `{{ param.type_hint or '—' }}` | {{ '是' if param.required else '否' }} | {{ param.description }} |
{% endfor %}
{% else %}
无参数。
{% endif %}

## 源码参考

{% if source_code %}
```{{ source_language or 'python' }}
{{ source_code }}
```
{% endif %}

{% if allowed_tool_count and allowed_tool_count > 20 %}
## Tool Access

This skill covers **{{ allowed_tool_count }} tools**. Invoke via Skill tool or
use ToolSearch to load individual tool schemas on demand.
{% endif %}
"""

_OVERVIEW_AGENT_TEMPLATE_SRC = """\
---
name: {{ name }}
description: '{{ description | replace("'", "''") }}'
model: {{ model or 'default' }}
tools:
{% for tool in proxy_tools %}
  - {{ tool }}
{% endfor %}
skills:
{% for skill in all_skills %}
  - {{ skill }}
{% endfor %}
when_to_use: '{{ description | replace("'", "''") }}'
---

# {{ name }} - 总览 Agent

## ⚠ 关键规则（最高优先级 — 必须首先遵守）

1. **用户提到 @<domain>-agent 时 → 立即委派，禁止任何搜索**
   - 立即 ``Agent(subagent_type="<domain>-agent", prompt="...")``
   - **禁止** Read / Grep / Glob / Bash 查找工具定义、参数 schema 或 SDK 源码
   - **禁止** 派 ``general-purpose`` / ``Explore`` 做工具发现

2. **overview 只做路由与汇总，不执行 SDK 任务**
   - **禁止** 自己调用域 ``Skill`` / ``ToolSearch`` / SDK 工具
   - **禁止** Grep / Glob / Bash 广搜 SDK 源码树
   - 域任务一律 ``Agent(subagent_type="<domain>-agent", prompt="...")``

3. **子代理内部顺序固定**: ``Skill → ToolSearch → SDK 工具``
   - 工具调用权限确认 ≠ 失败，等待用户批准即可，**不要**进入诊断
   - 工具真正失败后才可有限诊断（``Read tool spec`` → ``Read wrapper`` 取 ``_SOURCE_DIR``）
   - **禁止** 用 ``Bash`` / ``Grep`` 诊断；**禁止** Grep SDK 源码树

---

{{ description }}

{% if workflow_stages %}
## 工作流概述

{% for stage in workflow_stages %}
### 阶段 {{ stage.order }}: {{ stage.name }}

{{ stage.description }}

- **负责 Agent**: `@{{ stage.responsible_agent }}`
{% if stage.depends_on %}- **前置阶段**: {% for dep in stage.depends_on %}`{{ dep }}`{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}
- **输出**: {{ stage.output_type }}

{% endfor %}
{% endif %}
## 子 Agent 清单

{% for agent in component_agents %}
### @{{ agent.name }}

{{ agent.description }}

**能力**: {% for cap in agent.capabilities %}{{ cap }}{% if not loop.last %}, {% endif %}{% endfor %}

**调用方式**: `{{ agent.invoke_pattern }}`

{% endfor %}
## 使用示例

{% for agent in component_agents %}
- **{{ agent.description }}**: `{{ agent.invoke_pattern }}`
{% endfor %}

{% if workflow_stages %}
## 跨组件编排

当任务需要多个子 Agent 协作时，按工作流阶段顺序依次调用。
每个阶段的输出自动作为下一阶段的输入。

**用户最短指令（Overview）**：`在 run_dir=<绝对路径> 从 Stage N 做到 Stage M` — 详见运行时注入的「流水线 Stage 编排」一节。
{% endif %}
"""

_WORKFLOW_TEMPLATE_SRC = """\
# WORKFLOW.md — auto-generated by SOP Converter
# This IS an orchestrator config file (consumed by extensions/orchestrator,
# which hardcodes the filename WORKFLOW.md). Schema = orchestrator's:
# name/tracker/workspace/hooks/agent/stages/agents. Do NOT confuse with the
# `sop-<name>.yaml` node graph emitted by `clawcodex sop convert` — that is a
# sop-only execution DAG (nodes:), not loadable by the orchestrator.
# Edit this file to customize orchestrator behavior.

name: {{ name }}
description: '{{ description | replace("'", "''") }}'

tracker:
  type: local

workspace:
  type: shared
  base_branch: main

hooks:
  pre_commit: ""
  pre_push: ""
  post_sync: ""

agent:
  model: {{ model or 'default' }}
  test_command: ""
  build_command: ""
  lint_command: ""

{% if workflow_stages %}
stages:
{% for stage in workflow_stages %}
  - id: "stage-{{ stage.order }}"
    name: "{{ stage.name }}"
    agent: "{{ stage.responsible_agent }}"
    depends_on: [{% for dep in stage.depends_on or [] %}"{{ dep }}"{% if not loop.last %}, {% endif %}{% endfor %}]
{% endfor %}
{% endif %}

{% if component_agents %}
agents:
{% for agent in component_agents %}
  - name: "{{ agent.name }}"
    description: "{{ agent.description }}"
    tools:
      - "*"
    skills: []
{% endfor %}
{% endif %}
"""


# ---------------------------------------------------------------------------
# AgentMarkdownWriter
# ---------------------------------------------------------------------------


class AgentMarkdownWriter:
    """Generate CLI-loadable markdown files from Agent definitions + Skill specs.

    Parameters
    ----------
    templates_dir : Path | None
        Optional directory from which to load Jinja2 templates.
        Defaults to the inline templates.
    """

    def __init__(self, templates_dir: Path | None = None) -> None:
        self._templates_dir = templates_dir
        self._templates: dict[str, Template] = {}

    def _get_template(self, name: str, default_src: str) -> Template:
        """Get a template (loaded from a file first, falling back to inline)."""
        if name not in self._templates:
            if self._templates_dir:
                tmpl_path = self._templates_dir / f"{name}.j2"
                if tmpl_path.is_file():
                    self._templates[name] = Template(tmpl_path.read_text(encoding="utf-8"))
                    return self._templates[name]
            self._templates[name] = Template(default_src)
        return self._templates[name]

    def write_agent(
        self,
        agent_def: dict[str, Any],
        output_dir: Path,
        bundle: str | Path | None = None,
    ) -> Path:
        """Generate ``.claude/agents/<name>.md``.

        Parameters
        ----------
        agent_def : dict
            Required keys: ``name``, ``description``; optional ``model``,
            ``tools``, ``skills``, ``when_to_use``.
        output_dir : Path
            Output root directory (``.claude/agents/`` is created inside it).
        bundle : str | Path | None
            Optional bundle directory. When provided, the generated domain
            agent body includes the lifecycle prompt block if the bundle
            contains ``.clawcodex/tool-dependencies.yaml``.

        Returns
        -------
        Path
            Path to the generated agent markdown file.
        """
        agents_dir = output_dir / ".claude" / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)

        name = agent_def.get("name", "unnamed-agent")
        file_path = agents_dir / f"{name}.md"

        raw_tools = agent_def.get("tools", [])
        is_domain_agent = str(name).endswith("-agent") and name != "clawcodex-overview"
        use_lazy = len(raw_tools) > AgentToolConstants.MAX_INLINE_TOOL_DISPLAY

        if is_domain_agent:
            render_tools = AgentToolConstants.registered_domain_agent_tools()
            skill_name = agent_type_to_skill_name(name)
            skills_list = agent_def.get("skills") or [skill_name]
        else:
            render_tools = (
                AgentToolConstants.registered_proxy_base_tools()
                if use_lazy
                else [t for t in raw_tools if t not in AgentToolConstants.UNREGISTERED_SPECIAL_TOOLS]
            )
            skills_list = agent_def.get("skills", [])

        tmpl = self._get_template("agent_md", _AGENT_MD_TEMPLATE_SRC)
        content = tmpl.render(
            name=name,
            description=agent_def.get("description", ""),
            model=agent_def.get("model"),
            tools=render_tools,
            skills=skills_list,
            when_to_use=agent_def.get("when_to_use"),
            use_lazy_tools=use_lazy and not is_domain_agent,
            domain_tool_count=len(raw_tools),
        )

        if is_domain_agent:
            skill_name = agent_type_to_skill_name(name)
            body = domain_agent_sop_body(
                agent_type=name,
                description=agent_def.get("description", ""),
                skill_name=skill_name,
                bundle=bundle,
            )
            # Replace markdown body after closing frontmatter fence.
            # Split on `---` lines only (not dashes inside quoted YAML
            # string values) so a description containing ``---`` cannot
            # truncate the frontmatter.
            parts = re.split(r"^---\s*$", content, maxsplit=2, flags=re.MULTILINE)
            if len(parts) >= 3:
                content = f"---{parts[1]}---\n\n{body}\n"
            else:
                content = f"{content.strip()}\n\n{body}\n"

        file_path.write_text(content.strip() + "\n", encoding="utf-8")
        logger.info("Wrote agent markdown: %s", file_path)
        return file_path

    def write_skills(
        self,
        skills: list[dict[str, Any]],
        output_dir: Path,
        bundle: str | Path | None = None,
    ) -> list[Path]:
        """Generate ``.atomcode/skills/<name>/SKILL.md`` and reference source snippets.

        Parameters
        ----------
        skills : list[dict]
            Each dict requires keys: ``name``, ``description``;
            optional ``allowed_tools``, ``parameters``, ``source_code``,
            ``source_language`` (e.g. ``"python"``, ``"typescript"``).
        output_dir : Path
            Output root directory.
        bundle : str | Path | None
            Optional bundle directory. Reserved for future lifecycle-aware
            skill rendering; currently the caller is expected to put the
            rendered task guide into ``skill["task_guide"]``.

        Returns
        -------
        list[Path]
            List of paths to the generated SKILL.md files.
        """
        skills_base = output_dir / ".atomcode" / "skills"
        generated: list[Path] = []

        for skill in skills:
            skill_name = skill.get("name", "unnamed-skill")
            skill_dir = skills_base / skill_name
            skill_dir.mkdir(parents=True, exist_ok=True)
            source_language = _source_language_for(skill)

            # Write SKILL.md
            skill_file = skill_dir / "SKILL.md"
            tmpl = self._get_template("skill_md", _SKILL_MD_TEMPLATE_SRC)
            allowed_tools = skill.get("allowed_tools", [])
            content = tmpl.render(
                name=skill_name,
                description=skill.get("description", ""),
                allowed_tools=allowed_tools,
                allowed_tool_count=len(allowed_tools),
                lifecycle_deps=skill.get("lifecycle_deps", ""),
                parameters=skill.get("parameters", []),
                source_code=skill.get("source_code", ""),
                source_language=source_language,
                task_guide=skill.get("task_guide", ""),
            )
            skill_file.write_text(content.strip() + "\n", encoding="utf-8")
            generated.append(skill_file)

            # Write reference source code snippets
            source_code = skill.get("source_code", "")
            if source_code:
                ref_dir = skill_dir / "reference"
                ref_dir.mkdir(parents=True, exist_ok=True)
                ref_file = ref_dir / f"{skill_name}{_source_extension_for(source_language)}"
                ref_file.write_text(source_code, encoding="utf-8")
                generated.append(ref_file)

        return generated

    def write_workflow(
        self,
        name: str,
        description: str,
        component_agents: list[AgentComponentInfo],
        workflow_stages: list[WorkflowStage],
        output_dir: Path,
        model: str = "default",
    ) -> Path:
        """Optionally generate the orchestrator ``WORKFLOW.md``.

        Parameters
        ----------
        name : str
            Agent name.
        description : str
            Agent description.
        component_agents : list[AgentComponentInfo]
            List of sub-agents.
        workflow_stages : list[WorkflowStage]
            Workflow stages.
        output_dir : Path
            Output root directory.
        model : str
            Default model.

        Returns
        -------
        Path
            Path to the generated WORKFLOW.md file.
        """
        file_path = output_dir / "WORKFLOW.md"
        tmpl = self._get_template("workflow", _WORKFLOW_TEMPLATE_SRC)
        content = tmpl.render(
            name=name,
            description=description,
            model=model,
            component_agents=component_agents,
            workflow_stages=workflow_stages,
        )
        file_path.write_text(content.strip() + "\n", encoding="utf-8")
        logger.info("Wrote WORKFLOW.md: %s", file_path)
        return file_path

    def write_overview_agent(
        self,
        name: str,
        description: str,
        component_agents: list[AgentComponentInfo],
        workflow_stages: list[WorkflowStage],
        output_dir: Path,
        model: str = "default",
        sdk_source_dir: str | Path | None = None,
    ) -> Path:
        """Generate the overview agent's ``.claude/agents/<name>.md``.

        Includes the full frontmatter plus system prompt (workflow overview +
        sub-agent delegation guidance).

        Parameters
        ----------
        name : str
            Overview agent name (recommended ``clawcodex-overview``).
        description : str
            Overview agent description.
        component_agents : list[AgentComponentInfo]
            List of sub-agents.
        workflow_stages : list[WorkflowStage]
            Workflow stages.
        output_dir : Path
            Output root directory.
        model : str
            Default model.

        Returns
        -------
        Path
            Path to the generated overview agent markdown file.
        """
        agents_dir = output_dir / ".claude" / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)

        file_path = agents_dir / f"{name}.md"

        # Skill names aligned with pos convert: harness_merged-skill
        all_skills = [agent_type_to_skill_name(a.name) for a in component_agents]

        tmpl = self._get_template("overview_agent", _OVERVIEW_AGENT_TEMPLATE_SRC)
        content = tmpl.render(
            name=name,
            description=description,
            model=model,
            all_skills=all_skills,
            proxy_tools=AgentToolConstants.registered_proxy_base_tools(),
            component_agents=component_agents,
            workflow_stages=workflow_stages,
        )
        content = append_sop_overview_routing(
            content,
            bundle_path=output_dir,
            sdk_source_dir=sdk_source_dir,
            component_agents=component_agents,
        )

        file_path.write_text(content.strip() + "\n", encoding="utf-8")
        logger.info("Wrote overview agent: %s", file_path)
        return file_path
