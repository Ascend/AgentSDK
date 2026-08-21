---
# ============================================================================
# Orchestrator Workflow Configuration — TEMPLATE
# ============================================================================
#
# Usage:
#   1. Copy this file to your project root as `workflow.md`
#   2. Replace every {{PLACEHOLDER}} with your actual values
#   3. Customize the Agent Prompt body (the markdown after the second `---`)
#   4. Set the required environment variables (see "Required env vars" below)
#
# Frontmatter (between the two `---` markers) is parsed as YAML by
# extensions/orchestrator/config/schema.py. The body is the agent prompt
# template rendered with Jinja-like syntax for each issue.
#
# Required env vars (depending on tracker.kind):
#   - linear     : LINEAR_API_KEY
#   - github     : GITHUB_TOKEN (or GITHUB_API_KEY)
#   - gitee      : GITEE_TOKEN (or GITEE_API_KEY)
#   - gitcode    : GITCODE_TOKEN (or GITCODE_API_KEY)
#   - local      : (none; uses tracker.issues_path)
# ============================================================================
# ============================================================================
# Tracker — Issue source platform
# ============================================================================
# Picks which issue tracker the orchestrator polls. One of:
#   linear, github, gitee, gitcode, local
tracker:
  kind: {{TRACKER_KIND}}                                  # e.g. gitcode
  # API base URL. Leave blank to use the platform's default endpoint.
  endpoint: {{TRACKER_ENDPOINT}}                          # e.g. https://api.gitcode.com/api/v5
  # --- Repository-backed trackers (github / gitee / gitcode) ---
  owner: {{REPO_OWNER}}                                   # e.g. chadwweng
  repo: {{REPO_NAME}}                                     # e.g. AgentSDK
  clone_url: {{UPSTREAM_CLONE_URL}}                       # upstream URL (issue/PR API target)
  api_key: ${{TRACKER_API_KEY_ENV}}                       # $VAR form reads env at load time
  # Only poll issues assigned to this user/login. Leave blank to track all.
  assignee: {{REPO_ASSIGNEE}}
  # Branch prefix the orchestrator uses when checking out issue branches.
  branch_prefix: {{BRANCH_PREFIX}}                        # e.g. clawcodex
  # --- Linear-only (ignored by other trackers) ---
  # project_slug: my-team/my-project
  # --- Local-only (ignored by repository trackers) ---
  # issues_path: .clawcodex_local_issues
  # --- State filter (works for all repository-backed trackers) ---
  # GitHub / Gitee / GitCode all use state=open|closed|all. The defaults
  # are returned by tracker.default_active_states_for_kind() — override here
  # only if your repo uses custom state names.
  active_states:
    - open                                                 # all repo trackers (github/gitee/gitcode)
  terminal_states:
    - closed
  # Issues carrying any of these labels (case-insensitive) are excluded from
  # the candidate queue at fetch time. Use for web-only workflow labels
  # (e.g. "completed", "wontfix") that the tracker state does not reflect.
  # skip_labels: [wontfix]
  # Issues must carry at least ONE of these labels (OR semantics) to enter
  # the candidate queue. Use to scope the orchestrator to a class of work
  # (e.g. only `priority/high`). Empty = no requirement.
  # require_any_labels: [priority/high]

# ============================================================================
# Polling — how often to fetch issue lists
# ============================================================================
polling:
  interval_ms: 60000                                       # 60s

# ============================================================================
# Workspace — local clone and per-issue checkout
# ============================================================================
workspace:
  root: {{WORKSPACE_ROOT}}                                 # e.g. /tmp/symphony_workspaces/myrepo
  repo_clone_url: {{REPO_CLONE_URL}}                       # clone URL (passed to `git clone`)
  # Fork workflow: upstream repo URL (PR target). Same as repo_clone_url = single-repo mode.
  upstream_clone_url: {{UPSTREAM_CLONE_URL}}               # upstream repo URL (PR target)
  clone_depth: 1                                           # shallow clone for speed
  base_branch: ""                                           # repository default branch (main / master); empty = auto-detect
  checkout_issue_branch: true                              # create per-issue branch from main
  git_username: {{GIT_PUSH_USER}}                         # git commit author name
  git_email: {{GIT_PUSH_EMAIL}}                              # git commit author email
  git_token: ${{GIT_PUSH_TOKEN_ENV}}                       # $VAR form reads from env at load time
  gitignore_patterns:                                      # paths excluded from agent workspace
    - "*.pyc"
    - __pycache__
    - "*.egg-info"
    - .pytest_cache
    - analysis.md
    - changes_summary.md
    - implementation_notes.md
    - verification_report.md
  # Workspace strategy: isolated (default, per-issue dir) | shared | sequential
  strategy: isolated
  # Sequential mode: the integration branch all issues commit onto.
  # integration_branch: main
  # Require a clean workspace before starting / between issues.
  require_clean_start: true
  require_clean_between_issues: true
  # Keep the workspace after terminal states so you can inspect artifacts,
  # re-run verification, or debug failures. Default: keep on all terminals.
  preserve_on_terminal: true
  preserve_on_failure: true
  preserve_on_abandoned: true
  preserve_on_timeout: true
  sequential_lock: true
  # Python interpreter resolution. Empty = auto-detect via python_detect_files,
  # then fall back to agent.python_executable, then PATH python3.
  # python_executable: ""
  python_auto_detect: true
  python_detect_files:
    - .python-version
    - pyvenv.cfg
    - .venv/pyvenv.cfg
    - Pipfile
    - environment.yml

# ============================================================================
# Agent — Claude invocation parameters
# ============================================================================
agent:
  max_concurrent_agents: 1                                 # global concurrency cap
  max_concurrent_agents_by_state:                          # per-state cap (keys = active state names)
    open: 3
  max_turns: 200                                            # tool-call turns per issue run
  max_retry_attempts: 6
  max_retry_backoff_ms: 300000                             # 5 min backoff between retries
  max_retries_per_issue: 3                                 # max operator-driven retries per issue
  max_turns_retry_delay_ms: 30000                          # base delay before retrying after max_turns
  run_timeout_ms: 1800000                                  # 30 min per-run timeout
  delay_between_requests_ms: 2000                          # min gap between provider API requests (anti-rate-limit)
  stall_timeout_ms: 300000                                 # abort run after this much stream inactivity (0=off)
  stall_warn_ms: 30000                                     # emit stall_suspected diagnostic after this silence
  audit_log: minimal                                       # tool-call audit: none | minimal | full
  provider: anthropic                                      # anthropic | openai | ...
  model: {{MODEL}}                                         # model name, leave blank for provider default
  # permission_mode MUST be bypassPermissions for unattended orchestrator runs.
  # - dontAsk still triggers ApprovalPolicy checks → can block headless runs
  # - bypassPermissions → headless sets permission_handler=None, all tools auto-approved
  # schema.py auto-promotes dontAsk → bypassPermissions when tracker is present.
  permission_mode: bypassPermissions
  # Test / build / lint commands (runs before git push)
  test_command: ""                                # test command (empty = auto-detect)
  # build_command: "make build"                            # build command
  # lint_command: "ruff check ."                           # lint command
  # Human review gate: when True, each completed issue requires manual approval
  # via `clawcodex-dev orchestrator issue review --approve --id <id>`.
  # review_required: false
  # auto_approve: false                                    # auto-approve completed issues
  # coordinator_mode: false                                # boot agent with restricted coordinator tool set
  # allowed_changed_files: [src/**]                        # whitelist globs for files allowed into the commit
  # Stagnation / loop guards
  # max_no_op_turns: 3                                     # consecutive empty turns → stagnation
  # loop_detection_window: 5                               # look-back window for loop detection
  # loop_detection_threshold: 3                            # repeated tool signature → loop_detected
  # max_tools_per_turn: 50                                 # per-turn tool call cap (prevents infinite loops)
  # phases: [analyze, implement, verify]                   # named phases driving honest progress %
  # fallback_to_phase_step: false                          # legacy phase*25 progress fallback
  # Environment variables merged into every Bash subprocess and orchestrator
  # verification/hook subprocess. Values override inherited daemon env; use
  # $PATH in PATH values to prepend/append without losing the host PATH.
  env: {}
  #   PATH: "/custom/bin:$PATH"
  #   MY_VAR: "value"
  # Test verification gate — controls behavior before git push.
  verification:
    timeout_ms: 600000                                     # test timeout (10 min)
    regression_guard: true                                 # compare against baseline; block only net-new failures
    # fallback_test_command: "pytest"                      # explicit fallback, empty = auto-detect
  # Repro-first gate — reproduce the bug before the fix stage may run.
  repro_first:
    enabled: false
    timeout_ms: 900000                                     # reproduction agent budget (15 min)
    command_timeout_ms: 300000                             # reproduction command execution budget
    # labels: [bug]                                        # only issues with these labels; empty = all

# ============================================================================
# Sandbox — execution sandbox and approval policy
# ============================================================================
sandbox:
  thread_sandbox: workspace-write
  turn_timeout_ms: 3600000                                 # 1h per turn
  read_timeout_ms: 5000
  stall_timeout_ms: 300000                                 # 5m idle abort
  approval_policy:
    reject:
      sandbox_approval: true
      rules: true
      mcp_elicitations: true
  # command: ""                                            # sandbox launch command
  # turn_sandbox_policy: {}                                # per-turn sandbox policy

# ============================================================================
# Hooks — shell commands run around each issue run
# ============================================================================
hooks:
  after_create: echo "[orchestrator] workspace created for $ISSUE_IDENTIFIER"
  before_run: echo "[orchestrator] starting work on issue $ISSUE_IDENTIFIER"
  after_run: echo "[orchestrator] finished issue $ISSUE_IDENTIFIER"
  before_remove: echo "[orchestrator] removing workspace $ISSUE_IDENTIFIER"
  # Local test gates — run BEFORE commit/push so CI is not the first line of defense.
  # pre_commit: "ruff check --fix && pytest -x tests/"
  # pre_push: "pytest -x tests/ --timeout=120"
  # post_sync: "pytest tests/e2e/ --timeout=300"
  timeout_ms: 60000

# ============================================================================
# Pull Request Template — fixed structure plus generated implementation notes
# ============================================================================
# `title` and `body` use safe {{ variable }} substitutions (not executable
# template code). Available variables: issue.id, issue.identifier, issue.title,
# issue.url, branch_name, base_branch, commit_sha, verification_status,
# verification_summary, changes_summary, implementation_notes,
# pull_request.number, pull_request.url.
# The dynamic Markdown values are read after the agent finishes from
# `changes_summary.md`, `implementation_notes.md`, and `verification_report.md`.
# Omit this section or leave body empty to retain the built-in PR body.
pr_template:
  title: "{{ issue.identifier }}: {{ issue.title }}"
  body: |
    ## Summary
    {{ changes_summary }}

    ## Implementation Notes
    {{ implementation_notes }}

    ## Verification
    {{ verification_summary }}

    ## Related Issue
    {{ issue.url }}

    ## Checklist

    - [x] Changes are self-reviewed
    - [x] Relevant tests were run

# ============================================================================
# Observability — TUI dashboard
# ============================================================================
observability:
  dashboard_enabled: true
  refresh_ms: 1000
  render_interval_ms: 16

# ============================================================================
# Server — local HTTP API
# ============================================================================
server:
  host: 127.0.0.1
  port: 8765

# ============================================================================
# Review Feedback — PR检视意见自动处理（可选）
# ============================================================================
# 开启后每轮 poll 自动采集 PR 行内检视意见，触发 follow-up agent 修复。
# 必须配合 Issue 评论 /agent follow-up 或 review_feedback 自动检测使用。
review_feedback:
  enabled: false                                           # 开启后自动检测PR检视意见
  mode: manual                                             # auto=自动拉起agent, manual=等待CLI审批
  poll_interval_ms: 60000                                  # 检测间隔
  max_followup_attempts_per_pr: 5                          # 单PR最大followup次数
  include_ci_failures: true                                # 是否包含CI失败
  reply_to_comments: true                                  # 处理后是否回复评论
  # ignore_authors: [ascend-robot]                         # 忽略指定作者（平台/CI机器人）的评论，按实际机器人账号配置
  ignored_comment_commands: [/lgtm, /approve, /approved]  # 仅忽略整条为该命令的评论
  # ignored_feedback_sources: [ci]                         # 采集但不触发指定来源的反馈
  # ignored_body_patterns: ['(?i)^all checks have passed$'] # 整条正文正则匹配
  # max_feedback_items_per_run: 20                         # 每轮最多处理反馈条数
  # max_log_chars_per_check: 12000                         # 单条 CI 日志截断长度
  # pending_feedback_timeout_seconds: 600                  # pending 反馈超时（秒）
  # bot_login: clawcodex-bot                               # 机器人登录名，跳过自己的评论

# ============================================================================
# Verification & Repro First — 已移至 agent 段下（agent.verification / agent.repro_first）
# ============================================================================
# 注意：verification 和 repro_first 必须配置在 agent 段内才能生效。
# 顶层放置会被 schema 忽略（见 config/schema.py 的 WorkflowConfig.from_dict）。

# ============================================================================
# Modes — 多Agent协作模式（可选）
# ============================================================================
# 配置多 Agent 协作模式。省略此段则使用默认 single 模式。
modes:
  enabled: [single]                                        # 注册的模式：single | pipeline | coordinator | debate | swarm
  default: single                                          # 路由失败时的回退模式
  router:                                                  # Issue→模式 路由器
    kind: heuristic                                        # heuristic（关键词）| llm | none
    min_confidence: 0.5                                    # 低于此置信度回退 default
    # model: deepseek-v4-flash                             # kind=llm 时的路由模型
    # endpoint: https://api.deepseek.com/chat/completions  # kind=llm 时的路由端点
    # api_key_env: DEEPSEEK_API_KEY                        # kind=llm 时的 API Key 环境变量名
  pipeline:                                                # pipeline 模式配置
    stages: [analyzer, implementer, tester]
    # handoff: prompt                                       # prompt=注入前序输出 | mailbox=工作区邮箱交接
    # max_retries_per_stage: 1
    # stage_models:                                        # 每阶段独立模型（异构 LLM）
    #   implementer: deepseek-v4
    # stage_max_turns:                                     # 每阶段独立轮数预算
    #   analyzer: 80
  # debate:                                                # debate 模式配置
  #   proposers: [proposer_a, proposer_b]
  #   judge_mode: pick                                     # pick | synthesize
  #   judge_model: deepseek-v4
  #   isolation: reset                                     # reset | worktree | none
  #   parallel: false                                      # 并行提案（需 isolation=worktree）
  # swarm:                                                 # swarm 模式配置
  #   max_subtasks: 8
  #   max_parallel: 3
  #   max_waves: 6

# ============================================================================
# Rules — 从PR检视意见中学习规则（可选）
# ============================================================================
# 自动从 PR review 反馈中提取可复用的编码规则，注入后续 agent prompt。
rules:
  enabled: false
  # path: .clawcodex_rules.md                             # 规则存储路径
  max_rules: 20
  min_confidence: low                                      # low | medium | high

# ============================================================================
# PR Conflict Scan — PR冲突自动检测（可选）
# ============================================================================
# 后台定时扫描 PR 合并状态，检测到冲突时自动触发 rebase。
# 注意：GitCode 不暴露 mergeable 字段，此功能在 GitCode 上无效。
pr_conflict_scan:
  enabled: false
  poll_interval_ms: 300000                                 # 5分钟
  max_rebase_attempts_per_issue: 3
  # max_prs_per_scan: 25                                  # 每轮扫描 PR 上限
  use_force_push: false                                    # 是否使用 force push
  # bot_login: clawcodex-bot                              # 跳过该账号自己的评论
  # scan_states: [open]                                   # 扫描的 PR 状态

# ============================================================================
# Clarifier — Issue 清晰度分析（可选）
# ============================================================================
# 在 issue 分派前分析描述是否足够清晰，不清晰时可自动评论提问。
clarifier:
  enabled: true
  block_on_unclear: true                                  # 不清晰时是否阻止分派（false=放行让 agent 探索）
  author_first: true                                       # 优先询问 Issue 作者
  max_questions: 3
  max_rounds: 2
  min_confidence: 0.7
  # max_input_tokens: 6000                                # 分析输入 token 上限
  # max_output_tokens: 800                                # 分析输出 token 上限
  # fail_open: true                                       # LLM 不可用时放行（不卡住流水线）
  # cache_enabled: true                                   # 缓存分析结果（按 Issue 内容指纹）
  # max_analyses_per_poll: 4                              # 每轮 poll 最多分析数
  # workspace_focus_enabled: false                        # follow-up 时用工作区改动富化上下文
  # remote_label: "needs-clarification"                    # 可选远端标签

# ============================================================================
# Worker — 分布式Worker（可选）
# ============================================================================
# 配置 SSH 远程 worker 主机。省略则所有 agent 在本地运行。
worker:
  ssh_hosts: []                                          # SSH 主机列表
  max_concurrent_agents_per_host: 2                      # 单机最大并发 agent
---

# Orchestrator Agent Prompt

You are an autonomous development agent working on an issue from the
**{{REPO_NAME}}** repository ({{REPO_URL}}). The issue has already been
assigned to you.

**Current state (do NOT repeat these steps):**

- The issue description below has been fetched from the issue tracker API.
- The repository has been cloned (depth=1) into your workspace.
- A working branch `{{BRANCH_PREFIX}}/<issue-id>` has been checked out from the
  base branch (default: `main`).

**Your task:**

1. Read and understand the issue description, labels, and any comments.
2. Explore the {{REPO_NAME}} codebase as needed (Read, Grep, Glob, Bash).
3. Implement the required changes against the base branch baseline.
4. Run the existing test suite (and add new tests if behavior is non-trivial).
   Before committing, write `changes_summary.md` and `implementation_notes.md`.
   Describe only real changes, tests run, and known limitations; do not copy the
   diff or claim verification that was not executed.
5. Commit your changes with a descriptive message
   (`feat: ...` / `fix: ...` / `refactor: ...`).
6. Push the branch and open a pull request back to the base branch.

**Coding conventions:**

- Follow the existing module structure of {{REPO_NAME}}.
- Keep changes focused on the issue; do not bundle unrelated cleanups.
- Add or update tests alongside any behavioral change.
- Use Conventional Commits for the commit message.

**Issue:** {{ issue.identifier }} - {{ issue.title }}
{% if issue.description %}
**Description:**
{{ issue.description }}
{% endif %}
{% if issue.labels %}
**Labels:** {{ issue.labels | join(", ") }}
{% endif %}

Work directly in the workspace. Make the necessary code changes, commit, and
push — the orchestrator will then create the pull request.
