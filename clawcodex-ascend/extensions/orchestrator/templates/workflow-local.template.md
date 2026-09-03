---
# =============================================================================
# ClawCodex Orchestrator — Local-Tracker Workflow Template
# =============================================================================
# This template is for the "local file-based issue tracking" scenario:
#   - tracker.kind = local
#   - issues are stored as <issues_path>/*.md with frontmatter + body
#   - workspace.repo_clone_url points to the target repo (orchestrator fresh-clones)
#   - local tracker triggers no_push; commits stay in the workspace branch
#   - after manual review, pull the branch from workspace into the target repo
#
# Usage:
#   clawcodex orchestrator workflow init --template workflow-local
#
# Placeholders (replace before starting, or use env references):
#   {{OWNER}}         Repository owner
#   {{REPO}}          Repository name
#   {{REPO_URL}}      Repository clone URL (https / ssh / file)
#   {{BRANCH_PREFIX}} Issue branch prefix, e.g. feature / fix
#   {{REVIEW_REMOTE}} Remote name for post_sync review push (default: origin)
#   {{REVIEW_PREFIX}} Review branch prefix (default: review)
# =============================================================================

tracker:
  kind: local
  issues_path: {{ISSUES_PATH}}            # e.g. $HOME/projects/{{REPO}}/.issues
  assignee: {{OWNER}}                     # informational only, not used for polling filter
  branch_prefix: {{BRANCH_PREFIX}}        # generated branches: {{BRANCH_PREFIX}}/<id>-<slug>
  active_states:
    - open
    - ready
  terminal_states:
    - completed
    - closed
    - cancelled
    - failed
    - abandoned

# -----------------------------------------------------------------------------
# Polling: scan the issues directory every 30 seconds
# -----------------------------------------------------------------------------
polling:
  interval_ms: 30000

# -----------------------------------------------------------------------------
# Workspace: fresh-clone the target repo; issue branches are committed in the clone
# -----------------------------------------------------------------------------
# Note: workspace is an independent working tree, NOT your current project directory.
# All branches and commits live inside this clone's .git/.
# After review, use git push / cherry-pick / pull to move changes into the main project.
workspace:
  root: {{WORKSPACE_ROOT}}                # e.g. $HOME/.cache/clawcodex-workspaces
  repo_clone_url: {{REPO_URL}}            # e.g. https://gitcode.com/{{OWNER}}/{{REPO}}.git
  clone_depth: 1
  checkout_issue_branch: true
  git_username: {{OWNER}}
  git_token: $GITCODE_TOKEN               # use env var to avoid plaintext secrets
  gitignore_patterns:
    - .reports
    - "*.pyc"
    - __pycache__
    - "*.egg-info"
    - .pytest_cache
    - .mypy_cache
    - .ruff_cache
    - "*.log"
    - analysis.md
    - changes_summary.md
    - implementation_notes.md
    - verification_report.md
    - ".issues/*.comments.ndjson"

# -----------------------------------------------------------------------------
# Agent: ClawCodex execution parameters
# -----------------------------------------------------------------------------
agent:
  max_concurrent_agents: 1              # LocalTracker shares ProgressReporter; keep at 1
  max_turns: 200
  max_retry_attempts: 6
  max_retry_backoff_ms: 300000
  max_concurrent_agents_by_state:
    open: 1
    ready: 1
  provider: anthropic
  # Leave blank to let schema.py auto-promote dontAsk → bypassPermissions (headless safe)
  permission_mode: bypassPermissions
  # Verification trio. Empty string = skip that step.
  test_command: "{{TEST_COMMAND}}"        # e.g. "python3 -m pytest tests/test_orchestrator_*.py -q"
  build_command: ""
  lint_command: ""
  verification:
    timeout_ms: 600000
  # Max auto-retries per issue (agent:retry disabled after this)
  max_retries_per_issue: 3
  allow_anyone_to_retry: false

sandbox:
  thread_sandbox: workspace-write
  turn_timeout_ms: 3600000
  read_timeout_ms: 5000
  stall_timeout_ms: 300000
  approval_policy:
    reject:
      sandbox_approval: true
      rules: true
      mcp_elicitations: true

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
# - before_run / after_run: lightweight notifications at start / end
# - pre_commit / pre_push / post_sync: sync hooks; empty string = skip
# - post_sync here pushes the branch to {{REVIEW_REMOTE}} as review/<id> for
#   manual review. Push does not modify the working tree, so the post_sync
#   dirty check will not trigger. Leave post_sync empty to keep branches local only.
hooks:
  before_run: "echo '[orchestrator] starting $ISSUE_IDENTIFIER'"
  after_run:  "echo '[orchestrator] finished $ISSUE_IDENTIFIER'"
  pre_commit: ""
  pre_push: ""
  post_sync: "git push -u {{REVIEW_REMOTE}} HEAD:{{REVIEW_PREFIX}}/$ISSUE_IDENTIFIER"
  timeout_ms: 120000

# -----------------------------------------------------------------------------
# Pull request template — used when this workflow is switched to a remote tracker
# -----------------------------------------------------------------------------
# Safe variables include issue.identifier, issue.title, issue.url,
# changes_summary, implementation_notes, and verification_summary.
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

# -----------------------------------------------------------------------------
# Review feedback: pull PR review comments to trigger follow-up
# No real PRs in local mode; leave disabled. Set enabled=true and configure
# mode to turn on.
# -----------------------------------------------------------------------------
review_feedback:
  enabled: false
  mode: manual
  poll_interval_ms: 60000
  max_feedback_items_per_run: 20
  include_ci_failures: true
  reply_to_comments: true
  ignore_authors: []
  ignored_comment_commands: [/lgtm, /approve, /approved]
  ignored_feedback_sources: []
  ignored_body_patterns: []
  max_log_chars_per_check: 12000
  max_followup_attempts_per_pr: 5

# -----------------------------------------------------------------------------
# Observability / Server
# -----------------------------------------------------------------------------
observability:
  dashboard_enabled: true
  refresh_ms: 1000
  render_interval_ms: 16

server:
  host: 127.0.0.1
  port: 8765
---

# Orchestrator Agent Prompt

You are working on a sub-feature for the **{{REPO_URL}}** repository,
defined by a local issue card.

**Current state (do NOT repeat these steps):**

- Issue frontmatter and body have been parsed from `<ISSUES_PATH>/<id>.md`
- The repository has been fresh-cloned into the current working directory (`<WORKSPACE_ROOT>/<safe_id>/`)
- You are on the `{{ issue.branch_name }}` branch, based on `{{ issue.base_branch or 'default branch' }}`

**Your task:**

1. Read the issue card (frontmatter + body), focusing on "acceptance criteria" and "do not do"
2. Run `git status && git log --oneline -5` to confirm the baseline
3. Explore the codebase, locate files that need changes
4. Implement changes, add / update tests
5. Run `agent.test_command` (configured in the workflow) and ensure it passes
6. Before committing, write `changes_summary.md` and `implementation_notes.md`
   with the actual changes, tests run, and known limitations. Do not claim
   verification that was not executed.
7. **Do NOT push, do NOT open a PR, do NOT merge** — the orchestrator handles sync
8. Use Conventional Commits style (`feat:` / `fix:` / `refactor:` etc.)

**Constraints:**

- Only modify files inside the working tree; do not touch files outside the workspace root
- Do not create new repos, modify `.git/config`, or force push
- Do not change the `TrackerAdapter` interface in `extensions/orchestrator/tracker.py`
- Keep changes focused on the current issue; do not clean up unrelated code

**Issue:** {{ issue.identifier }} — {{ issue.title }}
{% if issue.description %}
**Description:**
{{ issue.description }}
{% endif %}
{% if issue.labels %}
**Labels:** {{ issue.labels | join(", ") }}
{% endif %}
{% if issue.base_branch %}
**Base branch:** `{{ issue.base_branch }}`
{% endif %}
{% if issue.priority is not none %}
**Priority:** P{{ issue.priority }}
{% endif %}

The working directory is the root of the issue branch. Make your changes, commit,
and leave the rest to the orchestrator.
