---
name: clawcodex-repl-pty-debug
description: Use when an external agent needs to debug real ClawCodex REPL terminal behavior, same-session multi-turn flows, slash-command rendering, permission prompts, live-provider boundaries, or nested ClawCodex PTY evidence.
---

# ClawCodex REPL PTY Debugging

Use this skill when the real terminal matters: prompt rendering, slash
commands, streaming redraws, permission menus, live-provider boundaries, or
multi-turn REPL state. PTY is high fidelity, not the default debug surface.

## First Decision

Start below PTY unless terminal behavior is the issue. Use `QueryRunner`,
headless tests, `clawcodex -p --output-format stream-json`, or
`CLAW_HEADLESS_BACKEND=stub` for lower-fidelity checks. Use PTY for real REPL
state, rendered slash commands, permission menus, streaming redraws, or
same-session behavior. Change code only for root cause with focused
regression proof.

## Workflow

Run the lowest-fidelity proof that matches the issue class. When PTY is needed,
drive one persistent controller session with labeled JSONL operations, use
`text_file` for long prompts, then stop the child and inspect `result.json`,
`clean.txt`, and only then `raw.log`. If the run creates files, independently
verify them and clean only paths explicitly created by this run.

## Adaptive Multi-turn

Adaptive multi-turn runs must use the persistent `interactive` controller. Send
one operation, read its JSON response, inspect `delta`, `kind`, `state`,
`error_kind`, and `signals`, then choose the next `send`, `observe`, or `key`.
Use the current response `delta`, `kind`, `state`, and `signals` for active
prompt detection. `screen` is cumulative terminal state and may contain stale
menus from earlier turns; use it only as context or when explicitly auditing a
completed transcript.
When the child creates files or state, independently inspect that filesystem or
SQLite evidence before returning the next operation.
After loading this skill, do not reread large repo files just to rediscover the workflow.

Invoke skills with JSON input, for example
`/tool Skill {"skill":"clawcodex-repl-pty-debug"}`. The shorthand
`/tool Skill clawcodex-repl-pty-debug` is invalid because `/tool` parses the
third field as JSON.

After `/tool Skill {"skill":"..."}`, treat the returned prompt as already
loaded guidance. Do not read `tests/skills`, `.claude/skills`, bundled scripts,
or `clawcodex_ext/debug/repl_pty_session.py` merely to rediscover the workflow.
Use the `skillRoot` path returned by `/tool Skill`; do not Glob/search for skill
copies. Mirror directories are not authoritative.

If the task already provides an exact helper command, run that command directly;
do not `ls`, Glob, or read the skill directory merely to verify the helper
exists. Treat any `ls`, `find`, Glob, Read, or Grep under `.agents/skills`,
`.claude/skills`, `tests/skills`, bundled skill dirs, or helper implementation
files as a failed no-discovery run when an exact helper command was already
provided. No-discovery applies to the outer A agent's own tool calls. Checking
only `adaptive-driver.jsonl` is insufficient.

After outer A stops, prefer the host-side audit helper instead of ad hoc grep:
`.agents/skills/clawcodex-repl-pty-debug/scripts/audit_outer_transcript.py --json <outer-clean.txt> [outer-result.json raw.log]`.
Add `--require-adaptive-order` when the run must prove observe/read -> decision
-> next send/key/observe ordering from the outer transcript.
When reporting no-discovery, audit the outer transcript, outer `result.json`,
outer `clean.txt`, outer `raw.log`, or the controlling agent tool log; if A used
forbidden discovery, do not mark it passed.

## Permission Mode

Default child PTY starts use `--permission-mode bypassPermissions`. Every run plan must state whether permission prompts are in scope.

Use the default command with `--permission-mode bypassPermissions` for PTY,
skill loading, nested orchestration, file creation, or happy paths without
permission UI. For `Permission Required`, approval keys, denial, sandbox
policy, or prompt classification, override `start.cmd`, omit
`--permission-mode`, and handle the menu with `key` or `raw`.

Use `--permission-mode bypassPermissions` only when permission prompts are out of scope. Do not use `bypassPermissions` when permission prompts are in scope.

## Live Provider Data Boundary

Sandbox or filesystem approval is not live-provider data approval. An instruction such as "allow unsandboxed execution" is still insufficient for natural-language live-provider turns. Run that path only if the user explicitly allows sending workspace, skill, and prompt context to the external model provider; otherwise run fake/stub/local slash-command checks and mark the live-provider portion blocked by policy.

## Start The Controller

Preferred persistent session: `scripts/debug/repl_pty_session.py interactive`
through the repo `uv` command. If the host has no persistent PTY session API,
use `.agents/skills/clawcodex-repl-pty-debug/scripts/pty_jsonl_driver.py` with
`--repo-root`, `--artifact-root`, and `--ops`.

Use `pty_adaptive_driver.py` when a shell-only agent needs adaptive decisions
without rewriting controller plumbing. If the Bash surface suppresses direct
Python stdout/stderr or the agent needs a visible exit/artifact summary, use
`scripts/pty_adaptive_driver.sh` with the same arguments.
Decider files can `import decider_helpers` from the skill scripts directory.
Prefer `has_current_permission_prompt(response)`, `has_current_text(response,
...)`, `bash_exit_code(response)`, and `bash_succeeded(response)` over ad hoc
checks against cumulative `screen`.
Returned requests must use `"op"`, not `"action"`. Return `None` only when no
more controller operations are needed; return `{"op":"observe","timeout":...}`
when B needs more time. Do not stop and restart B between turns unless restart
behavior is the explicit target. Do not stop B just because the final file
exists. If a deliberate probe may fail before recovery, put `"allow_error":true`
on that one request; the error remains in artifacts but does not fail the run if
later turns recover. If required evidence is missing, send a repair prompt based
on the missing evidence. Do not read helper implementation files unless the run has
already been classified as a helper-layer failure and no-discovery failure.
After any `key` or `raw` menu action, insert an `observe` settle step before a
subsequent `send`, even if the key response already contains a tool result. A
key response that is enough for stopping or reporting can still be terminal, but
starting another REPL input immediately after menu teardown is a race-prone
driver pattern.
The helper writes `adaptive-driver.jsonl`; inspect it if the outer agent times
out before `result.json` exists. If the wrapper exits nonzero, inspect
`driver-error.json` first for `stage`, `error`, and `decider`, then use
`adaptive-driver.jsonl` to see the last controller response and decider request.
When A runs the shell wrapper around B, treat wrapper success as proven by
`PTY_ADAPTIVE_DRIVER_EXIT=0`, a Bash tool result with `"exit_code": 0`, or B's
`result.json` containing `"ok": true`; do not fail a good run only because one
surface did not echo the expected marker.

Shell-visible wrapper: run
`bash .agents/skills/clawcodex-repl-pty-debug/scripts/pty_adaptive_driver.sh
--repo-root <repo> --artifact-dir <artifacts> --decider <decider.py>`.

Do not import `scripts/debug/repl_pty_session.py` as an API, and do not use
repeated one-shot shell commands to simulate a session.
Multiline `send.text` and `send.text_file` are folded into one REPL input line;
use `raw` when literal newlines are the behavior under test.

## References

Read only the reference needed now: `references/controller-protocol.md` for
JSONL/controller details, `references/failure-classification.md` for fake/stub
and error routing, `references/live-provider-and-goals.md` for provider/data
boundaries, and `references/nested-skill-validation.md` for nested skill proof.

## Rules

- Stop escalating fidelity once the current layer answers the question.
- Treat slash-command success and live-provider success as separate checkpoints.
- Use `key` or `raw` for prompt-toolkit menus; `send` always submits a full line.
- After `key` or `raw`, follow with `observe` before any later `send`. Skip that
  settle only when the key/raw response already contains the expected result and
  the next operation is `stop` or no further controller operation is needed.
- For denial probes, omit `--permission-mode bypassPermissions`, use `key` or
  `raw` to deny, verify the requested file/state is absent, then choose the next
  repair or allow turn from that evidence.
- Keep `--agent-debug` enabled so state paths such as `CLAWCODEX_HOME`,
  sessions, history, and telemetry stay under the artifact state directory.
- Do not trust role wording as multi-agent proof; require actual tool markers or
  artifact evidence.
