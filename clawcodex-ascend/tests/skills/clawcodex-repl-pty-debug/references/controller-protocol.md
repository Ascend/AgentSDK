# Controller Protocol

Use this reference for JSONL operations, driver usage, input semantics, and
artifact review.

## Launch

Persistent controller:

```bash
env UV_SKIP_WHEEL_FILENAME_CHECK=1 uv --cache-dir "${TMPDIR:-/tmp}/clawcodex-uv-cache" run --extra dev --frozen python scripts/debug/repl_pty_session.py interactive
```

Reusable temporary JSONL driver for hosts without a persistent PTY session API:

```bash
artifact_root="${TMPDIR:-/tmp}/clawcodex-pty-loop-<run-id>"
.venv/bin/python3 .agents/skills/clawcodex-repl-pty-debug/scripts/pty_jsonl_driver.py \
  --repo-root "$(pwd)" \
  --artifact-root "$artifact_root" \
  --ops "$artifact_root/ops.jsonl"
```

If the driver prints
`pty_jsonl_driver: controller exited before writing a response`, classify the
appended stderr first. Missing script paths, dependency resolution, and sandbox
launch failures are controller launch mechanics, not REPL behavior.

Do not import `scripts/debug/repl_pty_session.py` as an API. Do not replace the
controller with repeated one-shot shell commands.

## Adaptive Turns

For adaptive multi-turn debugging, keep the `interactive` controller open. Send
one JSON operation, read the JSON response, then choose the next `send`, `observe`, or `key` only after reading the previous JSON response. Use
the current response `delta`, `kind`, `state`, `error_kind`, and `signals` as
the active decision surface. `screen` is cumulative terminal state; it can still
contain a stale permission menu after the child has already resolved it, so use
`screen` only for context or completed-transcript audit.
Prefer direct `start`/`send`/`observe`/`stop` controller operations before inspecting helper implementation. Read `pty_jsonl_driver.py` or `clawcodex_ext/debug/repl_pty_session.py` only when a controller failure specifically implicates launch mechanics or helper behavior.

A prewritten `ops.jsonl` is fixed-script evidence, not adaptive multi-turn proof.
Use the temporary driver only for smoke tests, deterministic regression steps,
or hosts that cannot keep a persistent controller session.

For local deterministic regression without a live provider, use
`run_adaptive_jsonl`: pass a first request plus a callback that receives each
JSON response and returns the next request. This proves the next operation was
chosen after reading the previous response, unlike a prewritten `StringIO` or
`ops.jsonl` sequence.

The callback may also inspect independent evidence before choosing the next
request, such as a file, SQLite row, or artifact that the child just reported or
wrote. In reports, record the order as response or artifact read, decision, then next send/key/observe.
For `run_adaptive_jsonl` and `pty_adaptive_driver.py`, return `None` only when
no more controller operations are needed. If the child needs more time, return
an explicit `{"op":"observe","timeout":...}` request.
If a probing operation is expected to fail before recovery, add
`"allow_error":true` only to that request. The response still has
`"ok":false`, `event:error`, and `error_kind`, but the controller can return
success after later adaptive repair turns pass. Do not use it for unexpected
failures.
Shell-only agents should run `scripts/pty_adaptive_driver.py` with a small decider file instead of copying `pty_jsonl_driver.py` or rewriting controller stdin/stdout plumbing. Decider files can `import decider_helpers`; use helpers such as `has_current_permission_prompt(response)`, `has_current_text(response, ...)`, `bash_exit_code(response)`, and `bash_succeeded(response)` instead of hand-rolled checks against cumulative `screen`. Decider requests must use `"op"`, not `"action"`. `first_request()` must return `{"op":"start",...}` because the helper owns a fresh controller; put the first B user prompt in `decide_next(response)` after the `ready` response. `send` text is ClawCodex user input, not a shell command; prompt B as an agent and verify the filesystem before the next turn. The helper writes both controller JSONL and `decider_request` audit events to `adaptive-driver.jsonl` in the artifact dir before final `result.json` exists.
After a `stopped` response, return `None`; do not return a second `stop`. The helper records a `decider_warning` and ignores one duplicate `stop` after `stopped` so successful runs do not end with `session has not been started` noise. Any other operation after `stopped` should be treated as a decider bug.
Use JSON input for direct skill invocation: `/tool Skill {"skill":"clawcodex-repl-pty-debug"}`. `/tool Skill clawcodex-repl-pty-debug` is invalid because the REPL parses the third field as JSON.

Minimal decider shape:

```python
def first_request():
    return {"op": "start", "label": "start B"}

def decide_next(response):
    if response.get("event") == "stopped":
        return None
    if response.get("event") == "ready":
        return {"op": "send", "text": "do the task", "label": "turn1"}
    if response.get("label") == "turn1":
        # Inspect response fields and any filesystem artifact before deciding.
        return {"op": "observe", "timeout": 30, "label": "settle"}
    return None
```

Useful `response` fields include `event`, `op`, `label`, `kind`, `state`,
`error_kind`, `signals`, `artifact_dir`, `step`, and `input_source`. In
`adaptive-driver.jsonl`, controller responses show what B returned; following
`decider_request` lines show which response fields the decider used as `basis`
and what request it sent next.
When A launches a nested B through `pty_adaptive_driver.sh`, a successful nested
run can surface as `PTY_ADAPTIVE_DRIVER_EXIT=0`, as a Bash tool-result JSON with
`"exit_code": 0`, or as B's `result.json` with `"ok": true`; any one of those is
sufficient if the artifact evidence matches the task.

## Operations

```json
{"op":"start","label":"start"}
{"op":"send","text":"/help","expect":"Available Commands:","label":"help"}
{"op":"send","text":"/tools","expect":"Available tools:","label":"tools"}
{"op":"send","text":"/goal","expect":"No goal set","label":"goal status"}
{"op":"send","text":"/cost","expect":"Total units:","label":"cost"}
{"op":"observe","timeout":2.0,"label":"settle"}
{"op":"stop","label":"stop"}
{"op":"exit","label":"exit controller"}
```

With no custom `start.cmd`, the child uses the default command with
`--permission-mode bypassPermissions`. Override `start.cmd` and omit that flag
only when permission prompts are the behavior under test.

`send` submits a complete line with Enter. Multiline `send.text` and
`send.text_file` are folded into one REPL input line so a long prompt does not
become multiple unintended user turns. Use `send` for chat prompts and slash
commands. Use `key` or `raw` for prompt-toolkit menus or literal newline tests
because they send exactly the provided bytes:

```json
{"op":"key","text":"1","timeout":1,"label":"quick select"}
{"op":"key","text":"\r","timeout":1,"label":"accept current item"}
{"op":"key","text":"\u001b","timeout":1,"label":"escape"}
```

`key` and `raw` send bytes; they do not prove the resulting tool action has
finished. If their response does not already contain the expected tool result or
state transition, return a follow-up `observe` before deciding the next turn.
After `key` or `raw`, follow with `observe` unless that response already
contains the expected tool result.

A permission prompt's `[y]` and `[n]` text can be labels, not key bindings. If a
raw digit only echoes and no tool result follows, classify permission-prompt
handling instead of blindly retrying.
For denial-and-recovery tests, deny with `key` or `raw`, observe until the REPL
is idle, independently verify that the denied file or state was not created,
then choose the next repair or allow turn from that evidence.

## Long Prompts

PTY line discipline can truncate very long single-line JSON before the
controller sees a newline. For long prompts, write the prompt to a run-scoped
file and send a short `text_file` command:

```json
{"op":"send","text_file":"<artifact-root>/prompt.txt","timeout":180,"max_output_chars":4000,"label":"long prompt"}
{"op":"observe","timeout":30,"max_output_chars":4000,"label":"settle"}
```

The controller folds newlines from `text` and `text_file` into one REPL input line. It
echoes `op`, `label`, and `input_source` in stdout and persists them in
`result.json` events. `max_output_chars` truncates only stdout `delta` and
`screen`; `raw.log`, `clean.txt`, and `result.json` remain complete.
Child input is written in bounded chunks, so a too-large prompt should return a
write timeout instead of hanging the controller. If that occurs, split the task
into smaller adaptive turns and keep using `text_file` for each long turn.

## Artifacts

Stop the child REPL before auditing:

```json
{"op":"stop","label":"stop child"}
```

Read artifacts in this order:

1. `result.json` for `ok`, `error`, and failed step.
2. `clean.txt` for human-readable transcript.
3. `raw.log` only for ANSI, cursor, or redraw bugs.

Quick audit:

```bash
jq '{ok,error,events:[.events[] | {event,ok,kind,state,error_kind,signals,op,label,input_source}]}' /path/to/result.json
```

For active goal summaries, `/goal` may report `Tokens used:`. Treat that as a
different stable output from a fresh `/goal` response.

When counting nested child sessions, inspect where `result.json`, `clean.txt`,
and `raw.log` were written. No child directories under that artifact root can mean one child session when the helper writes directly to the requested artifact directory. `session-*` child directories mean the child was restarted.
