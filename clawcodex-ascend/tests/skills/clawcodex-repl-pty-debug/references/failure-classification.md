# Failure Classification

Use this reference when deciding whether a problem belongs to the skill, PTY
adapter, REPL runtime, provider/network, sandbox, or one-off test plan.

## Edit Policy

For continuous PTY improvement, default to reporting only until the failure
layer is proven.

| Root cause | Action |
| --- | --- |
| Runtime, REPL, or PTY adapter behavior is wrong and a focused regression can show it | Change the smallest code layer and add or extend focused tests |
| Existing code is unreasonable, over-specialized, or obscures the root cause | Optimize the narrow code path with a regression; keep the change behavior-preserving unless the bug requires behavior change |
| This skill is misleading, stale, or missing reusable instructions that caused repeated agent mistakes | Edit the skill and its tracked fixture |
| A one-off expect string, test-plan typo, sandbox/network limit, or single-run prompt issue explains the failure | Do not edit the skill; record it in the round report and next regression list |

Every report should state why the finding is or is not a code defect and why a
skill change is or is not warranted.

## Isolation Ladder

For behavior that does not require terminal rendering, use the stub path:

```bash
env CLAW_HEADLESS_BACKEND=stub .venv/bin/python3 -c "from pathlib import Path; import io; from extensions.capabilities.headless_runner import HeadlessSessionOptions, run_headless_session; events=[]; buf=io.StringIO(); code=run_headless_session(HeadlessSessionOptions(prompt='test', workspace_root=Path.cwd(), stdout=buf, stderr=buf, on_event=events.append)); print({'code': code, 'events': len(events)})"
```

For terminal-specific behavior, prove the controller with the fake child before
using a live provider:

```bash
env UV_SKIP_WHEEL_FILENAME_CHECK=1 uv --cache-dir "${TMPDIR:-/tmp}/clawcodex-uv-cache" run --extra dev --frozen python scripts/debug/repl_pty_session.py run-script --timeout 5 --cmd .venv/bin/python3 tests/debug/fake_repl_child.py
```

If this smoke fails because `uv` tries to resolve dependencies and DNS/package
access is blocked, classify that as sandbox dependency resolution before
changing ClawCodex or the controller.

## Common Signatures

| Signature | Classification |
| --- | --- |
| Local `/help`, `/tools`, `/goal` work, natural language fails with `httpcore.ConnectError`, `nodename nor servname provided`, `APIConnectionError`, or `Connection error` | `network_error`; retest outside network-restricted sandbox before blaming provider credentials |
| Rendered provider traceback, API auth, quota, or model failure | `provider_error`; report exact text from `result.json` or `clean.txt` |
| Permission screen remains after a raw digit echoes but no tool result follows | permission-prompt handling; use `key`/`raw`, not `send` |
| A run intended to validate permission prompts used `--permission-mode bypassPermissions` | test-plan error; rerun in default permission mode before changing PTY handling |
| Write/read proof says path outside allowed working directories after approval | filesystem path policy, not permission prompt failure |
| `Read` returns `"type": "file_unchanged"` after same-session `Write` | successful write/read pairing; independently read the filesystem if bytes matter |
| Fake child echoes `/help` and times out waiting for `Available Commands:` | test-plan mismatch; fake child proves controller/PTY mechanics, not real slash-command rendering |
| Later `result.json` reports an earlier failure | stale artifact/session state; ensure each child start gets its own artifact directory |
| Focused pytest node is not found | stale or guessed test node; confirm the exact node before treating it as product evidence |

Never treat model role wording as proof of tools, goals, or multi-agent
execution. Trust transcript tool markers, artifact files, and independent
filesystem or SQLite checks.
