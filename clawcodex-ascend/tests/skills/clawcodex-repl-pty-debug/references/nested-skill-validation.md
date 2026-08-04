# Nested Skill Validation

Use this reference for ClawCodex loading this skill, nested ClawCodex runs, and
multi-agent/coordinator proof.

## Managed Skill Source

When validating this repo-local skill from a child ClawCodex REPL, make the
skill source explicit. Do not rely on `.claude/skills` during PTY debugging:
ignored user-local copies can be stale and shadow `.agents/skills`.

For managed-skill and nested happy-path runs, bypass unrelated local tool
permission UI. Start the child with the current repo skill directory:

```json
{
  "op": "start",
  "cmd": [".venv/bin/python3", "-P", "-m", "clawcodex_ext.cli.main", "--legacy-repl", "--stream", "--agent-debug", "--permission-mode", "bypassPermissions"],
  "env": {"CLAWCODEX_MANAGED_SKILLS_DIR": "<repo-root>/.agents/skills"},
  "label": "managed skill child"
}
```

Default permission mode keeps `Permission Required` prompts visible. Use this
variant when validating permission detection, key/raw approval, denial, or
sandbox policy:

```json
{
  "op": "start",
  "cmd": [".venv/bin/python3", "-P", "-m", "clawcodex_ext.cli.main", "--legacy-repl", "--stream", "--agent-debug"],
  "env": {"CLAWCODEX_MANAGED_SKILLS_DIR": "<repo-root>/.agents/skills"},
  "label": "permission-probe child"
}
```

For `/tool Skill` proofs, verify the transcript or JSON output reports
`"loadedFrom": "managed"` and the expected `skillRoot`. If it loads from
`.claude/skills`, classify that as skill-source isolation evidence instead of
editing or deleting the user's local copy.

## Nested ClawCodex

For nested ClawCodex validation, prove the outer ClawCodex used this skill
before starting the inner ClawCodex. A valid chain is:

1. Codex controls outer ClawCodex through the PTY controller.
2. Outer ClawCodex invokes the actual `Skill` tool or `/tool Skill` for
   `clawcodex-repl-pty-debug`.
3. Outer ClawCodex uses that guidance to launch or drive an inner ClawCodex.
4. The report records both outer and inner artifact roots.

Accept the nested run only if the outer transcript contains an actual `Skill` tool call
or `/tool Skill` output plus inner `result.json`, `clean.txt`, and `raw.log`.
Directly reading `tests/skills/.../SKILL.md` is fallback evidence, not proof
that the skill mechanism was exercised.

Nested adaptive B turns must continue on the same inner controller and child
REPL after each observe/read/decision step. Multiple inner artifact directories for ordinary B turns means the run restarted B, so report it as a workflow finding instead of accepting it as same-session multi-turn proof.

## Coordinator And Multi-agent Proof

For coordinator-mode validation, pass the env var through `start.env`:

```json
{
  "op": "start",
  "cmd": [".venv/bin/python3", "-P", "-m", "clawcodex_ext.cli.main", "--legacy-repl", "--stream", "--agent-debug", "--permission-mode", "bypassPermissions"],
  "env": {"CLAUDE_CODE_COORDINATOR_MODE": "1"},
  "label": "coordinator child"
}
```

The coordinator example uses bypass mode because permission prompts are not the
subject of that proof. Omit `--permission-mode` when coordinator permission
behavior is the target.

Verify coordinator mode from the rendered `Coordinator Mode ACTIVE` banner, then
prove real delegation from actual tool output such as `Agent (@worker ...)`,
`TeamCreate`, `TaskOutput`, or worker results in `clean.txt` / `result.json`.
Planner/executor/verifier wording alone is role play, not multi-agent evidence.
