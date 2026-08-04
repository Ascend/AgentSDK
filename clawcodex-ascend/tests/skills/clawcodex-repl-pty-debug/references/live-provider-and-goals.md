# Live Provider And Goals

Use this reference when the acceptance requires a real model response,
provider/network access, active-goal continuation, or goal-runtime artifact
proof.

## Sandbox Boundary

Fake child and local slash-command checks can run inside the normal sandbox.
Live-provider turns should use a network-capable controller path. The provider
request is made by the child REPL process, so a sandboxed controller can fail
with DNS or HTTPS errors even when the user's machine and provider config are
healthy.

Treat these as live-provider requests:

- natural-language prompts that should be answered by the configured model
- active-goal continuation after `/goal <objective>`
- explicit "real API", "live provider", or "credentials/network are available"
- any test whose success depends on assistant output rather than local
  slash-command output

Before requesting a network-capable or unsandboxed path, state that the run may
send workspace, skill, or prompt context to the external model provider. Continue
only with explicit user approval for that data boundary. If approval is missing
or denied, run fake/stub/local slash-command checks only and mark the
live-provider portion blocked by environment/security policy.

Ask for the required escalation up front. If approval is not granted, run only
fake/stub/local slash-command checks and mark the live-provider portion blocked
by environment.

## Local Slash Commands Versus Provider Turns

Slash commands such as `/help`, `/tools`, fresh `/goal`, and `/cost` are local
checkpoints. They can pass even when the next natural-language turn fails with
`network_error`.

Use local proof:

```json
{"op":"send","text":"/help","expect":"Available Commands:","label":"help"}
{"op":"send","text":"/tools","expect":"Available tools:","label":"tools"}
{"op":"send","text":"/goal","expect":"No goal set","label":"fresh goal"}
{"op":"send","text":"/cost","expect":"Total units:","label":"cost"}
```

Use live-provider proof only where network is allowed:

```json
{"op":"send","text":"/goal verify live continuation","expect":"Goal set: verify live continuation","label":"set goal"}
{"op":"observe","timeout":30,"label":"provider continuation"}
```

`/goal <objective>` can print local success and then immediately start an
assistant turn. In a network-restricted sandbox, the continuation may fail even
though the slash command succeeded.

## Goal Artifacts

When proving real goal-tool execution, prefer transcript tool markers plus the
artifact state database over role wording. In debug mode, `--agent-debug`
redirects state paths such as `CLAWCODEX_HOME`, sessions, history, telemetry,
and goal storage into the artifact state directory.

Reliable evidence includes:

- `clean.txt` containing actual `create_goal`, `get_goal`, or `update_goal`
  tool output
- `state/goals_1.sqlite` showing the durable goal status and accounting
- `result.json` showing the step that triggered the goal flow

If setting `/goal <objective>` prints `attempt to write a readonly database`,
the child probably used an older/custom harness that still points goal storage
at the real `~/.clawcodex` database.

For active goals, steer the model to use the existing goal directly. Do not let
it create subgoals unless that is the behavior under test.
