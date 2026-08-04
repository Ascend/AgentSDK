---
name: spec-audit
description: Run an unattended, evidence-backed audit of a repository against explicitly supplied specifications. Use for spec compliance, implementation-gap, and specification-conflict analysis.
---

# Spec Audit

Audit specifications against a repository without changing either input. Produce
one human-readable `report.md` and one Markdown dossier per supported problem in
`findings/`.

## Unattended contract

Never ask a question after invocation. Supplied inputs are already confirmed.
Use repository=current directory and output=`./spec-audit-report/` when omitted.
If no specification was supplied, stop with
`Audit not started: no authoritative specification supplied.` Invalid inputs
also stop with a concrete no-start reason; never guess a substitute.

Do not edit the target, install or repair anything, call an older Spec-Audit,
use a plugin/MCP/nested model launcher, select a model, or use tests, Oracles,
old reports, memory, and unpinned documents as audit evidence. Inherit one host
model policy for the whole run. A later invocation always starts over.

Resolve `<skill-root>` as this file's directory. Run helpers only as:

`python3 -I -S <skill-root>/scripts/<name>.py`

Helpers are black boxes. Do not read their source or help during an audit.

## Meaning of completion

`Complete` means the procedure below finished for every declared mechanism:
all priority leads have evidence cards, every candidate was falsified, inputs
were unchanged, and report lint passed. It is not an exhaustiveness claim.
`Partial` means procedure-critical work remains. Partial may contain supported
findings but never means zero problems.

Three rules prevent false completion:

1. A line returned by a helper is not `inspected` until its surrounding context
   was read and its stable lead ID appears in an evidence-backed Closure card.
2. Supplying an applicable specification puts its observable obligations in
   scope. Missing code is not evidence that an obligation is out of scope.
3. A correct imported or vendored core is not evidence that project-owned
   integration reaches it or preserves its behavior.

## Ordered workflow

### 1. Pin once

Run `prepare_audit.py` exactly once with `--repo`, repeated `--spec`, optional
`--output`, and `--budget-seconds` only when the caller supplied a budget. Keep
its returned `run_file`, `output_dir`, `report_file`, `candidates_dir`, and
Specification IDs.
Preserve the generated `## Pinned Inputs` section byte-for-byte. Context
compaction continues the same run; never prepare again or use `run.json` as a
resume entry point.

### 2. Choose probe mode once

A Probe is optional. Run one obvious repository-owned safe command only when
its target behavior and a 60-second bound are known without exploration.
Otherwise select Static-Only immediately. A failed or non-isolated Probe never
gets repaired or retried; continue statically.

### 3. Build inventory and orient sources

Run `inventory.py build --run <run_file>` once. Use bounded helper queries:

- `paths --run ... --scope repository|specification [--path-pattern REGEX] [--cursor N] [--limit 1..30]`
- `search --run ... --scope repository|specification --pattern REGEX [--path-pattern REGEX] [--cursor N] [--limit 1..20]`
- `read --run ... --scope repository|specification --path PATH [--start-line N] [--lines 1..80]`
- `requirements --run ... --source SPEC-xxx [--term REGEX] [--limit 1..30]`
- `hotspots --run ... --scope repository (--term REGEX | --path-pattern REGEX) [--kind boundary|dispatch|state|capability] [--limit 1..20]`
- `triad --run ... --term REGEX --integration-path PATH --core-path PATH [--limit 1..20]`

Repeat path/term flags instead of constructing fragile shell regexes. Ordinary
`rg` and short source reads are allowed when narrow and bounded. Never dump the
full inventory or unbounded recursive output. Paginate to completion only when
an absence claim depends on it; otherwise state the bound.

For every source, read its actual title/opening and relevant headings or top
level records. Record subject, applicability, named mechanisms, and vocabulary;
do not infer them from filename, number, memory, or a neighboring source. Run a
source-wide `requirements` query before package planning. Its `risk_lanes` are
lexical planning evidence, not semantic verdicts.

Record the non-empty boundary/state-timing/routing-traversal/capability lanes
for every source in Coverage's `Source risk lanes` field, and assign them across
packet `Risk lanes` fields exactly as the report contract specifies. A Complete
report may not leave a declared source lane without a mechanism packet.

### 4. Plan mechanism packages before deep reading

Create one package per independent named mechanism or low-overlap requirement
cluster, never one catch-all package per document. A source may own several
packages. Cover every source and every non-empty source-wide risk lane:

- boundary/multiplicity/truncation and chain traversal;
- state, timing, retries, ordering, and unsolicited behavior;
- dispatch, routing, filtering, transformation, and observable output;
- mandatory or optional capabilities and delegation.

A package may cover several lanes only when the same implementation path and
applicability conditions govern them. Different section anchors, actors, input
classes, or state transitions normally mean different packages. Keep at most
eight by merging only tightly coupled mechanisms; never drop a lane to satisfy
the cap. If eight cannot cover the oriented mechanisms, finish as Partial.

Before deep discovery, make at most three architecture queries: root build/docs,
one oriented project-path search, and one project-owned hotspot pass. Determine
ownership from build references, metadata, and docs rather than directory names.
Include every locally built source tree until evidence classifies it as imported
or generated.

Native discovery workers are optional. Use exactly two only when the host
guarantees the same model policy and at least two packages have low overlap.
Workers receive the whole pinned repository, assigned source/package IDs,
bounded commands, and return only oriented claims, scopes, candidates, Closure
cards, and remaining counter-search. They do not write formal output. If worker
support is absent, errors, stalls for about 60 seconds, or returns incomplete
cards, abandon it once and run those packages serially. Never accept a worker's
bare zero-candidate conclusion.

### 5. Discover each package in both directions

Start from pinned obligations and trace implementation; also start from APIs,
registration/dispatch tables, parsers, state machines, finite guards, feature
flags, bridges/adapters/offload surfaces, and capability declarations and map
them back to pinned obligations. Expand at least one relationship beyond the
first hit: caller/callee, registration/handler, adapter/core, or input/output.

Inspect project-owned wrappers, compatibility code, configuration, ingress and
egress before spending substantial effort on an imported core. Every package
needs a concrete project-owned integration anchor, connected core anchor, and
observable result. A build-file inclusion alone is not a runtime seam.

Run one package-specific `requirements` query and inspect every returned
normative lead. Then run one `triad` with distinct integration and core scopes,
all three signatures, repeated literal paths when possible, oriented terms, and
`--limit 10` unless a smaller explicit budget requires Partial.

The triad exposes `priority_leads`. Read context for every priority lead and
close it with the exact card syntax in the [report contract](references/report-contract.md).
Only those evidence-card lead IDs count as inspected. Low-value returned leads
outside the priority set remain bounded discovery, not silently claimed
conformance. If `complete=false`, record that search bound.

A contradiction may also arise directly from comparing an oriented obligation
with implementation flow even when no lexical priority marker exists. Give that
candidate the next packet-local `D-xxx` Derived lead ID and close it under the
appropriate signature. Derived IDs are candidate-only; they can never be used
to manufacture satisfying or out-of-scope coverage. Never invent an `L-*` ID.

Treat these universal signals asymmetrically:

- `explicit-omission`: immediately create a candidate unless an alternate
  implementation, usable delegation, applicability exclusion, or authoritative
  exception is already evidenced. It cannot be closed as ordinary satisfying.
- `finite-boundary`: identify the bounded domain and compare a permitted input
  below, at, and above the effective boundary. Satisfying needs an authoritative
  narrowing clause.
- `dispatch-diversion`: enumerate applicable variants at the classifier and
  trace every destination to an equivalent handler/result. One ordinary path
  cannot prove all variants.
- `state-timing`: construct the required and observed event timeline, including
  trigger, delay/ordering, state transition, and emitted result.
- capability absence: search names/synonyms, APIs, registrations, configuration,
  generated/bound code, alternate locations, callers, and delegation. Lack of
  an in-tree companion is acceptable only when a usable external contract is
  traced.

Optional `MAY` behavior may be a supported interoperability/capability gap, but
must be labeled optional rather than a mandatory violation. A `SHOULD` omission
is contradicted only by evidence that its stated exception applies.

Within at most twelve bounded discovery queries per package, checkpoint its
oriented claims, priority lead cards, strongest candidate chains, and remaining
counter-search. Promote a contradiction as soon as required and observed
behavior diverge; do not first attempt to prove general conformance.

### 6. Form and falsify candidates

One `F-xxx` is one independently fixable implementation cause and may connect
several spec/code anchors. One `S-xxx` is an irreconcilable conflict among
applicable specification clauses after scope/version/precedence search; S alone
does not prove F.

Every candidate contains exact pinned anchors and short quotes, repository
anchors and short quotes, a discriminating witness, required-versus-observed
chain, and material counter-search. Conditional clauses establish only
condition-to-obligation; never reverse them.

Reviewer work starts only after candidates exist. Prefer one fresh native
same-policy reviewer; otherwise use a visibly separate serial falsification
pass. It reopens cited anchors and searches exceptions and satisfying paths,
returning only Supported, Contradicted, or Insufficient. It cannot invent a new
problem. Publish only Supported; Insufficient remains uncertain and forces
Partial. Contradicted candidates retain counterevidence in their Closure cards.

Bind review to the exact candidate content. Write each complete F/S draft under
the temporary `candidates_dir`, never directly under formal `findings/`. Use the
dossier template with at least one `R-xxx` under Affected Requirements and these
review fields: `Outcome: Pending`, the actual review mode, `Digest: Pending`, and
a non-empty Basis. Then run, in order, waiting for each result:

1. `lint_report.py --candidate-review-start <candidate.md>` and retain the
   printed `SERIAL_REVIEW_START <ID> <digest>`;
2. give the reviewer that exact draft and digest, then run
   `lint_report.py --candidate-review-complete <candidate.md> --outcome <Outcome>`;
3. only for Supported, run
   `lint_report.py --candidate-finalize-supported <candidate.md>` and require
   `FINALIZED_CANDIDATE <ID> <same digest>` before copying the finalized file to
   `<output_dir>/findings/<ID>.md` and adding its report link.

If evidence changes, the content digest changes: discard the stale review
result, restore Pending fields, and restart at review-start. Contradicted and
Insufficient drafts stay temporary and are never published. Do not batch review
operations in parallel.

### 7. Challenge zero-finding packages

For every package without a supported finding, perform a separate skeptic pass
over project-owned integration for fixed guards/truncation, dropped or rerouted
variants, parser/chain traversal, missing state/timers, unsupported capabilities,
and alternate/later handling. Record its bounded evidence as the required
Zero-finding challenge card. An apparently conforming imported core is never a
sufficient challenge result.

### 8. Write and validate

Read the [report contract](references/report-contract.md) and use the
[dossier template](assets/templates/dossier.md). Closure and Capability card
counts, not Agent-authored summary arithmetic, determine whether every package
is closed. An uncertainty, missing seam, uncovered risk lane, unreviewed
candidate, or missing evidence card makes the report Partial. Partial never
suppresses an already Supported dossier.

Reserve roughly the final 20% of a known budget for falsification, report
convergence, verification, and lint. Then run exactly:

`python3 -I -S <skill-root>/scripts/inventory.py verify --run <run_file>`

Set Validation to the actual result and run:

`python3 -I -S <skill-root>/scripts/lint_report.py --report <output_dir>`

Exit codes are 0 valid, 1 contract error, 2 tool error. Correct structural
errors without changing semantic claims. Finish only when status, cards,
dossier links, reviewer outcomes, input verification, and actual lint agree.

## Completion boundary

- Formal output is only `report.md` and `findings/*.md`.
- Repository and pinned specifications are unchanged.
- Every published problem has replayable evidence and Supported review.
- Inspected and unchecked scope are explicit; Complete makes no global claim.
- Tests, Oracle data, scoring, and run recovery remain outside this Skill.
