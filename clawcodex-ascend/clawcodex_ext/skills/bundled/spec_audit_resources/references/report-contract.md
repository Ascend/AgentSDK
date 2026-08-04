# Lean v1 Report Contract

The formal output directory contains `report.md` and `findings/`. It contains no run state, JSON report, Oracle, test case, or recovery file.

## Report index

`report.md` uses these H2 sections exactly once and in this order:

1. `Status`
2. `Pinned Inputs`
3. `Execution Mode`
4. `Probe Preflight`
5. `Coverage`
6. `Findings (N)`
7. `Specification Conflicts (N)`
8. `Uncertain and Unfinished Work`
9. `Limitations`
10. `Validation`

Status includes `Contract: Lean v1`, `Result: Complete|Partial`, and a plain-language Meaning. Complete means the procedure finished, not exhaustive correctness. Preserve the preparation-generated Pinned Inputs section byte-for-byte.

Execution Mode records `Runnable|Static-Only`, actual scheduling (`Serial` or `Native discovery (2 workers)`), and unchanged host model policy. Probe Preflight names the command and repository anchor or says None, its bound, whether target code was reached, and why.

Coverage records oriented specifications, completed discovery packages, concrete repository scope inspected, the bidirectional search strategy actually executed, and unchecked/bounded scope. Describe work done, not plans. A Complete report has all planned packages complete and no procedure-critical unfinished item. It may retain explicit bounded or uninspected scope because the audit is not an exhaustiveness proof.

Coverage contains exactly one `### Signature Triad Receipts` subsection. It has
one `#### P-xxx` block per discovery package, using this compact form:

Before that subsection, Coverage has exactly one source-level declaration:

```markdown
- Source risk lanes: SPEC-001=boundary,state-timing; SPEC-002=none
```

```markdown
#### P-001
- Specifications: `SPEC-001`
- Mechanism: Request admission and state propagation
- Risk lanes: boundary,state-timing,capability
- Grounding anchors: `SPEC-001/M-001:12-18`
- Grounded terms: parser, routing, extension
- Normative signals: returned=7; inspected=7
- Capability obligations: implemented=1; delegated=0; gap=0; out-of-scope=0; uncertain=0
- Integration scope: adapter/, bridge/input.go
- Core scope: src/core/
- Seam connection: integration=`adapter/input.go:18`; core=`src/core/store.go:44`; observable=accepted request reaches storage; relationship=the adapter delegates validated input to the core
- Pass boundary/integration: returned=1; inspected=1; disposition=candidate-bearing
- Pass boundary/core: returned=2; inspected=2; disposition=satisfying
- Lead closure boundary: finding=1; contradicted=0; satisfying=2; out-of-scope=0; uncertain=0
- Pass dispatch-state/integration: returned=1; inspected=1; disposition=satisfying
- Pass dispatch-state/core: returned=0; inspected=0; disposition=out-of-scope
- Lead closure dispatch-state: finding=0; contradicted=0; satisfying=1; out-of-scope=0; uncertain=0
- Pass capability/integration: returned=1; inspected=1; disposition=satisfying
- Pass capability/core: returned=0; inspected=0; disposition=out-of-scope
- Lead closure capability: finding=0; contradicted=0; satisfying=1; out-of-scope=0; uncertain=0
- Closure card: leads=`L-e44cf70be302`; signature=boundary; scope=integration; markers=finite-boundary; outcome=finding; spec=`SPEC-001/M-001:12`; implementation=`adapter/input.go:18`; candidate=`C-001`; dossier=`F-001`; review=Supported; counterevidence=None; exclusion=None; basis=the first excluded accepted input contradicts the requirement; witnesses=below=accepted|at=rejected|above=rejected|narrowing=None
- Closure card: leads=`L-53c9b1266f10`, `L-09a4df265bc1`; signature=boundary; scope=core; markers=none; outcome=satisfying; spec=`SPEC-001/M-001:12`; implementation=`src/core/store.go:44`; candidate=None; dossier=None; review=None; counterevidence=None; exclusion=None; basis=the core preserves both inspected leads; witnesses=None
- Closure card: leads=`L-a8215d2094ce`; signature=dispatch-state; scope=integration; markers=dispatch-diversion; outcome=satisfying; spec=`SPEC-001/M-001:13`; implementation=`adapter/input.go:18`; candidate=None; dossier=None; review=None; counterevidence=None; exclusion=None; basis=each variant reaches its intended destination; witnesses=variants=create,update|destinations=store,replace
- Closure card: leads=`L-387bbdf2b7c1`; signature=capability; scope=integration; markers=none; outcome=satisfying; spec=`SPEC-001/M-001:14`; implementation=`src/core/store.go:44`; candidate=None; dossier=None; review=None; counterevidence=None; exclusion=None; basis=the capability is implemented; witnesses=None
- Capability card: id=`K-001`; status=implemented; spec=`SPEC-001/M-001:14`; repository=`src/core/store.go:44`; candidate=None; dossier=None; review=None; exclusion=None; basis=the repository implements the obligation
- State: complete
```

Multiple Specification IDs are separated by semicolons. Every pinned source
appears in at least one package; a source may participate in multiple distinct
mechanisms. No two packages may claim the same `(Specifications set,
Mechanism)` pair. `Mechanism` is a non-empty plain-language identity for the
behavioral path being audited.

Source risk lanes use only `boundary`, `state-timing`, `routing-traversal`, and
`capability`; `none` is allowed only as the complete source-level value. The
source declaration covers every pinned SPEC exactly once. Every packet has a
non-empty comma-separated `Risk lanes` field using the same four lane names.
For each SPEC, the union of lanes from all packets that own it covers every lane
in its source declaration. This is a structural record of orientation output;
the linter does not semantically recompute risk lanes.

Integration and core scopes are non-empty and different; for a co-located
implementation, name distinct role anchors rather than repeating the same
scope. `Seam connection` binds both roles to replayable repository `path:line`
anchors and names an observable and relationship. The six pass rows are
mandatory even when a pass returned zero priority leads. Complete closes every
returned priority lead: the count of L-IDs in that pass's Closure cards equals
`returned`. `inspected` is the count of all L-IDs and D-IDs in those cards, so
it may exceed `returned`. Capability follows the same all-priority-leads rule.
`Normative signals` records the bounded `requirements` result and requires every
returned signal to be inspected. Raw command output is unnecessary.

A Closure card contains either one to three inventory IDs in the exact
`L-[0-9a-f]{12}` form or one to three audit-derived IDs in `D-xxx` form; one
card cannot mix the namespaces. IDs are unique within a packet and may recur in
another mechanism packet. D-IDs represent direct specification/implementation
contradictions or candidates found outside the helper priority list. They are
allowed only with `finding`, `contradicted`, or `uncertain` outcomes and require
the same complete candidate/review evidence as those outcomes on L-IDs. Each
card names its signature, integration/core scope,
markers, outcome, canonical specification anchor, replayable implementation
anchor, candidate/dossier/review state, counterevidence, exclusion anchor,
basis, and marker-specific witnesses. Cards derive both pass `inspected` and
all Lead closure counters. A `candidate-bearing` pass has at least one finding,
contradicted, or uncertain card; a `satisfying` pass has satisfying and optional
out-of-scope cards but no candidate cards; an `out-of-scope` pass contains only
out-of-scope cards. If both sides of a signature return zero, use exactly one
`Empty signature SIGNATURE: basis=...` record and no cards for that signature.
When returned is zero but D-cards exist, those cards replace the Empty signature
record and are included in `inspected` and Lead closure outcome counts.

Finding cards require a candidate ID, Supported review, and published F
dossier. Contradicted cards require a candidate ID, Contradicted review, and
replayable counterevidence. Uncertain cards require an Insufficient review and
make Complete invalid. Satisfying cards require specification and implementation
anchors plus a basis. Out-of-scope cards require an exclusion anchor and basis;
`explicit-omission` can never be satisfying. A satisfying `finite-boundary`
card records below/at/above/narrowing witnesses, `dispatch-diversion` records
variants/destinations, and `state-timing` records a timeline.

`Capability obligations` records structural closure counts for implemented,
delegated, gap, out-of-scope, and uncertain obligations. Its total is at least
one and is derived exactly from globally unique K-ID Capability cards.
Implemented and delegated cards require repository evidence. Gap requires a
Supported candidate and published F dossier also present in a finding Closure
card. Out-of-scope requires an exclusion anchor. Uncertain requires an
Insufficient candidate and makes Complete invalid.

Each package also has exactly three Lead closure rows: `boundary`,
`dispatch-state`, and `capability`. Their fixed counters are `finding`,
`contradicted`, `satisfying`, `out-of-scope`, and `uncertain`. For each
signature, those five counters sum to the `inspected` counts from its
integration and core pass rows. These are structural disposition counts, not a
semantic score. A Complete package has `uncertain=0` for all three signatures.

For `Complete`, every package is `complete`, all six pass rows meet those
inspection bounds, all three Lead closure rows close without uncertainty, all
pinned sources are covered, Inventory verification and Report lint are Passed,
and no procedure-critical work is unfinished. An initial `Partial` whose Report
lint is Not completed may omit receipts. Once a Partial declares `Report lint:
Passed`, it contains at least one structurally valid receipt packet with all six
pass rows and all three Lead closure rows; packages may remain `unfinished`,
inspection bounds may remain unmet, and uncertainty may be non-zero. It must
name those gaps under Uncertain and Unfinished Work.

Every packet with zero finding Closure cards has exactly one replayable
`Zero-finding challenge` evidence record and basis. Complete Limitations must
not claim that the report, audit, or procedure is Partial, unfinished, or not
completed.

```markdown
- Zero-finding challenge: evidence=`adapter/input.go:18`; basis=the adversarial sweep challenged every packet signature
- Empty signature capability: basis=both capability passes returned zero priority leads
```

When the repository embeds, vendors, or ports a subsystem, Coverage also distinguishes project-owned integration paths from imported core paths and names the inspected connection to observable behavior for every applicable package. A missing side or untraced connection makes the package unfinished. A Complete zero-finding report additionally records the separate project-owned counter-candidate sweep.

Findings and Specification Conflicts contain one relative Markdown link per published dossier or `None.` Counts equal links. IDs are unique and zero-padded: `F-001`, `S-001`. No dossier is orphaned or linked twice.

Uncertain work names unresolved candidates and incomplete procedure. Limitations names evidence and coverage bounds. Validation records `Result`, Inventory verification, and Report lint. Set `Result: Passed` and report lint to Passed only immediately before the final linter invocation; a Complete report cannot pass without them.

## Evidence syntax

Use pinned anchors as ``SPEC-xxx/M-xxx:N[-M]`` and repository-relative paths with 1-based line numbers. Specification bullets use the exact `Source; Declared provenance; Anchor; Quote` form shown in the dossier template. Implementation bullets use the exact `Implementation evidence: Source; Lines; Quote; Observed` form; Quote is one inline-code span, using a longer Markdown backtick fence when its literal text contains backticks. Quotes are short exact contiguous substrings of their cited lines. Separate non-contiguous excerpts into separate bullets. Do not cite temporary pinned paths as source identity.

## Finding dossier

`findings/F-xxx.md` starts `# F-xxx: title` and has these H2 sections exactly once:

- Root Cause
- Affected Requirements
- Specification Evidence
- Implementation and Probe Evidence
- Contradiction Chain
- Counter-Search
- Adversarial Review
- Limitations

One dossier is one independently fixable cause, even when multiple requirements or anchors are affected. Its chain must state applicability/condition, required behavior, observed behavior, and the discriminating mismatch.

## Specification-conflict dossier

`findings/S-xxx.md` starts `# S-xxx: title` and has:

- Conflicting Specification Anchors
- Affected Requirements
- Applicability Overlap
- Precedence Search
- Specification Conflict Chain
- Adversarial Review
- Limitations

The chain explains why applicable clauses cannot be jointly satisfied after scope/version/precedence analysis. It does not imply an implementation defect.

## Review and publication

Adversarial Review contains `Outcome: Supported`, `Mode: Fresh native|Serial falsification`, the content-bound finalized `Digest`, and a non-empty Basis. A candidate draft lives only in the temporary `candidates_dir` with `Outcome: Pending` and `Digest: Pending`; use the linter's review-start, review-complete, and finalize-supported commands in order. Contradicted and Insufficient candidates are never published. The reviewer cannot create new issues.

The linter checks structure, IDs, headings, links, counts, placeholders, review outcome, and replayable anchored implementation quotes when possible. It does not decide semantic truth.
