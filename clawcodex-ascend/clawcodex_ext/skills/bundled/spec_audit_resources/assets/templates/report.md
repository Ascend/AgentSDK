# Spec-Audit Report

## Status

- Contract: Lean v1
- Result: {{Complete or Partial}}
- Meaning: {{what this status establishes}}

## Pinned Inputs

{{Preserve the prepare-generated section byte-for-byte.}}

## Execution Mode

- Mode: {{Runnable or Static-Only}}
- Scheduling: {{Serial or Native discovery (2 workers)}}
- Model policy: Host-configured policy, unchanged

## Probe Preflight

- Command: `{{command or None}}`
- Anchor: `{{repository-relative anchor or None}}`
- Bound: {{bound or Not executed}}
- Reachability: {{Reached target or Not reached}}
- Reason: {{why}}

## Coverage

- Specifications oriented: {{N/N}}
- Discovery packages completed: {{N/N}}
- Repository scope inspected: {{paths, subsystems, relationships}}
- Search strategy executed: {{bidirectional searches and counter-search}}
- Unchecked or bounded scope: {{explicit limits or None}}
- Source risk lanes: {{SPEC-001=boundary,state-timing; SPEC-002=none}}

### Signature Triad Receipts

#### P-001

- Specifications: `{{SPEC-xxx}}`
- Mechanism: {{non-empty behavioral mechanism}}
- Risk lanes: {{comma-separated boundary, state-timing, routing-traversal, capability}}
- Grounding anchors: `{{SPEC-xxx/M-xxx:N-M}}`
- Grounded terms: {{oriented terms}}
- Normative signals: returned={{N}}; inspected={{N}}
- Capability obligations: implemented={{N}}; delegated={{N}}; gap={{N}}; out-of-scope={{N}}; uncertain={{N}}
- Integration scope: {{repository-relative paths}}
- Core scope: {{repository-relative paths}}
- Seam connection: integration=`{{path:line}}`; core=`{{path:line}}`; observable={{non-empty observable}}; relationship={{non-empty relationship}}
- Pass boundary/integration: returned={{N}}; inspected={{N}}; disposition={{candidate-bearing, satisfying, or out-of-scope}}
- Pass boundary/core: returned={{N}}; inspected={{N}}; disposition={{candidate-bearing, satisfying, or out-of-scope}}
- Lead closure boundary: finding={{N}}; contradicted={{N}}; satisfying={{N}}; out-of-scope={{N}}; uncertain={{N}}
- Pass dispatch-state/integration: returned={{N}}; inspected={{N}}; disposition={{candidate-bearing, satisfying, or out-of-scope}}
- Pass dispatch-state/core: returned={{N}}; inspected={{N}}; disposition={{candidate-bearing, satisfying, or out-of-scope}}
- Lead closure dispatch-state: finding={{N}}; contradicted={{N}}; satisfying={{N}}; out-of-scope={{N}}; uncertain={{N}}
- Pass capability/integration: returned={{N}}; inspected={{N}}; disposition={{candidate-bearing, satisfying, or out-of-scope}}
- Pass capability/core: returned={{N}}; inspected={{N}}; disposition={{candidate-bearing, satisfying, or out-of-scope}}
- Lead closure capability: finding={{N}}; contradicted={{N}}; satisfying={{N}}; out-of-scope={{N}}; uncertain={{N}}
- Closure card: leads=`{{L-12hex or D-xxx}}`; signature={{boundary, dispatch-state, or capability}}; scope={{integration or core}}; markers={{none or comma-separated markers}}; outcome={{finding, contradicted, satisfying, out-of-scope, or uncertain}}; spec=`{{SPEC-xxx/M-xxx:N-M}}`; implementation=`{{path:line}}`; candidate={{`C-xxx` or None}}; dossier={{`F-xxx` or None}}; review={{Supported, Contradicted, Insufficient, or None}}; counterevidence={{`anchor` or None}}; exclusion={{`anchor` or None}}; basis={{non-empty basis}}; witnesses={{marker-specific witnesses or None}}
- Capability card: id=`{{K-xxx}}`; status={{implemented, delegated, gap, out-of-scope, or uncertain}}; spec=`{{SPEC-xxx/M-xxx:N-M}}`; repository={{`path:line` or None}}; candidate={{`C-xxx` or None}}; dossier={{`F-xxx` or None}}; review={{Supported, Insufficient, or None}}; exclusion={{`anchor` or None}}; basis={{non-empty basis}}
{{Repeat Closure and Capability cards as required. If a signature returned zero leads, add: - Empty signature SIGNATURE: basis=REASON}}
{{For a packet with zero finding cards add: - Zero-finding challenge: evidence=`ANCHOR`; basis=REASON}}
- State: {{complete or unfinished}}

## Findings ({{N}})

{{Relative F links or None.}}

## Specification Conflicts ({{N}})

{{Relative S links or None.}}

## Uncertain and Unfinished Work

{{items or None.}}

## Limitations

- Complete means the declared procedure finished, not that every possible inconsistency was disproved.
{{other limits or None.}}

## Validation

- Result: {{Passed or Not completed}}
- Inventory verification: {{Passed, Drift, or Not completed}}
- Report lint: {{Passed or Not completed}}
