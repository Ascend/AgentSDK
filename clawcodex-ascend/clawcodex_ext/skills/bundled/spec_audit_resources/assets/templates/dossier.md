# Dossier Templates

Copy one matching block, replace every placeholder, and publish only after Supported review.

## F-xxx implementation Finding

# F-{{NNN}}: {{title}}

## Root Cause

{{One coherent, independently fixable cause.}}

## Affected Requirements

- R-{{NNN}}: {{observable requirement and conditions}}

## Specification Evidence

- Source: `{{SPEC-xxx}}`; Declared provenance: `{{human-readable source title or section}}`; Anchor: `{{SPEC-xxx/M-xxx:N[-M]}}`; Quote: "{{exact pinned substring}}"

## Implementation and Probe Evidence

- Implementation evidence: Source: `{{repository-relative path}}`; Lines: `{{N or N-M}}`; Quote: ``{{exact source substring}}``; Observed: {{behavior established}}

## Contradiction Chain

{{Applicability -> required behavior -> observed behavior -> discriminating mismatch.}}

## Counter-Search

{{Exceptions, alternate implementations, callers, registration, and later handling searched, with bounds and result.}}

## Adversarial Review

- Outcome: {{Pending in candidate draft; finalized to Supported by linter}}
- Mode: {{Fresh native or Serial falsification}}
- Digest: {{Pending in candidate draft; finalized by linter}}
- Basis: {{What the reviewer reopened and attempted to disprove.}}

## Limitations

{{Finding-local limitations or None.}}

---

## S-xxx Specification Conflict

# S-{{NNN}}: {{title}}

## Conflicting Specification Anchors

- Source: `{{SPEC-xxx}}`; Declared provenance: `{{human-readable source title or section}}`; Anchor: `{{SPEC-xxx/M-xxx:N[-M]}}`; Quote: "{{exact pinned substring}}"
- Source: `{{SPEC-xxx}}`; Declared provenance: `{{human-readable source title or section}}`; Anchor: `{{SPEC-xxx/M-xxx:N[-M]}}`; Quote: "{{exact pinned substring}}"

## Affected Requirements

- R-{{NNN}}: {{observable requirement kept uncertain}}

## Applicability Overlap

{{Why the clauses govern the same conditions.}}

## Precedence Search

{{Authority, version, scope, and precedence checked.}}

## Specification Conflict Chain

{{Why the clauses cannot be jointly satisfied.}}

## Adversarial Review

- Outcome: {{Pending in candidate draft; finalized to Supported by linter}}
- Mode: {{Fresh native or Serial falsification}}
- Digest: {{Pending in candidate draft; finalized by linter}}
- Basis: {{What the reviewer reopened and attempted to disprove.}}

## Limitations

{{Conflict-local limitations or None.}}
