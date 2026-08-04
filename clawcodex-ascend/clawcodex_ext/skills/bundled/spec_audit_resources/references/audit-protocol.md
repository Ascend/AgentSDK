# Audit Protocol

This reference expands the discovery logic in `SKILL.md`; the ordered workflow there is authoritative.

## Evidence funnel

1. Orient each pinned specification from its actual content.
2. Extract observable obligations with applicability, conditions, quantifiers, timing, and ordering intact.
3. Search named concepts and implementation synonyms across the repository.
4. Expand one relationship hop through callers, registrations, dispatch, adapters, or observable output.
5. Challenge boundaries, early exits, state transitions, filters, chain traversal, and capability surfaces.
6. Create a candidate as soon as required and observed behavior separate.
7. Counter-search exceptions, alternate paths, later handling, and authoritative narrowing.
8. Send surviving candidates through adversarial review.

Search absence must be bounded and reproducible. Search pagination to completion when absence is essential, or narrow by path/mechanism and state the remaining bound. Text absence alone is weak; include registration, generated/bound code, feature flags, callers, and runtime routing where applicable.

For conditions, prove the antecedent before claiming a violated consequence. For finite guards, reason below, at, and above the effective boundary and name the first valid input whose behavior changes. For cross-boundary behavior, trace adapter/ingress through the core to observable egress rather than treating a correct core function as the whole system.

Runnable probes are optional evidence. A single safe, obvious repository-owned target command may enable them; otherwise Static-Only is immediate and permanent for the run. Never install, build a new harness, or repair the environment.

When a budget is explicit, reserve roughly the final 20% for counter-search, review, report convergence, inventory verification, and lint. Discovery ends when that reserve begins. Report unfinished work honestly; never reset the clock or prepare a second run.
