# Finding Protocol

An F finding is one independently fixable implementation cause. It can affect several requirements and connect several specification and code anchors. An S finding is an unresolved conflict between applicable specification clauses after scope and precedence analysis; it cannot by itself prove an implementation defect.

A publishable candidate contains:

- exact pinned specification anchors and short literal quotes;
- repository-relative implementation anchors and short literal quotes;
- the relevant conditions and applicability;
- required behavior and observed behavior;
- a discriminating witness, including boundary/state/order when relevant;
- a concise contradiction or conflict chain;
- material counter-search with bounds and results;
- adversarial review outcome Supported.

Counter-search attempts to falsify, not decorate, the candidate. Reopen cited anchors, search authoritative exceptions and narrowing, alternate implementations, callers, registrations, filters, later handling, and generated/bound code. Reviewer outcomes are only Supported, Contradicted, or Insufficient. Publish Supported only. Put Insufficient material in report uncertainty and discard Contradicted candidates.

Optional `MAY` behavior may be reported as an accurately labeled interoperability/capability gap only when applicable and absent after path-level counter-search. Never call it a MUST/SHOULD violation.

An applicable implementation comment or branch that explicitly says behavior is unsupported, disabled, TODO, or not implemented is affirmative observed evidence, not merely a search-based absence claim. Promote it to a candidate unless concrete evidence shows an alternate implementation, usable delegation, applicability exclusion, or authoritative exception. For a `SHOULD`, the reviewer must establish the specification's permitted exception before contradicting the candidate; optionality alone is not counterevidence. For a `MAY`, preserve the optional label and interoperability impact.
