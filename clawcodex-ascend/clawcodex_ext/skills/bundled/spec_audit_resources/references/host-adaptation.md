# Host Adaptation

Use the host-configured model and reasoning policy for the entire run. The Skill never chooses or switches models.

Serial discovery is always valid. Native discovery may use at most two workers when the host guarantees they inherit the same model policy and at least two specification sources or mechanism clusters are low-overlap. Each worker receives the whole pinned repository plus assigned specification IDs, performs bounded discovery, and returns concise candidate evidence and searched scope. Workers do not write formal reports. The main Agent deduplicates, counter-searches, reviews, and publishes.

When candidates exist, prefer one fresh same-policy reviewer. If unavailable, perform a visibly separate serial falsification pass. Reviewer output is only Supported, Contradicted, or Insufficient and cannot introduce new findings.

Record scheduling that actually occurred: `Serial` or `Native discovery (2 workers)`. Missing delegation never blocks the audit.
