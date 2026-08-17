#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Normalization helpers for repository-backed tracker payloads.

Split out of ``repo_tracker/client.py``: pure functions that turn raw
GitHub / Gitee / GitCode API payloads into the orchestrator's normalized
models (``Issue`` / ``PullRequestRef`` / ``PullRequestFeedback`` /
``MergeableStatus``).  No HTTP I/O here — the client layer feeds payloads
in and consumes normalized models out.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..issue import Issue  # pylint: disable=relative-beyond-top-level
from ..tracker import MergeableStatus, PullRequestFeedback, PullRequestRef  # pylint: disable=relative-beyond-top-level

if TYPE_CHECKING:
    from .client import RepositoryPlatform


class RepositoryTrackerError(Exception):
    """Raised when repository issue tracker operations fail."""


# Page size for paginated list endpoints, shared by client + PR mixin.
_PAGE_SIZE = 100

# State aliases used by the normalization helpers above.
_OPEN_STATE_ALIASES = {"open", "opened", "reopen", "reopened"}
_TERMINAL_STATE_ALIASES = {
    "closed",
    "close",
    "done",
    "completed",
    "cancelled",
    "canceled",
    "duplicate",
    "failed",
    "abandoned",
    "verification_failed",
}


def _normalize_issue(
    payload: Any,
    *,
    active_states: list[str],
) -> Issue | None:
    """Normalize a raw issue payload into an ``Issue``.

    Args:
        payload: Raw platform issue payload.
        active_states: Issue states treated as active candidates.

    Returns:
        The normalized ``Issue``, or None for non-dict or PR payloads.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("pull_request"):
        return None

    labels = _extract_labels(payload)
    issue_number = payload.get("number")
    raw_state = payload.get("state")
    normalized_state = _choose_issue_state(raw_state, labels, active_states)
    assignee = payload.get("assignee") or {}

    return Issue(
        id=str(issue_number) if issue_number is not None else None,
        identifier=_build_identifier(payload, issue_number),
        title=payload.get("title"),
        description=payload.get("body") or payload.get("description"),
        state=normalized_state,
        branch_name=_extract_branch_name(payload),
        url=payload.get("html_url") or payload.get("url"),
        assignee_id=_assignee_value(assignee),
        author_login=_extract_issue_author(payload),
        labels=labels,
        created_at=_parse_datetime(payload.get("created_at") or payload.get("createdAt")),
        updated_at=_parse_datetime(payload.get("updated_at") or payload.get("updatedAt")),
    )


def _normalize_pull_request(payload: Any) -> PullRequestRef | None:
    """Normalize a raw pull request payload into a ``PullRequestRef``."""
    if not isinstance(payload, dict):
        return None
    number = payload.get("number") or payload.get("iid") or payload.get("id")
    url = payload.get("html_url") or payload.get("url")
    title = payload.get("title")
    return PullRequestRef(
        number=str(number) if number is not None else None,
        url=url if isinstance(url, str) else None,
        title=title if isinstance(title, str) else None,
    )


def _build_issue_comment_url(
    platform: RepositoryPlatform,
    owner: str,
    repo: str,
    number: str,
    comment_id: str,
) -> str | None:
    """Build a human-facing issue/PR comment URL when the API omits html_url.

    GitCode's issue-comments endpoint does not return ``html_url`` for
    conversation comments, so the normalized feedback carries no URL and
    ``issue feedback --list`` falls back to the internal id. This builds
    the canonical web link per platform:

      - GitCode/Gitee: ``{web_host}/{owner}/{repo}/issues/{number}#tid-{id}``
        (the ``#tid-{comment_id}`` anchor is the platform's comment
        permalink fragment).
      - GitHub: ``{web_host}/{owner}/{repo}/issues/{number}#issuecomment-{id}``.

    Returns ``None`` when any required component is missing (caller keeps
    the existing ``url``/falls back to the id).
    """
    if not platform.web_host or not owner or not repo or not number or not comment_id:
        return None
    if platform.name == "github":
        return f"{platform.web_host}/{owner}/{repo}/issues/{number}#issuecomment-{comment_id}"
    # GitCode + Gitee share the #tid-{id} anchor convention.
    return f"{platform.web_host}/{owner}/{repo}/issues/{number}#tid-{comment_id}"


def _normalize_conversation_feedback(payload: dict[str, Any]) -> PullRequestFeedback | None:
    """Normalize a conversation comment payload into feedback, or None."""
    body = payload.get("body")
    feedback_id = payload.get("id")
    if not isinstance(body, str) or not body.strip() or feedback_id is None:
        return None
    return PullRequestFeedback(
        id=f"conversation:{feedback_id}",
        source="conversation",
        body=body,
        author_login=_extract_comment_author(payload),
        status="open",
        created_at=payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        url=_string_value(payload.get("html_url") or payload.get("url")),
    )


def _normalize_inline_feedback(payload: dict[str, Any]) -> PullRequestFeedback | None:
    """Normalize an inline review comment payload into feedback, or None."""
    body = payload.get("body")
    feedback_id = payload.get("id")
    if not isinstance(body, str) or not body.strip() or feedback_id is None:
        return None
    return PullRequestFeedback(
        id=f"inline_review:{feedback_id}",
        source="inline_review",
        body=body,
        author_login=_extract_comment_author(payload),
        file_path=_string_value(payload.get("path") or payload.get("file_path")),
        line=_int_value(payload.get("line") or payload.get("new_line") or payload.get("position")),
        diff_hunk=_string_value(payload.get("diff_hunk")),
        severity="warning",
        status=_normalize_feedback_status(payload),
        created_at=payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        commit_sha=_string_value(payload.get("commit_id") or payload.get("commit_sha")),
        url=_string_value(payload.get("html_url") or payload.get("url")),
    )


def _normalize_review_feedback(payload: dict[str, Any]) -> PullRequestFeedback | None:
    """Normalize a review-summary payload into feedback, or None."""
    body = payload.get("body")
    feedback_id = payload.get("id")
    if not isinstance(body, str) or not body.strip() or feedback_id is None:
        return None
    state = str(payload.get("state") or "").strip().lower()
    severity = "error" if state in {"changes_requested", "request_changes"} else "info"
    return PullRequestFeedback(
        id=f"review_summary:{feedback_id}",
        source="review_summary",
        body=body,
        author_login=_extract_comment_author(payload),
        severity=severity,
        status="open",
        created_at=payload.get("submitted_at") or payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        commit_sha=_string_value(payload.get("commit_id") or payload.get("commit_sha")),
        url=_string_value(payload.get("html_url") or payload.get("url")),
    )


def _normalize_ci_feedback(
    payload: dict[str, Any],
    *,
    commit_sha: str,
    max_log_chars_per_check: int,
) -> PullRequestFeedback | None:
    """Normalize a failing CI check payload into feedback, or None.

    Args:
        payload: Raw CI check payload.
        commit_sha: Commit SHA the check belongs to.
        max_log_chars_per_check: Max body length before truncation.

    Returns:
        The normalized CI feedback, or None when the check did not fail.
    """
    state = str(payload.get("conclusion") or payload.get("state") or "").strip().lower()
    if state not in {"failure", "failed", "error", "cancelled", "timed_out"}:
        return None
    name = _string_value(payload.get("name") or payload.get("context")) or "CI check"
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    summary = _string_value(output.get("summary") if isinstance(output, dict) else None)
    text = _string_value(output.get("text") if isinstance(output, dict) else None)
    description = _string_value(payload.get("description"))
    details_url = _string_value(payload.get("details_url") or payload.get("html_url") or payload.get("target_url"))
    parts = [f"{name} reported {state}."]
    if description:
        parts.append(description)
    if summary:
        parts.append(summary)
    if text:
        parts.append(f"Output:\n{text}")

    annotations = payload.get("_annotations")
    if isinstance(annotations, list) and annotations:
        ann_lines = ["Annotations:"]
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            ann_path = ann.get("path", "")
            ann_line = ann.get("start_line") or ann.get("line")
            ann_end = ann.get("end_line")
            ann_level = ann.get("annotation_level", "")
            ann_msg = ann.get("message", "")
            ann_title = ann.get("title", "")
            loc = ann_path
            if ann_line:
                loc += f":{ann_line}"
                if ann_end and ann_end != ann_line:
                    loc += f"-{ann_end}"
            prefix = f"[{ann_level}] " if ann_level else ""
            title_part = f" {ann_title}:" if ann_title else ""
            ann_lines.append(f"  - {prefix}{loc}{title_part} {ann_msg}")
        parts.append("\n".join(ann_lines))

    body = "\n\n".join(parts)
    if len(body) > max_log_chars_per_check:
        body = body[:max_log_chars_per_check] + "\n...<truncated>"
    feedback_id = payload.get("id") or payload.get("context") or name
    return PullRequestFeedback(
        id=f"ci:{commit_sha}:{feedback_id}",
        source="ci",
        body=body,
        severity="error",
        status="open",
        created_at=payload.get("started_at") or payload.get("created_at"),
        updated_at=payload.get("completed_at") or payload.get("updated_at"),
        commit_sha=commit_sha,
        url=details_url,
    )


def _normalize_feedback_status(payload: dict[str, Any]) -> str:
    """Normalize a comment payload into "resolved", "outdated" or "open"."""
    if payload.get("resolved") is True:
        return "resolved"
    if payload.get("outdated") is True:
        return "outdated"
    return "open"


def _coerce_bool(value: Any) -> bool | None:
    """Tolerant string-to-bool coercion.

    Some platforms (GitCode nested ``conflict_passed``) emit
    ``true`` / ``false`` as strings rather than JSON booleans.
    Returns ``None`` for unknown values so the caller can fall
    back to other signals.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _coerce_int(value: Any) -> int | None:
    """Coerce an integer value (int or numeric string) to int, else None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _extract_gitcode_conflict_state(payload: dict[str, Any]) -> tuple[bool | None, str | None]:
    """GitCode exposes merge status as a nested object.

    The field path varies; we look at:
      - ``mergeable_state`` as object → ``conflict_passed`` (bool),
        ``message`` (str).
      - top-level ``mergeable_state`` as string.
    Returns ``(conflict_passed, state_string)``.
    """
    state_obj = payload.get("mergeable_state")
    if isinstance(state_obj, dict):
        passed = _coerce_bool(state_obj.get("conflict_passed"))
        message = state_obj.get("message")
        state_str = str(message).strip() if isinstance(message, str) and message.strip() else None
        return passed, state_str
    if isinstance(state_obj, str):
        return None, state_obj.strip() or None
    return None, None


def _normalize_mergeable_status(
    payload: Any,
    *,
    platform: str,
    raw: dict[str, Any] | None = None,
) -> MergeableStatus:
    """Normalize a PR payload into MergeableStatus.

    Behavior per platform:

    - **github** / **gitee** — direct read of ``mergeable`` (bool |
      None) and ``mergeable_state`` (string). ``has_conflicts`` is
      set when ``mergeable is False`` or ``mergeable_state == "dirty"``.
    - **gitcode** — fields are often missing or nested. When
      ``mergeable`` is missing and ``mergeable_state`` is missing
      too, returns ``MergeableStatus(mergeable=None, has_conflicts=False)``
      so the daemon treats it as a silent no-op. When the nested
      ``mergeable_state.conflict_passed`` is available, it
      directly drives ``has_conflicts``.

    ``ahead_by`` / ``behind_by`` are populated when the payload
    exposes them (GitHub: ``commits`` count + base ref compare).

    The returned ``raw`` is always a ``{"platform": <name>, "payload": <dict>}``
    structure so callers can introspect both the platform routing and
    the original payload (e.g. for audit logging).
    """
    if not isinstance(payload, dict):
        return MergeableStatus(raw={"platform": platform, "payload": {}})
    payload_dict: dict[str, Any] = payload

    mergeable_raw = payload_dict.get("mergeable")
    mergeable = _coerce_bool(mergeable_raw)
    state_raw = payload_dict.get("mergeable_state")
    if isinstance(state_raw, dict):
        # GitCode sometimes nests even at top level after a merge;
        # collapse to the inner message.
        inner_passed, inner_state = _extract_gitcode_conflict_state(payload_dict)
        mergeable_state = inner_state
        if mergeable is None:
            # Inner conflict_passed overrides top-level null mergeable.
            if inner_passed is True:
                mergeable = True
            elif inner_passed is False:
                mergeable = False
    elif isinstance(state_raw, str):
        mergeable_state = state_raw.strip() or None
    else:
        mergeable_state = None

    has_conflicts = False
    if platform == "gitcode":
        # GitCode: when fields are missing, leave has_conflicts False
        # so daemon treats it as no-op.
        if mergeable is False or mergeable_state == "dirty":
            has_conflicts = True
    else:
        if mergeable is False or mergeable_state == "dirty":
            has_conflicts = True

    ahead_by = _coerce_int(payload_dict.get("ahead_by"))
    behind_by = _coerce_int(payload_dict.get("behind_by"))

    return MergeableStatus(
        mergeable=mergeable,
        mergeable_state=mergeable_state,
        ahead_by=ahead_by,
        behind_by=behind_by,
        has_conflicts=has_conflicts,
        raw={"platform": platform, "payload": payload_dict},
    )


def _string_value(value: Any) -> str | None:
    """Return a non-empty string value, else None."""
    return value if isinstance(value, str) and value.strip() else None


def _int_value(value: Any) -> int | None:
    """Return an integer value, else None."""
    return value if isinstance(value, int) else None


def _build_identifier(payload: dict[str, Any], issue_number: Any) -> str | None:
    """Build a human-readable ``repo#number`` identifier for an issue."""
    if issue_number is None:
        return None
    repo_name = payload.get("repository") or payload.get("repo") or payload.get("repository_name")
    if isinstance(repo_name, str) and repo_name.strip():
        return f"{repo_name}#{issue_number}"
    return f"#{issue_number}"


def _choose_issue_state(
    raw_state: Any,
    labels: list[str],
    active_states: list[str],
) -> str | None:
    """Pick the issue state from active-state labels, falling back to raw.

    Returns:
        The first active state found among labels, the normalized raw
        state, or None when neither is available.
    """
    normalized_active = [state.strip().lower() for state in active_states if state.strip()]
    label_set = {label.lower() for label in labels}
    for state_name in normalized_active:
        if state_name in label_set:
            return state_name
    if isinstance(raw_state, str):
        return raw_state.strip().lower()
    return None


def _extract_labels(payload: dict[str, Any]) -> list[str]:
    """Extract lowercased label names from a raw issue payload."""
    labels = payload.get("labels", [])
    result: list[str] = []
    if not isinstance(labels, list):
        return result
    for item in labels:
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = item
        if isinstance(name, str) and name.strip():
            result.append(name.strip().lower())
    return result


def _extract_branch_name(payload: dict[str, Any]) -> str | None:
    """Extract the branch name from an issue body directive, or None."""
    body = payload.get("body") or payload.get("description")
    if not isinstance(body, str) or not body.strip():
        return None

    patterns = (
        r"(?im)^\s*branch(?:_name)?\s*[:=]\s*`?([A-Za-z0-9._/\-]+)`?\s*$",
        r"(?im)^\s*git\s+branch\s*[:=]\s*`?([A-Za-z0-9._/\-]+)`?\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, body)
        if match:
            return match.group(1).strip()
    return None


def _matches_assignee(issue: Issue, assignee: str | None) -> bool:
    """Return whether an issue is assigned to the given assignee (or any)."""
    if not assignee:
        return True
    normalized = assignee.strip().lower()
    if not normalized:
        return True
    return (issue.assignee_id or "").strip().lower() == normalized


def _assignee_value(assignee: Any) -> str | None:
    """Extract an assignee identifier from a dict or string payload."""
    if isinstance(assignee, dict):
        for key in ("login", "name", "username", "id"):
            value = assignee.get(key)
            if isinstance(value, str) and value.strip():
                return value
    elif isinstance(assignee, str) and assignee.strip():
        return assignee
    return None


def _repository_label_filter(active_states: list[str]) -> list[str]:
    """Map active states to candidate labels, skipping state-name aliases.

    Returns:
        The active state names that are not plain open/terminal aliases.
    """
    labels: list[str] = []
    for state_name in active_states:
        normalized = state_name.strip().lower()
        if not normalized:
            continue
        if normalized in _OPEN_STATE_ALIASES or normalized in _TERMINAL_STATE_ALIASES:
            continue
        labels.append(state_name)
    return labels


def _build_issue_update_payload(
    *,
    state: str | None,
    labels: list[str] | None,
    platform: RepositoryPlatform,
) -> dict[str, Any]:
    """Build the PATCH payload for :meth:`RepositoryIssueClient.update_issue`.

    For ``access_token``-auth platforms (Gitee, GitCode) the ``state``
    parameter is translated to ``state_event`` — GitLab-style API.
    **Known limitation (GitCode)**: ``state_event=close`` is accepted
    (HTTP 200) but does **not** actually close the issue. This is a
    platform-side bug; callers that rely on the close side-effect
    should verify the issue state after the call or degrade gracefully.
    See ``tests/telemetry/telemetry_issue_push_real.py`` for reproduction.
    """
    payload: dict[str, Any] = {}
    normalized = (state or "").strip().lower()
    if normalized:
        if normalized in _TERMINAL_STATE_ALIASES:
            if platform.auth_mode == "bearer":
                payload["state"] = platform.closed_state
            else:
                payload["state_event"] = "close"
        elif normalized in _OPEN_STATE_ALIASES:
            if platform.auth_mode == "bearer":
                payload["state"] = platform.open_state
            else:
                payload["state_event"] = "reopen"
    # An explicit empty list is a real update: it clears all remote labels.
    # ``None`` alone means the caller did not request a label mutation.
    if labels is not None:
        if platform.auth_mode == "bearer":
            payload["labels"] = labels
        else:
            payload["labels"] = ",".join(labels)
    return payload


def _find_pull_request_in_payload(
    payload: Any,
    *,
    head_branch: str,
    base_branch: str,
    allow_unique_unmatched: bool = False,
) -> PullRequestRef | None:
    """Find a pull request matching head/base branches in a list payload.

    Args:
        payload: Raw list payload of pull requests.
        head_branch: Branch carrying the changes.
        base_branch: Branch the changes target.
        allow_unique_unmatched: When True, return the sole PR whose payload
            carries no branch fields.

    Returns:
        The matching ``PullRequestRef``, or None when not found.
    """
    if not isinstance(payload, list):
        return None
    unmatched_without_branches: list[PullRequestRef] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        pr = _normalize_pull_request(item)
        if pr is None:
            continue
        if _pull_request_matches(item, head_branch=head_branch, base_branch=base_branch):
            return pr
        if not _payload_has_branch_fields(item):
            unmatched_without_branches.append(pr)
    if allow_unique_unmatched and len(unmatched_without_branches) == 1:
        return unmatched_without_branches[0]
    return None


def _pull_request_matches(
    payload: dict[str, Any],
    *,
    head_branch: str,
    base_branch: str,
) -> bool:
    """Return whether a PR payload's head/base refs match the branches."""
    head = _extract_ref_name(payload.get("head") or payload.get("source_branch"))
    base = _extract_ref_name(payload.get("base") or payload.get("target_branch"))
    if head is None and base is None:
        return False
    if head is not None and head != head_branch:
        return False
    if base is not None and base != base_branch:
        return False
    return True


def _payload_has_branch_fields(payload: dict[str, Any]) -> bool:
    """Return whether a PR payload exposes any branch reference fields."""
    return any(key in payload for key in ("head", "base", "source_branch", "target_branch"))


def _extract_ref_name(value: Any) -> str | None:
    """Extract a branch ref name from a string or ref-object value."""
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        for key in ("ref", "name", "branch", "label"):
            ref = value.get(key)
            if not isinstance(ref, str) or not ref:
                continue
            return ref.rsplit(":", 1)[-1] if key == "label" else ref
    return None


def _is_not_found_error(exc: RepositoryTrackerError) -> bool:
    """Return whether the error indicates a 404 Not Found response."""
    return "status=404" in str(exc)


def _parse_datetime(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp into a timezone-aware datetime, or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _comment_sort_key(comment: dict[str, Any]) -> tuple[float, int, int | str]:
    """Sort key ordering comments oldest-first with stable ID tie-breaking."""
    created_at = _parse_datetime(comment.get("created_at") or comment.get("createdAt"))
    timestamp = created_at.timestamp() if created_at is not None else 0.0
    raw_id = comment.get("id")
    try:
        return timestamp, 0, int(raw_id)
    except (TypeError, ValueError):
        return timestamp, 1, str(raw_id or "")


def _extract_comment_author(comment: dict[str, Any]) -> str | None:
    """Extract author login from a comment payload."""
    user = comment.get("user") or comment.get("author")
    if isinstance(user, dict):
        return user.get("login") or user.get("username") or user.get("name")
    if isinstance(user, str) and user.strip():
        return user
    return None


def _extract_issue_author(issue: dict[str, Any]) -> str | None:
    """Extract author login from an issue payload."""
    user = issue.get("user") or issue.get("author")
    if isinstance(user, dict):
        return user.get("login") or user.get("username") or user.get("name")
    if isinstance(user, str) and user.strip():
        return user
    return None
