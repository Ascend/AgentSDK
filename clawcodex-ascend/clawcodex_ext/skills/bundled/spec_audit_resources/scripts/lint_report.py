#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=too-many-lines

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Validate the structural contract of a Spec-Audit Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence


REPORT_TITLE = "# Spec-Audit Report"
LEAN_V1_MARKER = "- Contract: Lean v1"
LEAN_SIGNATURES = ("boundary", "dispatch-state", "capability")
LEAN_RISK_LANES = {
    "boundary",
    "state-timing",
    "routing-traversal",
    "capability",
}
LEAN_SIGNATURE_PASSES = {f"{signature}/{scope}" for signature in LEAN_SIGNATURES for scope in ("integration", "core")}
REPORT_SECTIONS = (
    "Status",
    "Pinned Inputs",
    "Execution Mode",
    "Probe Preflight",
    "Coverage",
    "Uncertain and Unfinished Work",
    "Limitations",
    "Validation",
)
REPORT_SECTION_PATTERNS = (
    r"^## Status\s*$",
    r"^## Pinned Inputs\s*$",
    r"^## Execution Mode\s*$",
    r"^## Probe Preflight\s*$",
    r"^## Coverage\s*$",
    r"^## Findings \(\d+\)\s*$",
    r"^## Specification Conflicts \(\d+\)\s*$",
    r"^## Uncertain and Unfinished Work\s*$",
    r"^## Limitations\s*$",
    r"^## Validation\s*$",
)
FINDING_SECTIONS = (
    "Root Cause",
    "Affected Requirements",
    "Specification Evidence",
    "Implementation and Probe Evidence",
    "Contradiction Chain",
    "Counter-Search",
    "Adversarial Review",
    "Limitations",
)
CONFLICT_SECTIONS = (
    "Conflicting Specification Anchors",
    "Affected Requirements",
    "Applicability Overlap",
    "Precedence Search",
    "Specification Conflict Chain",
    "Adversarial Review",
    "Limitations",
)

DOSSIER_NAME_RE = re.compile(r"^(?P<kind>[FS])-(?P<number>\d{3})\.md$")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\n]+)\)")
PLACEHOLDER_RE = re.compile(r"{{[^{}\n]+}}")
OUTCOME_RE = re.compile(r"^\s*(?:-\s*)?Outcome:\s*(.*?)\s*$", re.MULTILINE)
BASIS_RE = re.compile(r"^\s*(?:-\s*)?Basis:\s*(\S.*)$", re.MULTILINE)
REVIEW_MODE_RE = re.compile(r"^\s*(?:-\s*)?Mode:\s*(\S.*)$", re.MULTILINE)
DIGEST_RE = re.compile(r"^\s*(?:-\s*)?Digest:\s*([0-9a-f]{64})\s*$", re.MULTILINE)
DIGEST_FIELD_RE = re.compile(r"^\s*(?:-\s*)?Digest:\s*(.*?)\s*$", re.MULTILINE)
PINNED_REPOSITORY_RE = re.compile(
    r"^- Repository:\s+.+;\s+Pin SHA-256:\s+`([0-9a-f]{64})`\s*$",
    re.MULTILINE,
)
PINNED_REPOSITORY_PATH_RE = re.compile(
    r"^- Repository:\s+`([^`\n]+)`;\s+Pin SHA-256:\s+`[0-9a-f]{64}`\s*$",
    re.MULTILINE,
)
PINNED_SPEC_RE = re.compile(
    r"^- Specification `(SPEC-\d{3})`:\s+.+;\s+SHA-256:\s+`([0-9a-f]{64})`\s*$",
    re.MULTILINE,
)
PINNED_SPEC_MEMBER_RE = re.compile(
    r"^  - Member `(?P<identifier>SPEC-\d{3}/M-\d{3})`: "
    r"`(?P<identity>[^`\n]+)`; Bytes: (?P<size>\d+); "
    r"SHA-256: `(?P<sha256>[0-9a-f]{64})`\s*$",
    re.MULTILINE,
)
SOURCE_UNIT_RE = re.compile(
    r"^- Source unit `(SPEC-\d{3})`: total=(\d+); mapped=(\d+); "
    r"classified=(\d+); unfinished=(\d+)\s*$",
    re.MULTILINE,
)
WORK_PACKET_RE = re.compile(
    r"^#### (?P<identifier>P-\d{3})\s*$\n(?P<body>.*?)(?=^#### |^### |\Z)",
    re.MULTILINE | re.DOTALL,
)
WORK_PACKET_CLASSES = {
    "normative",
    "boundary",
    "state",
    "routing",
    "capability",
    "alternatives",
}
REPOSITORY_COVERAGE_RE = re.compile(
    r"^- Repository inventory:\s*(Complete|Incomplete)\s*$",
    re.MULTILINE,
)
SPECIFICATION_COVERAGE_RE = re.compile(
    r"^- Specification source coverage:\s*(Complete|Incomplete)\s*$",
    re.MULTILINE,
)
SPECIFICATION_ANCHOR_LINE_RE = re.compile(
    r"^- Source: `(SPEC-\d{3})`; Declared provenance: `([^`\n]+)`; "
    r'Anchor: `([^`\n]+)`; Quote: "(.*\S.*)"$'
)
CANONICAL_SPECIFICATION_ANCHOR_RE = re.compile(
    r"^(?P<source>SPEC-\d{3})/(?P<member>M-\d{3}):"
    r"(?P<start>[1-9]\d*)(?:-(?P<end>[1-9]\d*))?$"
)
IMPLEMENTATION_EVIDENCE_LINE_RE = re.compile(
    r"^- Implementation evidence: Source: `(?P<source>[^`\n]+)`; "
    r"Lines: `(?P<start>\d+)(?:-(?P<end>\d+))?`; "
    r"Quote: (?P<fence>`+)(?P<quote>[^\n]*?)(?P=fence); "
    r"Observed: (?P<observed>\S.*)$"
)
GUARD_CHALLENGE_RE = re.compile(
    r"^Guards=(?P<guards>None|`[^`;\n]+`); "
    r"Authoritative narrowing=(?P<narrowing>None|`[^`;\n]+`); "
    r"First changed input=(?P<changed>None|`[^`;\n]+`); "
    r"Comparison=(?P<comparison>same|different|undetermined)$"
)
TERMINAL_ASSESSMENTS = {
    "consistent",
    "inconsistent",
    "potential-or-uncertain",
    "non-verifiable",
    "failure-or-partial",
}
SOURCE_DISPOSITIONS = {
    "mapped",
    "non-requirement",
    "non-verifiable",
    "unfinished",
}
NONE_MARKERS = {"", "None.", "None", "- None.", "- None"}
CANDIDATE_DIGEST_DOMAIN = b"spec-audit-reviewed-dossier-v1\0"
TERMINAL_REVIEW_OUTCOMES = ("Supported", "Contradicted", "Insufficient")
CANDIDATE_REVIEW_HEADING_RE = re.compile(rb"(?m)^## Adversarial Review(?:\r\n|\n|$)")
NEXT_H2_HEADING_RE = re.compile(rb"(?m)^## [^\r\n]+(?:\r\n|\n|$)")


class ToolError(RuntimeError):
    """An environmental or filesystem failure, rather than a bad report."""


class ContractError(RuntimeError):
    """A caller-visible candidate lifecycle contract failure."""


def _read_dossier(path: Path) -> tuple[bytes, str]:
    try:
        dossier = path.read_bytes()
    except OSError as exc:
        raise ToolError(f"cannot read dossier {path}: {exc}") from exc
    try:
        text = dossier.decode("utf-8")
    except UnicodeError as exc:
        raise ToolError(f"dossier is not valid UTF-8: {path}: {exc}") from exc
    return dossier, text


def _candidate_dossier_digest_bytes(dossier: bytes, path: Path) -> str:
    """Return the reviewer-bound digest for already-read exact dossier bytes."""

    review_headings = list(CANDIDATE_REVIEW_HEADING_RE.finditer(dossier))
    if len(review_headings) != 1:
        raise ToolError(f"dossier must contain exactly one exact '## Adversarial Review' H2 section: {path}")
    review_heading = review_headings[0]
    next_heading = NEXT_H2_HEADING_RE.search(dossier, review_heading.end())
    review_end = next_heading.start() if next_heading is not None else len(dossier)
    bound_bytes = dossier[: review_heading.start()] + dossier[review_end:]
    return hashlib.sha256(CANDIDATE_DIGEST_DOMAIN + bound_bytes).hexdigest()


def candidate_dossier_digest(path: Path) -> str:
    """Return the reviewer-bound digest for one exact UTF-8 dossier."""

    dossier, _text = _read_dossier(path)
    return _candidate_dossier_digest_bytes(dossier, path)


def finalize_candidate_supported(path: Path, digest: str) -> None:
    """Atomically stamp only one validated Pending review section as Supported."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ToolError(f"cannot inspect candidate dossier {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ToolError(f"candidate dossier is not a regular file: {path}")

    dossier, _text = _read_dossier(path)
    if _candidate_dossier_digest_bytes(dossier, path) != digest:
        raise ToolError(f"candidate dossier changed before finalization: {path}")
    review_headings = list(CANDIDATE_REVIEW_HEADING_RE.finditer(dossier))
    if len(review_headings) != 1:
        raise ToolError(f"candidate dossier lost its unique Adversarial Review section before finalization: {path}")
    review_heading = review_headings[0]
    next_heading = NEXT_H2_HEADING_RE.search(dossier, review_heading.end())
    review_end = next_heading.start() if next_heading is not None else len(dossier)
    review = dossier[review_heading.start() : review_end]

    outcome_pattern = re.compile(rb"(?m)^([ \t]*(?:-[ \t]*)?Outcome:[ \t]*)Pending([ \t]*\r?)$")
    digest_pattern = re.compile(rb"(?m)^([ \t]*(?:-[ \t]*)?Digest:[ \t]*)Pending([ \t]*\r?)$")
    review, outcome_count = outcome_pattern.subn(
        lambda match: match.group(1) + b"Supported" + match.group(2),
        review,
    )
    review, digest_count = digest_pattern.subn(
        lambda match: match.group(1) + digest.encode("ascii") + match.group(2),
        review,
    )
    if outcome_count != 1 or digest_count != 1:
        raise ToolError(f"candidate review fields changed before finalization: {path}")
    finalized = dossier[: review_heading.start()] + review + dossier[review_end:]

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(finalized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, stat.S_IMODE(metadata.st_mode))
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass  # Best-effort operation failed; keep the surrounding fallback.
        raise ToolError(f"cannot finalize candidate dossier {path}: {exc}") from exc


def _review_receipt_path(path: Path) -> Path:
    return path.parent / f".{path.name}.review.json"


def _write_review_receipt(
    path: Path,
    *,
    digest: str,
    phase: str,
    outcome: str | None = None,
) -> None:
    receipt = _review_receipt_path(path)
    payload = {
        "schema_version": 1,
        "dossier_id": path.stem.upper(),
        "digest": digest,
        "phase": phase,
        "outcome": outcome,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{receipt.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, sort_keys=True, separators=(",", ":"))
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, receipt)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass  # Best-effort operation failed; keep the surrounding fallback.
        raise ToolError(f"cannot write candidate review receipt {receipt}: {exc}") from exc


def _read_review_receipt(path: Path) -> dict[str, object]:
    receipt = _review_receipt_path(path)
    try:
        metadata = receipt.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"candidate review receipt is not a regular file: {receipt}")
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"candidate has no content-bound review receipt: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ToolError(f"cannot read candidate review receipt {receipt}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"candidate review receipt is not an object: {receipt}")
    return payload


def _require_review_receipt(
    path: Path,
    *,
    digest: str,
    phase: str,
    outcome: str | None = None,
) -> None:
    payload = _read_review_receipt(path)
    expected = {
        "schema_version": 1,
        "dossier_id": path.stem.upper(),
        "digest": digest,
        "phase": phase,
        "outcome": outcome,
    }
    if payload != expected:
        raise ContractError(f"candidate review receipt does not match {phase} digest/outcome: {path}")


@dataclass
class PinnedSpecificationMember:
    """One member identity and byte length declared by input preparation."""

    identifier: str
    source: str
    identity: str
    size: int


@dataclass
class CanonicalSpecificationAnchor:
    """One canonical member-owned, 1-based specification line range."""

    source: str
    member: str
    start: int
    end: int


@dataclass
class SourceUnitEntry:
    """One structurally parsed Source Coverage Ledger entry."""

    identifier: str
    source: str
    member: str
    byte_start: int | None
    byte_end: int | None
    disposition: str
    requirements: set[str]


@dataclass
class RequirementEntry:
    """One structurally parsed Requirement Assessment Ledger entry."""

    identifier: str
    units: set[str]
    specifications: set[str]
    members: set[str]
    assessment: str
    comparison: str
    disposition: str
    dossier_ids: set[str]


@dataclass
class CoverageLedger:
    """Parsed ledger relationships needed for report/dossier reconciliation."""

    source_units: dict[str, SourceUnitEntry] = field(default_factory=dict)
    requirements: dict[str, RequirementEntry] = field(default_factory=dict)
    pinned_members: dict[str, PinnedSpecificationMember] = field(default_factory=dict)
    finding_requirements: dict[str, set[str]] = field(default_factory=dict)
    conflict_requirements: dict[str, set[str]] = field(default_factory=dict)
    terminal_uncertainty: set[str] = field(default_factory=set)
    partial_requirements: set[str] = field(default_factory=set)
    nonverifiable_source_units: set[str] = field(default_factory=set)


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ToolError(f"cannot read UTF-8 file {path}: {exc}") from exc


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def _heading_count(text: str, heading: str) -> int:
    return len(re.findall(rf"^## {re.escape(heading)}\s*$", text, re.MULTILINE))


def _section_span(text: str, heading: str) -> tuple[int, int, str] | None:
    match = re.search(rf"^## {re.escape(heading)}\s*$", text, re.MULTILINE)
    if match is None:
        return None
    next_heading = re.search(r"^##\s+", text[match.end() :], re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(text)
    return match.start(), end, text[match.end() : end]


def _section_body(text: str, heading: str) -> str:
    span = _section_span(text, heading)
    return span[2] if span is not None else ""


def _require_sections(
    text: str,
    required: Sequence[str],
    source: str,
    errors: list[str],
) -> None:
    for heading in required:
        count = _heading_count(text, heading)
        if count == 0:
            errors.append(f"{source}: missing required section '## {heading}'")
        elif count > 1:
            errors.append(f"{source}: duplicate section '## {heading}'")
        elif not _section_body(text, heading).strip():
            errors.append(f"{source}: required section '## {heading}' is empty")


def _require_heading_order(
    text: str,
    required: Sequence[str],
    source: str,
    errors: list[str],
) -> None:
    positions: list[int] = []
    for heading in required:
        match = re.search(rf"^## {re.escape(heading)}\s*$", text, re.MULTILINE)
        if match is None:
            return
        positions.append(match.start())
    if positions != sorted(positions):
        errors.append(f"{source}: required sections are not in contract order")


def _require_report_order(text: str, errors: list[str]) -> None:
    positions: list[int] = []
    for pattern in REPORT_SECTION_PATTERNS:
        match = re.search(pattern, text, re.MULTILINE)
        if match is None:
            return  # Missing/malformed headings are reported by other checks.
        positions.append(match.start())
    if positions != sorted(positions):
        errors.append("report.md: required sections are not in contract order")


def _require_status(text: str, errors: list[str]) -> None:
    matches = re.findall(
        r"^-\s*Result:\s*(Complete|Partial)\s*$",
        text,
        re.MULTILINE,
    )
    if len(matches) != 1:
        errors.append("report.md: Status must contain exactly one '- Result: Complete' or '- Result: Partial'")


def _probe_field(
    body: str,
    label: str,
    errors: list[str],
) -> str | None:
    matches = re.findall(
        rf"^-\s*{re.escape(label)}:\s*(\S.*)$",
        body,
        re.MULTILINE,
    )
    if len(matches) != 1:
        errors.append(f"report.md: Probe Preflight requires exactly one non-empty '- {label}: ...' field")
        return None
    return matches[0].strip()


def _backticked_probe_value(
    value: str | None,
    label: str,
    errors: list[str],
) -> str | None:
    if value is None:
        return None
    match = re.fullmatch(r"`([^`\n]+)`", value)
    if match is None:
        errors.append(f"report.md: Probe Preflight {label} must be one backticked value")
        return None
    return match.group(1)


def _lint_probe_contract(text: str, errors: list[str]) -> None:
    execution_mode = _section_body(text, "Execution Mode")
    probe = _section_body(text, "Probe Preflight")
    modes = re.findall(
        r"^-\s*Mode:\s*(Runnable|Static-Only|Not selected)\s*$",
        execution_mode,
        re.MULTILINE,
    )
    if len(modes) != 1:
        errors.append(
            "report.md: Execution Mode requires exactly one '- Mode: Runnable' or "
            "'- Mode: Static-Only'; an unfinished preflight may use '- Mode: Not selected'"
        )
        mode = None
    else:
        mode = modes[0]
    schedules = re.findall(
        r"^-\s*Scheduling:\s*(Serial|Native discovery \(2 workers\))\s*$",
        execution_mode,
        re.MULTILINE,
    )
    if len(schedules) != 1:
        errors.append(
            "report.md: Execution Mode requires exactly one actual Scheduling value: "
            "Serial or Native discovery (2 workers)"
        )

    command = _backticked_probe_value(
        _probe_field(probe, "Command", errors),
        "Command",
        errors,
    )
    anchor = _backticked_probe_value(
        _probe_field(probe, "Anchor", errors),
        "Anchor",
        errors,
    )
    verification = _probe_field(probe, "Anchor verification", errors)
    bound = _probe_field(probe, "Bound", errors)
    execution = _probe_field(probe, "Execution", errors)
    reachability = _probe_field(probe, "Reachability", errors)
    _probe_field(probe, "Reason", errors)
    if None in {
        command,
        anchor,
        verification,
        bound,
        execution,
        reachability,
    }:
        return

    if command == "None":
        required = {
            "Anchor": (anchor, "None"),
            "Anchor verification": (verification, "Not applicable"),
            "Bound": (bound, "Not executed"),
            "Execution": (execution, "Not executed"),
            "Reachability": (reachability, "Not reached"),
        }
        for label, (actual, expected) in required.items():
            if actual != expected:
                errors.append(f"report.md: Command None requires {label} {expected!r} (found {actual!r})")
        if mode not in {"Static-Only", "Not selected"}:
            errors.append("report.md: Command None requires Static-Only or unfinished Not selected mode")
        if mode == "Not selected":
            statuses = re.findall(
                r"^-\s*Result:\s*(Complete|Partial)\s*$",
                text,
                re.MULTILINE,
            )
            if statuses != ["Partial"]:
                errors.append("report.md: Not selected Execution Mode is allowed only in a Partial report")
        return

    if anchor == "None":
        errors.append("report.md: a non-None Probe command requires a repository-relative Anchor")
    else:
        posix = PurePosixPath(anchor)
        windows = PureWindowsPath(anchor)
        if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
            errors.append("report.md: Probe Anchor must be a safe repository-relative path")
    if verification != "Passed":
        errors.append("report.md: a non-None Probe command requires 'Anchor verification: Passed'")
    bound_match = re.fullmatch(
        r"(\d{1,3})\s+seconds\s+via\s+(\S.*)",
        bound,
    )
    if bound_match is None or not 1 <= int(bound_match.group(1)) <= 60:
        errors.append(
            "report.md: a non-None Probe command requires a 1..60 second bound and named enforcement mechanism"
        )
    if execution not in {"Reached target", "Failed before target"}:
        errors.append(
            "report.md: a non-None Probe command Execution must be 'Reached target' or 'Failed before target'"
        )
    if reachability not in {"Reached target", "Not reached"}:
        errors.append("report.md: Probe Reachability must be 'Reached target' or 'Not reached'")
    if execution == "Reached target":
        if reachability != "Reached target" or mode != "Runnable":
            errors.append("report.md: reaching target code requires Runnable mode and reached target")
    elif execution == "Failed before target":
        if reachability != "Not reached" or mode != "Static-Only":
            errors.append("report.md: pre-target Probe failure requires Static-Only mode and no reachability")


def _contains_line(body: str, line: str) -> bool:
    return re.search(rf"^{re.escape(line)}\s*$", body, re.MULTILINE) is not None


def _subsection_body(
    body: str,
    heading: str,
    source: str,
    errors: list[str],
) -> str:
    matches = list(re.finditer(rf"^### {re.escape(heading)}\s*$", body, re.MULTILINE))
    if len(matches) != 1:
        errors.append(f"{source}: requires exactly one '### {heading}' subsection")
        return ""
    match = matches[0]
    next_heading = re.search(r"^###\s+", body[match.end() :], re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(body)
    return body[match.end() : end]


def _ledger_blocks(
    body: str,
    identifier_pattern: str,
    ledger_name: str,
    errors: list[str],
) -> list[tuple[str, str]]:
    headings = list(re.finditer(r"^####\s+(.+?)\s*$", body, re.MULTILINE))
    blocks: list[tuple[str, str]] = []
    identifiers: list[str] = []
    for index, heading in enumerate(headings):
        identifier = heading.group(1).strip()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        block = body[heading.end() : end]
        if re.fullmatch(identifier_pattern, identifier) is None:
            errors.append(f"report.md: {ledger_name} has malformed entry heading '#### {identifier}'")
            continue
        identifiers.append(identifier)
        blocks.append((identifier, block))
    duplicates = sorted(identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1)
    for identifier in duplicates:
        errors.append(f"report.md: duplicate {ledger_name} identity {identifier}")
    return blocks


def _field_value(
    block: str,
    label: str,
    source: str,
    errors: list[str],
) -> str | None:
    matches = re.findall(
        rf"^- {re.escape(label)}:\s*(\S.*)$",
        block,
        re.MULTILINE,
    )
    if len(matches) != 1:
        errors.append(f"{source}: requires exactly one non-empty '- {label}: ...'")
        return None
    return matches[0].strip()


def _nested_field_value(
    block: str,
    label: str,
    source: str,
    errors: list[str],
) -> str | None:
    matches = re.findall(
        rf"^  - {re.escape(label)}:\s*(\S.*)$",
        block,
        re.MULTILINE,
    )
    if len(matches) != 1:
        errors.append(f"{source}: Trace requires exactly one non-empty '  - {label}: ...'")
        return None
    return matches[0].strip()


def _parse_code_id_list(
    value: str,
    prefix: str,
    source: str,
    errors: list[str],
    *,
    allow_none: bool,
) -> set[str]:
    if allow_none and value == "None":
        return set()
    pattern = rf"`{prefix}-\d{{3}}`(?:,\s*`{prefix}-\d{{3}}`)*"
    if re.fullmatch(pattern, value) is None:
        errors.append(
            f"{source}: expected {'None or ' if allow_none else ''}one or more backticked {prefix}-xxx identifiers"
        )
        return set()
    identifiers = set(re.findall(rf"`({prefix}-\d{{3}})`", value))
    return identifiers


def _canonical_specification_anchor(
    value: str,
    source_name: str,
    errors: list[str],
) -> CanonicalSpecificationAnchor | None:
    match = CANONICAL_SPECIFICATION_ANCHOR_RE.fullmatch(value)
    if match is None:
        errors.append(f"{source_name}: specification Anchor must use canonical SPEC-xxx/M-xxx:N[-M] form")
        return None
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    if end < start:
        errors.append(f"{source_name}: canonical specification Anchor line range must be increasing")
        return None
    source = match.group("source")
    return CanonicalSpecificationAnchor(
        source=source,
        member=f"{source}/{match.group('member')}",
        start=start,
        end=end,
    )


def _canonical_specification_anchor_list(
    value: str,
    source_name: str,
    errors: list[str],
) -> list[CanonicalSpecificationAnchor]:
    if re.fullmatch(r"`[^`\n]+`(?:;\s*`[^`\n]+`)*", value) is None:
        errors.append(
            f"{source_name}: Spec anchor must contain only semicolon-separated "
            "backticked canonical specification Anchors"
        )
        return []
    anchors: list[CanonicalSpecificationAnchor] = []
    for raw_anchor in re.findall(r"`([^`\n]+)`", value):
        anchor = _canonical_specification_anchor(raw_anchor, source_name, errors)
        if anchor is not None:
            anchors.append(anchor)
    return anchors


def _parse_pinned_spec_members(
    pinned: str,
    spec_ids: list[str],
    errors: list[str],
) -> dict[str, PinnedSpecificationMember]:
    members: dict[str, PinnedSpecificationMember] = {}
    ordered_ids: list[str] = []
    identities_by_source: dict[str, set[str]] = {}
    current_source: str | None = None

    for line in pinned.splitlines():
        specification = PINNED_SPEC_RE.fullmatch(line)
        if specification is not None:
            current_source = specification.group(1)
            continue
        if not line.startswith("  - Member"):
            continue
        member = PINNED_SPEC_MEMBER_RE.fullmatch(line)
        if member is None:
            errors.append(
                "report.md: every pinned specification member must use the exact Member/identity/Bytes/SHA-256 form"
            )
            continue
        identifier = member.group("identifier")
        source = identifier.split("/", 1)[0]
        identity = member.group("identity")
        if current_source is None or source != current_source:
            errors.append(f"report.md: pinned member {identifier} is not nested under its source {source}")
        if source not in spec_ids:
            errors.append(f"report.md: pinned member {identifier} references unpinned source {source}")
        if identifier in members:
            errors.append(f"report.md: duplicate pinned specification member {identifier}")
        else:
            members[identifier] = PinnedSpecificationMember(
                identifier=identifier,
                source=source,
                identity=identity,
                size=int(member.group("size")),
            )
            ordered_ids.append(identifier)
        source_identities = identities_by_source.setdefault(source, set())
        if identity in source_identities:
            errors.append(f"report.md: duplicate pinned member identity {identity!r} for {source}")
        source_identities.add(identity)

    for source in spec_ids:
        source_members = sorted(identifier for identifier, member in members.items() if member.source == source)
        if not source_members:
            errors.append(f"report.md: pinned specification {source} requires at least one member with byte length")
            continue
        expected = [f"{source}/M-{number:03d}" for number in range(1, len(source_members) + 1)]
        if source_members != expected:
            errors.append(f"report.md: pinned members for {source} must use contiguous M-001..M-N identities")

    expected_order = sorted(
        ordered_ids,
        key=lambda identifier: tuple(int(part.split("-", 1)[1]) for part in identifier.split("/")),
    )
    if ordered_ids != expected_order:
        errors.append("report.md: pinned specification members are not in stable SPEC/member order")
    return members


def _parse_source_coverage_ledger(
    coverage: str,
    pinned_spec_ids: set[str],
    pinned_members: dict[str, PinnedSpecificationMember],
    errors: list[str],
) -> tuple[dict[str, SourceUnitEntry], set[str]]:
    body = _subsection_body(
        coverage,
        "Source Coverage Ledger",
        "report.md: Coverage",
        errors,
    )
    entries: dict[str, SourceUnitEntry] = {}
    nonverifiable: set[str] = set()
    for identifier, block in _ledger_blocks(
        body,
        r"U-\d{3}",
        "Source Coverage Ledger",
        errors,
    ):
        source_name = f"report.md: Source Coverage Ledger {identifier}"
        source_value = _field_value(block, "Source", source_name, errors)
        member_value = _field_value(block, "Member", source_name, errors)
        anchor = _field_value(block, "Anchor", source_name, errors)
        byte_range = _field_value(block, "Byte range", source_name, errors)
        disposition = _field_value(block, "Disposition", source_name, errors)
        requirements_value = _field_value(
            block,
            "Requirements",
            source_name,
            errors,
        )
        reason = _field_value(block, "Reason", source_name, errors)

        source_id = ""
        if source_value is not None:
            match = re.fullmatch(r"`(SPEC-\d{3})`", source_value)
            if match is None:
                errors.append(f"{source_name}: Source must be one backticked SPEC-xxx")
            else:
                source_id = match.group(1)
                if source_id not in pinned_spec_ids:
                    errors.append(f"{source_name}: references unpinned source {source_id}")
        member_id = ""
        if member_value is not None:
            match = re.fullmatch(r"`(SPEC-\d{3}/M-\d{3})`", member_value)
            if match is None:
                errors.append(f"{source_name}: Member must be one backticked SPEC-xxx/M-xxx")
            else:
                member_id = match.group(1)
                pinned_member = pinned_members.get(member_id)
                if pinned_member is None:
                    errors.append(f"{source_name}: references undeclared pinned member {member_id}")
                elif source_id and pinned_member.source != source_id:
                    errors.append(f"{source_name}: Source {source_id} does not own Member {member_id}")
        canonical_anchor: CanonicalSpecificationAnchor | None = None
        if anchor is not None:
            anchor_match = re.fullmatch(r"`([^`\n]+)`", anchor)
            if anchor_match is None:
                errors.append(f"{source_name}: Anchor must be one backticked canonical specification Anchor")
            else:
                canonical_anchor = _canonical_specification_anchor(
                    anchor_match.group(1),
                    source_name,
                    errors,
                )
        if canonical_anchor is not None:
            if canonical_anchor.source != source_id:
                errors.append(
                    f"{source_name}: Anchor source {canonical_anchor.source} does not match Source {source_id}"
                )
            if canonical_anchor.member != member_id:
                errors.append(
                    f"{source_name}: Anchor member {canonical_anchor.member} does not match Member {member_id}"
                )
            if canonical_anchor.member not in pinned_members:
                errors.append(f"{source_name}: Anchor references undeclared pinned member {canonical_anchor.member}")
        byte_start: int | None = None
        byte_end: int | None = None
        if byte_range is not None:
            match = re.fullmatch(r"(\d+)-(\d+)", byte_range)
            if match is None or int(match.group(1)) >= int(match.group(2)):
                errors.append(f"{source_name}: Byte range must be an increasing start-end pair")
            else:
                byte_start = int(match.group(1))
                byte_end = int(match.group(2))
        if disposition is not None and disposition not in SOURCE_DISPOSITIONS:
            errors.append(f"{source_name}: unsupported Source Unit disposition {disposition!r}")
        requirements = set()
        if requirements_value is not None:
            requirements = _parse_code_id_list(
                requirements_value,
                "R",
                source_name,
                errors,
                allow_none=True,
            )
        if disposition == "mapped" and not requirements:
            errors.append(f"{source_name}: mapped Source Unit requires Requirement IDs")
        if disposition in {"non-requirement", "non-verifiable", "unfinished"} and requirements:
            errors.append(f"{source_name}: {disposition} Source Unit must use Requirements: None")
        if reason is None:
            pass  # The field error above is sufficient.
        if identifier not in entries:
            entries[identifier] = SourceUnitEntry(
                identifier=identifier,
                source=source_id,
                member=member_id,
                byte_start=byte_start,
                byte_end=byte_end,
                disposition=disposition or "",
                requirements=requirements,
            )
        if disposition == "non-verifiable":
            nonverifiable.add(identifier)
    return entries, nonverifiable


def _lint_member_partitions(
    source_units: dict[str, SourceUnitEntry],
    pinned_members: dict[str, PinnedSpecificationMember],
    *,
    require_complete: bool,
    errors: list[str],
) -> None:
    ordered = [
        entry
        for entry in source_units.values()
        if entry.member in pinned_members and entry.byte_start is not None and entry.byte_end is not None
    ]
    expected_order = sorted(
        ordered,
        key=lambda entry: (
            entry.member,
            entry.byte_start,
            entry.byte_end,
            entry.identifier,
        ),
    )
    if [entry.identifier for entry in ordered] != [entry.identifier for entry in expected_order]:
        errors.append("report.md: Source Coverage Ledger must use stable member, byte-offset, and Source Unit order")

    by_member: dict[str, list[SourceUnitEntry]] = {identifier: [] for identifier in pinned_members}
    for entry in ordered:
        by_member[entry.member].append(entry)

    for member_id, member in sorted(pinned_members.items()):
        cursor = 0
        member_units = sorted(
            by_member[member_id],
            key=lambda entry: (
                entry.byte_start if entry.byte_start is not None else -1,
                entry.byte_end if entry.byte_end is not None else -1,
                entry.identifier,
            ),
        )
        for unit in member_units:
            assert unit.byte_start is not None
            assert unit.byte_end is not None
            if unit.byte_end > member.size:
                errors.append(
                    f"report.md: Source Unit {unit.identifier} ends past EOF {member.size} of pinned member {member_id}"
                )
            if unit.byte_start < cursor:
                errors.append(
                    f"report.md: Source Unit {unit.identifier} overlaps the previous range of pinned member {member_id}"
                )
            elif require_complete and unit.byte_start != cursor:
                errors.append(
                    f"report.md: complete coverage has a byte gap {cursor}-"
                    f"{unit.byte_start} in pinned member {member_id}"
                )
            cursor = max(cursor, unit.byte_end)
        if require_complete and cursor != member.size:
            errors.append(
                f"report.md: complete coverage for pinned member {member_id} ends "
                f"at byte {cursor}, not exact EOF {member.size}"
            )


def _parse_published_disposition(
    value: str,
    kind: str,
) -> set[str] | None:
    match = re.fullmatch(
        rf"published-as-({kind}-\d{{3}}(?:,\s*{kind}-\d{{3}})*)",
        value,
    )
    if match is None:
        return None
    return set(re.findall(rf"{kind}-\d{{3}}", match.group(1)))


def _is_unsafe_repository_path(path_value: str) -> bool:
    """Return whether a path can escape or misaddress the repository."""

    posix = PurePosixPath(path_value)
    windows = PureWindowsPath(path_value)
    return not path_value or any(
        (
            path_value in {".", ".."},
            posix.is_absolute(),
            windows.is_absolute(),
            bool(windows.drive),
            ".." in posix.parts,
            "\\" in path_value,
        )
    )


def _lint_implementation_evidence(
    body: str,
    *,
    minimum: int,
    source_name: str,
    errors: list[str],
) -> None:
    records = [line.strip() for line in body.splitlines() if line.strip().startswith("- Implementation evidence:")]
    valid = 0
    for line in records:
        match = IMPLEMENTATION_EVIDENCE_LINE_RE.fullmatch(line)
        if match is None:
            errors.append(f"{source_name}: malformed structured Implementation evidence record")
            continue
        path_value = match.group("source").strip()
        if _is_unsafe_repository_path(path_value):
            errors.append(f"{source_name}: Implementation evidence Source must be a safe repository-relative path")
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if start < 1 or end < start:
            errors.append(
                f"{source_name}: Implementation evidence Lines must be a positive 1-based line or increasing range"
            )
            continue
        quote = match.group("quote")
        if not quote.strip():
            errors.append(f"{source_name}: Implementation evidence Quote must be non-empty")
            continue
        if match.group("fence") in quote:
            errors.append(
                f"{source_name}: Implementation evidence Quote must be one inline-code "
                "span; choose a longer fence instead of embedding the closing fence"
            )
            continue
        valid += 1
    if valid < minimum:
        errors.append(f"{source_name}: requires at least {minimum} structured Implementation evidence record(s)")


def _parse_requirement_assessment_ledger(
    coverage: str,
    pinned_members: dict[str, PinnedSpecificationMember],
    errors: list[str],
) -> dict[str, RequirementEntry]:
    body = _subsection_body(
        coverage,
        "Requirement Assessment Ledger",
        "report.md: Coverage",
        errors,
    )
    entries: dict[str, RequirementEntry] = {}
    for identifier, block in _ledger_blocks(
        body,
        r"R-\d{3}",
        "Requirement Assessment Ledger",
        errors,
    ):
        source_name = f"report.md: Requirement Assessment Ledger {identifier}"
        units_value = _field_value(block, "Units", source_name, errors)
        spec_anchor = _field_value(block, "Spec anchor", source_name, errors)
        claim = _field_value(block, "Claim", source_name, errors)
        assessment = _field_value(block, "Assessment", source_name, errors)
        basis = _field_value(block, "Basis", source_name, errors)
        boundary_quantifier = _field_value(
            block,
            "Boundary/quantifier",
            source_name,
            errors,
        )
        guard_challenge = _field_value(
            block,
            "Guard challenge",
            source_name,
            errors,
        )
        counter_search = _field_value(block, "Counter-search", source_name, errors)
        disposition = _field_value(block, "Disposition", source_name, errors)

        _lint_implementation_evidence(
            block,
            minimum=1 if assessment in {"consistent", "inconsistent"} else 0,
            source_name=source_name,
            errors=errors,
        )

        trace_markers = re.findall(r"^- Trace:\s*$", block, re.MULTILINE)
        if len(trace_markers) != 1:
            errors.append(f"{source_name}: requires exactly one '- Trace:' marker")
        _nested_field_value(block, "Input/precondition", source_name, errors)
        _nested_field_value(block, "Required", source_name, errors)
        _nested_field_value(block, "Observed", source_name, errors)
        comparison = _nested_field_value(block, "Comparison", source_name, errors)

        units = set()
        if units_value is not None:
            units = _parse_code_id_list(
                units_value,
                "U",
                source_name,
                errors,
                allow_none=False,
            )
        specifications: set[str] = set()
        specification_members: set[str] = set()
        if spec_anchor is not None:
            anchors = _canonical_specification_anchor_list(
                spec_anchor,
                source_name,
                errors,
            )
            specifications = {anchor.source for anchor in anchors}
            specification_members = {anchor.member for anchor in anchors}
            for canonical_anchor in anchors:
                pinned_member = pinned_members.get(canonical_anchor.member)
                if pinned_member is None:
                    errors.append(
                        f"{source_name}: Spec anchor references undeclared pinned member {canonical_anchor.member}"
                    )
                elif pinned_member.source != canonical_anchor.source:
                    errors.append(
                        f"{source_name}: Spec anchor member "
                        f"{canonical_anchor.member} does not belong to Source "
                        f"{canonical_anchor.source}"
                    )
        if claim is None or basis is None or boundary_quantifier is None or counter_search is None:
            pass  # Individual field errors above are sufficient.
        guard_match = GUARD_CHALLENGE_RE.fullmatch(guard_challenge) if guard_challenge is not None else None
        guard_values: tuple[str, str, str, str] | None = None
        if guard_challenge is not None and guard_match is None:
            errors.append(
                f"{source_name}: Guard challenge must use Guards, Authoritative "
                "narrowing, First changed input, and Comparison in the exact contract"
            )
        elif guard_match is not None:
            guard_values = (
                guard_match.group("guards"),
                guard_match.group("narrowing"),
                guard_match.group("changed"),
                guard_match.group("comparison"),
            )
            guards, narrowing, changed, _ = guard_values
            for label, value in (
                ("Guards", guards),
                ("Authoritative narrowing", narrowing),
                ("First changed input", changed),
            ):
                if value.casefold() in {"`none`", "`none.`"}:
                    errors.append(
                        f"{source_name}: {label} must use the bare None sentinel, not a backticked placeholder"
                    )
            if guards == "None" and (narrowing != "None" or changed != "None"):
                errors.append(
                    f"{source_name}: Guard challenge with Guards=None must also use "
                    "Authoritative narrowing=None and First changed input=None"
                )
            if guards != "None" and changed == "None":
                errors.append(f"{source_name}: a discovered Guard challenge requires the first behavior-changing input")
            guard_comparison = guard_values[3]
            if guards == "None" and guard_comparison != "same":
                errors.append(f"{source_name}: Guard challenge with Guards=None requires Comparison=same")
            if guards != "None" and narrowing == "None" and guard_comparison != "different":
                errors.append(
                    f"{source_name}: a claim-affecting guard without authoritative "
                    "narrowing and with a changed-input witness requires "
                    "Comparison=different"
                )
        if assessment is not None and assessment not in TERMINAL_ASSESSMENTS:
            errors.append(f"{source_name}: unsupported terminal Assessment {assessment!r}")

        dossier_ids: set[str] = set()
        if assessment == "consistent":
            if disposition != "no-finding":
                errors.append(f"{source_name}: consistent requires Disposition: no-finding")
            if comparison != "same":
                errors.append(f"{source_name}: consistent requires Comparison: same")
            if guard_values is not None:
                guards, narrowing, _, guard_comparison = guard_values
                if guard_comparison != "same":
                    errors.append(f"{source_name}: consistent requires Guard challenge Comparison=same")
                if guards != "None" and narrowing == "None":
                    errors.append(
                        f"{source_name}: consistent cannot retain a discovered guard "
                        "without exact authoritative narrowing"
                    )
        elif assessment == "inconsistent":
            published = _parse_published_disposition(disposition, "F") if disposition is not None else None
            if not published:
                errors.append(f"{source_name}: inconsistent requires published-as-F-xxx disposition")
            else:
                dossier_ids = published
            if comparison != "different":
                errors.append(f"{source_name}: inconsistent requires Comparison: different")
        elif assessment in {"potential-or-uncertain", "non-verifiable"}:
            published = _parse_published_disposition(disposition, "S") if disposition is not None else None
            if disposition != "reported-as-uncertain" and not published:
                errors.append(
                    f"{source_name}: {assessment} requires reported-as-uncertain or published-as-S-xxx disposition"
                )
            if published:
                dossier_ids = published
            if comparison != "undetermined":
                errors.append(f"{source_name}: {assessment} requires Comparison: undetermined")
        elif assessment == "failure-or-partial":
            if disposition != "unfinished":
                errors.append(f"{source_name}: failure-or-partial requires Disposition: unfinished")
            if comparison != "undetermined":
                errors.append(f"{source_name}: failure-or-partial requires Comparison: undetermined")
        if comparison is not None and comparison not in {"same", "different", "undetermined"}:
            errors.append(f"{source_name}: unsupported Trace Comparison {comparison!r}")
        if guard_values is not None and guard_values[3] == "different" and assessment != "inconsistent":
            errors.append(
                f"{source_name}: Guard challenge Comparison=different requires an inconsistent Requirement assessment"
            )

        if identifier not in entries:
            entries[identifier] = RequirementEntry(
                identifier=identifier,
                units=units,
                specifications=specifications,
                members=specification_members,
                assessment=assessment or "",
                comparison=comparison or "",
                disposition=disposition or "",
                dossier_ids=dossier_ids,
            )
    return entries


def _crosscheck_coverage_ledgers(
    ledger: CoverageLedger,
    pinned_spec_ids: set[str],
    errors: list[str],
) -> None:
    for unit_id, unit in ledger.source_units.items():
        if unit.disposition != "mapped":
            continue
        for requirement_id in unit.requirements:
            requirement = ledger.requirements.get(requirement_id)
            if requirement is None:
                errors.append(f"report.md: Source Unit {unit_id} references missing Requirement {requirement_id}")
            elif unit_id not in requirement.units:
                errors.append(
                    f"report.md: Source Unit {unit_id} and Requirement {requirement_id} are not bidirectionally linked"
                )

    for requirement_id, requirement in ledger.requirements.items():
        unit_sources: set[str] = set()
        unit_members: set[str] = set()
        for unit_id in requirement.units:
            unit = ledger.source_units.get(unit_id)
            if unit is None:
                errors.append(f"report.md: Requirement {requirement_id} references missing Source Unit {unit_id}")
                continue
            unit_sources.add(unit.source)
            unit_members.add(unit.member)
            if unit.disposition != "mapped" or requirement_id not in unit.requirements:
                errors.append(
                    f"report.md: Requirement {requirement_id} and Source Unit {unit_id} are not bidirectionally mapped"
                )
        if not requirement.units:
            errors.append(f"report.md: Requirement {requirement_id} has no Source Units")
        if requirement.specifications - pinned_spec_ids:
            rendered = ", ".join(sorted(requirement.specifications - pinned_spec_ids))
            errors.append(f"report.md: Requirement {requirement_id} references unpinned specifications: {rendered}")
        if unit_sources and requirement.specifications != unit_sources:
            errors.append(
                f"report.md: Requirement {requirement_id} Spec anchor does not match its Source Unit specifications"
            )
        if unit_members and requirement.members != unit_members:
            errors.append(f"report.md: Requirement {requirement_id} Spec anchor members do not match its Source Units")

        if requirement.assessment == "inconsistent":
            for dossier_id in requirement.dossier_ids:
                ledger.finding_requirements.setdefault(dossier_id, set()).add(requirement_id)
        if requirement.assessment in {"potential-or-uncertain", "non-verifiable"}:
            ledger.terminal_uncertainty.add(requirement_id)
            for dossier_id in requirement.dossier_ids:
                ledger.conflict_requirements.setdefault(dossier_id, set()).add(requirement_id)
        if requirement.assessment == "failure-or-partial":
            ledger.partial_requirements.add(requirement_id)


def _parse_work_packet_ledger(
    coverage: str,
    pinned_spec_ids: set[str],
    errors: list[str],
) -> tuple[set[str], set[str]]:
    """Validate generic discovery packet accounting without judging semantics."""

    heading = "### Audit Work Packet Ledger"
    if coverage.count(heading) != 1:
        errors.append("report.md: Coverage requires exactly one '### Audit Work Packet Ledger'")
        return set(), set()
    packet_text = coverage.split(heading, 1)[1]
    next_heading = re.search(r"^### ", packet_text, re.MULTILINE)
    if next_heading is not None:
        packet_text = packet_text[: next_heading.start()]

    packets = list(WORK_PACKET_RE.finditer(packet_text))
    if not packets:
        errors.append("report.md: Audit Work Packet Ledger requires at least one P-xxx packet")
        return set(), set()

    identifiers = [match.group("identifier") for match in packets]
    if len(identifiers) != len(set(identifiers)):
        errors.append("report.md: duplicate Audit Work Packet ID")

    covered: set[str] = set()
    unfinished: set[str] = set()
    field_patterns = {
        "scope": r"^- Specification scope:\s*(\S.*)$",
        "repository": r"^- Repository scope:\s*(\S.*)$",
        "classes": r"^- Discovery classes:\s*(\S.*)$",
        "search": r"^- Search record:\s*(\S.*)$",
        "state": r"^- State:\s*(complete|unfinished)\s*$",
        "result": r"^- Result:\s*(\S.*)$",
    }
    for match in packets:
        identifier = match.group("identifier")
        body = match.group("body")
        values: dict[str, str] = {}
        for field_name, pattern in field_patterns.items():
            matches = re.findall(pattern, body, re.MULTILINE)
            if len(matches) != 1:
                errors.append(f"report.md: Audit Work Packet {identifier} requires exactly one {field_name} field")
                continue
            values[field_name] = matches[0].strip()

        scope = values.get("scope", "")
        scoped_specs = re.findall(r"`(SPEC-\d{3})`", scope)
        residue = re.sub(r"`SPEC-\d{3}`", "", scope).replace(";", "").strip()
        if not scoped_specs or residue:
            errors.append(
                f"report.md: Audit Work Packet {identifier} Specification scope must "
                "contain only semicolon-separated backticked SPEC-xxx IDs"
            )
        if len(scoped_specs) > 4:
            errors.append(f"report.md: Audit Work Packet {identifier} covers more than four Specification Sources")
        if len(scoped_specs) != len(set(scoped_specs)):
            errors.append(f"report.md: Audit Work Packet {identifier} repeats a Specification Source")
        unknown = set(scoped_specs) - pinned_spec_ids
        if unknown:
            errors.append(
                f"report.md: Audit Work Packet {identifier} references unpinned "
                f"specifications: {', '.join(sorted(unknown))}"
            )
        covered.update(scoped_specs)

        classes = {item.strip() for item in values.get("classes", "").split(",") if item.strip()}
        if classes != WORK_PACKET_CLASSES:
            errors.append(
                f"report.md: Audit Work Packet {identifier} Discovery classes must be "
                "exactly normative, boundary, state, routing, capability, alternatives"
            )
        if values.get("state") == "unfinished":
            unfinished.add(identifier)

    missing = pinned_spec_ids - covered
    if missing:
        errors.append(
            "report.md: Audit Work Packet Ledger does not cover pinned specifications: " + ", ".join(sorted(missing))
        )
    return covered, unfinished


def _lint_coverage_contract(text: str, errors: list[str]) -> CoverageLedger:
    pinned = _section_body(text, "Pinned Inputs")
    coverage = _section_body(text, "Coverage")
    validation = _section_body(text, "Validation")
    uncertainty = _section_body(text, "Uncertain and Unfinished Work").strip()
    status_matches = re.findall(
        r"^-\s*Result:\s*(Complete|Partial)\s*$",
        text,
        re.MULTILINE,
    )

    repository_coverage = REPOSITORY_COVERAGE_RE.findall(coverage)
    if len(repository_coverage) != 1:
        errors.append(
            "report.md: Coverage requires exactly one '- Repository inventory: Complete' "
            "or '- Repository inventory: Incomplete'"
        )
    specification_coverage = SPECIFICATION_COVERAGE_RE.findall(coverage)
    if len(specification_coverage) != 1:
        errors.append(
            "report.md: Coverage requires exactly one '- Specification source coverage: Complete' "
            "or '- Specification source coverage: Incomplete'"
        )

    repositories = PINNED_REPOSITORY_RE.findall(pinned)
    if len(repositories) != 1:
        errors.append("report.md: Pinned Inputs requires one repository Pin SHA-256")

    specs = PINNED_SPEC_RE.findall(pinned)
    spec_ids = [item[0] for item in specs]
    if not spec_ids:
        errors.append("report.md: Pinned Inputs requires at least one labeled specification hash")
    if len(spec_ids) != len(set(spec_ids)):
        errors.append("report.md: duplicate pinned specification label")
    pinned_members = _parse_pinned_spec_members(pinned, spec_ids, errors)

    rows = SOURCE_UNIT_RE.findall(coverage)
    row_ids = [item[0] for item in rows]
    if len(row_ids) != len(set(row_ids)):
        errors.append("report.md: duplicate Source unit ledger row")
    if set(row_ids) != set(spec_ids):
        errors.append("report.md: Source unit ledger must cover every pinned specification exactly once")

    pinned_spec_ids = set(spec_ids)
    _packet_specs, unfinished_packets = _parse_work_packet_ledger(
        coverage,
        pinned_spec_ids,
        errors,
    )
    source_units, nonverifiable_source_units = _parse_source_coverage_ledger(
        coverage,
        pinned_spec_ids,
        pinned_members,
        errors,
    )
    _lint_member_partitions(
        source_units,
        pinned_members,
        require_complete=specification_coverage == ["Complete"],
        errors=errors,
    )
    requirements = _parse_requirement_assessment_ledger(
        coverage,
        pinned_members,
        errors,
    )
    ledger = CoverageLedger(
        source_units=source_units,
        requirements=requirements,
        pinned_members=pinned_members,
        nonverifiable_source_units=nonverifiable_source_units,
    )
    _crosscheck_coverage_ledgers(ledger, pinned_spec_ids, errors)

    unfinished_total = 0
    empty_source_units: list[str] = []
    aggregate: dict[str, tuple[int, int, int, int]] = {}
    for source_id, total, mapped, classified, unfinished in rows:
        values = tuple(int(value) for value in (total, mapped, classified, unfinished))
        aggregate[source_id] = values
        if values[0] != sum(values[1:]):
            errors.append(f"report.md: Source unit ledger for {source_id} is unbalanced")
        if values[0] == 0:
            empty_source_units.append(source_id)
        unfinished_total += values[3]

    detailed: dict[str, list[SourceUnitEntry]] = {spec_id: [] for spec_id in pinned_spec_ids}
    for entry in source_units.values():
        if entry.source in detailed:
            detailed[entry.source].append(entry)
    for spec_id, values in aggregate.items():
        entries = detailed.get(spec_id, [])
        actual = (
            len(entries),
            sum(entry.disposition == "mapped" for entry in entries),
            sum(entry.disposition in {"non-requirement", "non-verifiable"} for entry in entries),
            sum(entry.disposition == "unfinished" for entry in entries),
        )
        if values != actual:
            errors.append(
                f"report.md: aggregate Source Unit counts for {spec_id} do not match the Source Coverage Ledger"
            )

    if len(status_matches) != 1:
        return ledger
    status = status_matches[0]
    if status == "Complete":
        required_lines = (
            (coverage, "- Repository inventory: Complete"),
            (coverage, "- Specification source coverage: Complete"),
            (validation, "- Inventory verification: Passed"),
            (validation, "- Report lint: Passed"),
        )
        for body, line in required_lines:
            if not _contains_line(body, line):
                errors.append(f"report.md: Complete status requires {line!r}")
        if unfinished_total:
            errors.append("report.md: Complete status requires zero unfinished Source Units")
        if unfinished_packets:
            errors.append(
                "report.md: Complete status requires zero unfinished Audit Work Packets: "
                + ", ".join(sorted(unfinished_packets))
            )
        if empty_source_units:
            rendered = ", ".join(sorted(empty_source_units))
            errors.append(
                "report.md: Complete status requires at least one Source Unit for every "
                f"pinned specification (empty: {rendered})"
            )
        if not re.search(r"^- Semantically inspected scope:\s+\S", coverage, re.MULTILINE):
            errors.append("report.md: Complete status requires a non-empty semantically inspected scope")
        if ledger.partial_requirements:
            rendered = ", ".join(sorted(ledger.partial_requirements))
            errors.append(
                f"report.md: Complete status cannot contain failure-or-partial Requirement assessments: {rendered}"
            )
        has_terminal_uncertainty = bool(ledger.terminal_uncertainty or ledger.nonverifiable_source_units)
        if has_terminal_uncertainty and uncertainty in NONE_MARKERS:
            errors.append(
                "report.md: terminal uncertainty or non-verifiable Source Units "
                "must be explicit under Uncertain and Unfinished Work"
            )
        if not has_terminal_uncertainty and uncertainty not in NONE_MARKERS:
            errors.append(
                "report.md: Complete status reports uncertainty without a matching terminal ledger assessment"
            )
    else:
        if uncertainty in NONE_MARKERS:
            errors.append("report.md: Partial status requires explicit uncertain or unfinished work")
        structural_incomplete = any(
            (
                repository_coverage == ["Incomplete"],
                specification_coverage == ["Incomplete"],
                bool(unfinished_total),
                bool(unfinished_packets),
                bool(empty_source_units),
                bool(ledger.partial_requirements),
                not _contains_line(validation, "- Inventory verification: Passed"),
                not _contains_line(validation, "- Report lint: Passed"),
                not bool(
                    re.search(
                        r"^- Semantically inspected scope:\s+\S",
                        coverage,
                        re.MULTILINE,
                    )
                ),
            )
        )
        if not structural_incomplete:
            errors.append(
                "report.md: Partial status requires a structural incomplete signal; "
                "Static-Only mode alone is not incomplete"
            )
    return ledger


def _parse_count_section(
    text: str,
    heading: str,
    errors: list[str],
) -> tuple[int | None, str]:
    pattern = re.compile(
        rf"^## {re.escape(heading)} \((\d+)\)\s*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        errors.append(f"report.md: missing or malformed '## {heading} (N)' section")
        return None, ""
    if len(matches) > 1:
        errors.append(f"report.md: duplicate '## {heading} (N)' section")

    match = matches[0]
    next_heading = re.search(r"^##\s+", text[match.end() :], re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(text)
    body = text[match.end() : end]
    if not body.strip():
        errors.append(f"report.md: '## {heading} (N)' section is empty")
    return int(match.group(1)), body


def _parse_dossier_links(
    section: str,
    expected_kind: str,
    section_name: str,
    errors: list[str],
) -> list[str]:
    identifiers: list[str] = []
    for label, target in MARKDOWN_LINK_RE.findall(section):
        looks_like_dossier = target.startswith("findings/") or re.match(
            r"^[FS]-",
            label,
        )
        if not looks_like_dossier:
            continue

        target_match = re.fullmatch(r"findings/([FS]-\d{3})\.md", target)
        if target_match is None:
            errors.append(
                f"report.md: dossier link must be relative and use findings/{expected_kind}-xxx.md: {target!r}"
            )
            continue

        identifier = target_match.group(1)
        label_match = re.match(r"^([FS]-\d{3})(?:\b|:)", label)
        if label_match is None or label_match.group(1) != identifier:
            errors.append(f"report.md: link label and target identifier do not match: {label!r} -> {target!r}")
        if not identifier.startswith(f"{expected_kind}-"):
            errors.append(f"report.md: {identifier} is linked from the {section_name} section")
        identifiers.append(identifier)

    if len(identifiers) != len(set(identifiers)):
        errors.append(f"report.md: duplicate dossier link in {section_name}")
    return identifiers


def _inventory_dossiers(findings_dir: Path, errors: list[str]) -> dict[str, Path]:
    dossiers: dict[str, Path] = {}
    try:
        entries = sorted(findings_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ToolError(f"cannot list findings directory {findings_dir}: {exc}") from exc

    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            errors.append(f"findings/: unexpected non-file entry {entry.name!r}")
            continue
        match = DOSSIER_NAME_RE.fullmatch(entry.name)
        if match is None:
            errors.append(f"findings/: invalid dossier filename {entry.name!r}")
            continue
        dossiers[entry.stem] = entry
    return dossiers


def _lint_specification_anchors(
    body: str,
    minimum: int,
    pinned_spec_ids: set[str] | None,
    pinned_members: dict[str, PinnedSpecificationMember] | None,
    source: str,
    section: str,
    errors: list[str],
) -> None:
    bullet_lines = [line.strip() for line in body.splitlines() if line.strip().startswith("-")]
    anchors: list[tuple[str, str, str, str]] = []
    malformed = 0
    for line in bullet_lines:
        match = SPECIFICATION_ANCHOR_LINE_RE.fullmatch(line)
        if match is None:
            malformed += 1
            continue
        values = tuple(value.strip() for value in match.groups())
        if any(not value for value in values):
            malformed += 1
            continue
        anchors.append(values)

    if malformed:
        errors.append(
            f"{source}: every bullet in '## {section}' must use the structured "
            "Source/Declared provenance/Anchor/Quote form"
        )
    if len(anchors) < minimum:
        errors.append(f"{source}: '## {section}' requires at least {minimum} structured specification anchor record(s)")
    for spec_id, _provenance, anchor_value, _quote in anchors:
        if pinned_spec_ids is not None and spec_id not in pinned_spec_ids:
            errors.append(f"{source}: specification anchor references unpinned source {spec_id}")
        canonical_anchor = _canonical_specification_anchor(
            anchor_value,
            source,
            errors,
        )
        if canonical_anchor is None:
            continue
        if pinned_spec_ids is None or pinned_members is None:
            continue
        pinned_member = pinned_members.get(canonical_anchor.member)
        if canonical_anchor.source != spec_id:
            errors.append(
                f"{source}: specification Anchor member {canonical_anchor.member} does not belong to Source {spec_id}"
            )
        elif pinned_member is None:
            errors.append(
                f"{source}: specification Anchor references undeclared pinned member {canonical_anchor.member}"
            )
        elif pinned_member.source != spec_id:
            errors.append(
                f"{source}: specification Anchor member {canonical_anchor.member} does not belong to Source {spec_id}"
            )


def lint_candidate_dossier(path: Path) -> tuple[list[str], str | None]:
    """Validate one Pending F/S draft and return its digest when structurally valid."""

    dossier, text = _read_dossier(path)
    errors: list[str] = []
    source = f"candidate/{path.name}"
    filename = DOSSIER_NAME_RE.fullmatch(path.name)
    if filename is None:
        errors.append(f"{source}: filename must be F-xxx.md or S-xxx.md")
        return errors, None

    identifier = path.stem
    kind = filename.group("kind")
    first_line = _first_nonempty_line(text)
    expected_title = re.compile(rf"^# {re.escape(identifier)}:\s+\S.*$")
    if first_line is None or expected_title.fullmatch(first_line) is None:
        errors.append(f"{source}: title must begin '# {identifier}: '")
    if PLACEHOLDER_RE.search(text):
        errors.append(f"{source}: contains an unreplaced template placeholder")

    affected = set(re.findall(r"\bR-\d{3}\b", _section_body(text, "Affected Requirements")))
    if not affected:
        errors.append(f"{source}: Affected Requirements must name at least one R-xxx")

    if kind == "F":
        required = FINDING_SECTIONS
        forbidden = (
            "Conflicting Specification Anchors",
            "Applicability Overlap",
            "Precedence Search",
            "Specification Conflict Chain",
        )
        specification_section = "Specification Evidence"
        specification_minimum = 1
    else:
        required = CONFLICT_SECTIONS
        forbidden = (
            "Root Cause",
            "Specification Evidence",
            "Implementation and Probe Evidence",
            "Contradiction Chain",
            "Counter-Search",
        )
        specification_section = "Conflicting Specification Anchors"
        specification_minimum = 2

    _require_sections(text, required, source, errors)
    _require_heading_order(text, required, source, errors)
    for heading in forbidden:
        if _heading_count(text, heading):
            errors.append(f"{source}: {kind} dossier contains other-type section '## {heading}'")
    _lint_specification_anchors(
        _section_body(text, specification_section),
        specification_minimum,
        None,
        None,
        source,
        specification_section,
        errors,
    )
    if kind == "F":
        _lint_implementation_evidence(
            _section_body(text, "Implementation and Probe Evidence"),
            minimum=1,
            source_name=source,
            errors=errors,
        )

    exact_review_headings = list(CANDIDATE_REVIEW_HEADING_RE.finditer(dossier))
    if len(exact_review_headings) != 1:
        errors.append(f"{source}: must contain exactly one exact '## Adversarial Review' H2")
    review_span = _section_span(text, "Adversarial Review")
    review = review_span[2] if review_span is not None else ""
    outcomes = OUTCOME_RE.findall(review)
    if outcomes != ["Pending"]:
        errors.append(f"{source}: candidate Adversarial Review Outcome must be exactly Pending")
    digest_fields = DIGEST_FIELD_RE.findall(review)
    if digest_fields != ["Pending"]:
        errors.append(f"{source}: candidate Adversarial Review Digest must be exactly Pending")
    review_modes = REVIEW_MODE_RE.findall(review)
    if review_modes not in [["Fresh native"], ["Serial falsification"]]:
        errors.append(
            f"{source}: candidate Adversarial Review Mode must be exactly Fresh native or Serial falsification"
        )
    bases = BASIS_RE.findall(review)
    if len(bases) != 1:
        errors.append(f"{source}: candidate Adversarial Review requires exactly one non-empty Basis")
    if review_span is not None:
        outside_review = text[: review_span[0]] + text[review_span[1] :]
        if OUTCOME_RE.search(outside_review):
            errors.append(f"{source}: reviewer Outcome must appear only in '## Adversarial Review'")
        if DIGEST_FIELD_RE.search(outside_review):
            errors.append(f"{source}: reviewer Digest must appear only in '## Adversarial Review'")

    if errors:
        return errors, None
    return [], _candidate_dossier_digest_bytes(dossier, path)


def _lint_dossier(
    identifier: str,
    path: Path,
    pinned_spec_ids: set[str],
    pinned_members: dict[str, PinnedSpecificationMember],
    expected_requirements: set[str],
    errors: list[str],
) -> None:
    text = _read_utf8(path)
    source = f"findings/{path.name}"
    expected_title = re.compile(rf"^# {re.escape(identifier)}:\s+\S.*$")
    first_line = _first_nonempty_line(text)
    if first_line is None or expected_title.fullmatch(first_line) is None:
        errors.append(f"{source}: title must begin '# {identifier}: '")

    if PLACEHOLDER_RE.search(text):
        errors.append(f"{source}: contains an unreplaced template placeholder")

    affected = set(re.findall(r"\bR-\d{3}\b", _section_body(text, "Affected Requirements")))
    if not affected:
        errors.append(f"{source}: Affected Requirements must name at least one R-xxx")
    if affected != expected_requirements:
        errors.append(f"{source}: Affected Requirements do not match the Requirement Assessment Ledger disposition")

    if identifier.startswith("F-"):
        _require_sections(text, FINDING_SECTIONS, source, errors)
        _require_heading_order(text, FINDING_SECTIONS, source, errors)
        for conflicting_heading in (
            "Conflicting Specification Anchors",
            "Applicability Overlap",
            "Precedence Search",
            "Specification Conflict Chain",
        ):
            if _heading_count(text, conflicting_heading):
                errors.append(f"{source}: F dossier contains S-only section '## {conflicting_heading}'")
        _lint_specification_anchors(
            _section_body(text, "Specification Evidence"),
            1,
            pinned_spec_ids,
            pinned_members,
            source,
            "Specification Evidence",
            errors,
        )
        _lint_implementation_evidence(
            _section_body(text, "Implementation and Probe Evidence"),
            minimum=1,
            source_name=source,
            errors=errors,
        )
    else:
        _require_sections(text, CONFLICT_SECTIONS, source, errors)
        _require_heading_order(text, CONFLICT_SECTIONS, source, errors)
        for finding_heading in (
            "Root Cause",
            "Specification Evidence",
            "Implementation and Probe Evidence",
            "Contradiction Chain",
            "Counter-Search",
        ):
            if _heading_count(text, finding_heading):
                errors.append(f"{source}: S dossier contains F-only section '## {finding_heading}'")
        _lint_specification_anchors(
            _section_body(text, "Conflicting Specification Anchors"),
            2,
            pinned_spec_ids,
            pinned_members,
            source,
            "Conflicting Specification Anchors",
            errors,
        )

    review_span = _section_span(text, "Adversarial Review")
    review = review_span[2] if review_span is not None else ""
    outcomes = OUTCOME_RE.findall(review)
    if outcomes != ["Supported"]:
        rendered = ", ".join(outcomes) if outcomes else "missing"
        errors.append(f"{source}: published dossier reviewer outcome must be exactly Supported (found {rendered})")
    if review_span is not None:
        outside_review = text[: review_span[0]] + text[review_span[1] :]
        if OUTCOME_RE.search(outside_review):
            errors.append(f"{source}: reviewer Outcome must appear only in '## Adversarial Review'")
        if DIGEST_FIELD_RE.search(outside_review):
            errors.append(f"{source}: reviewer Digest must appear only in '## Adversarial Review'")
    bases = BASIS_RE.findall(review)
    if len(bases) != 1:
        errors.append(f"{source}: Adversarial Review requires exactly one non-empty Basis")
    review_modes = REVIEW_MODE_RE.findall(review)
    if review_modes not in [["Fresh native"], ["Serial falsification"]]:
        rendered = ", ".join(review_modes) if review_modes else "missing"
        errors.append(
            f"{source}: Adversarial Review Mode must be exactly Fresh native or Serial falsification (found {rendered})"
        )
    digest_fields = DIGEST_FIELD_RE.findall(review)
    digests = DIGEST_RE.findall(review)
    if len(digest_fields) != 1 or len(digests) != 1:
        errors.append(f"{source}: Adversarial Review requires exactly one lowercase 64-hex Digest")
    elif _heading_count(text, "Adversarial Review") == 1:
        recomputed = candidate_dossier_digest(path)
        if digests[0] != recomputed:
            errors.append(
                f"{source}: Adversarial Review Digest does not match the dossier bytes outside that review section"
            )


def _replay_implementation_evidence(
    report: str,
    dossiers: dict[str, Path],
    errors: list[str],
    *,
    allow_lean_repository: bool = False,
) -> None:
    repositories = PINNED_REPOSITORY_PATH_RE.findall(_section_body(report, "Pinned Inputs"))
    if not repositories and allow_lean_repository:
        repositories = re.findall(
            r"^- Repository:\s+`([^`\n]+)`\s*$",
            _section_body(report, "Pinned Inputs"),
            re.MULTILINE,
        )
    if len(repositories) != 1:
        return
    repository = Path(repositories[0]).expanduser().absolute()
    try:
        metadata = repository.lstat()
    except FileNotFoundError:
        # Structural fixture reports may use a non-existent repository identity.
        # Real prepared runs use an existing absolute path; the external harness
        # independently requires and replays it.
        return
    except OSError as error:
        errors.append(f"report.md: cannot stat pinned repository for replay: {error}")
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        errors.append("report.md: pinned repository is not a real directory for replay")
        return

    sources: list[tuple[str, str]] = [("report.md", report)]
    for identifier, path in sorted(dossiers.items()):
        if identifier.startswith("F-"):
            sources.append((f"findings/{path.name}", _read_utf8(path)))

    for source_name, text in sources:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- Implementation evidence:"):
                continue
            match = IMPLEMENTATION_EVIDENCE_LINE_RE.fullmatch(stripped)
            if match is None:
                continue  # The structural validator reports this record.
            relative = PurePosixPath(match.group("source").strip())
            candidate = repository.joinpath(*relative.parts)
            current = repository
            try:
                for part in relative.parts:
                    current = current / part
                    current_metadata = current.lstat()
                    if stat.S_ISLNK(current_metadata.st_mode):
                        errors.append(f"{source_name}: Implementation evidence traverses a symlink: {relative}")
                        break
                else:
                    if not stat.S_ISREG(candidate.lstat().st_mode):
                        errors.append(
                            f"{source_name}: Implementation evidence source is not a regular file: {relative}"
                        )
                        continue
                    try:
                        code_lines = candidate.read_text(encoding="utf-8").splitlines()
                    except (OSError, UnicodeError) as error:
                        errors.append(f"{source_name}: cannot read Implementation evidence source {relative}: {error}")
                        continue
                    start = int(match.group("start"))
                    end = int(match.group("end") or start)
                    if end > len(code_lines):
                        errors.append(
                            f"{source_name}: Implementation evidence lines are outside {relative}: {start}-{end}"
                        )
                        continue
                    selected = "\n".join(code_lines[start - 1 : end])
                    quote = match.group("quote")
                    if quote not in selected:
                        errors.append(
                            f"{source_name}: Implementation evidence quote does not occur "
                            f"within lines {start}-{end} of {relative}"
                        )
            except FileNotFoundError:
                errors.append(f"{source_name}: Implementation evidence source does not exist: {relative}")
            except OSError as error:
                errors.append(f"{source_name}: cannot stat Implementation evidence source {relative}: {error}")


def _lean_link_labels(section: str) -> dict[str, str]:
    """Return canonical dossier link labels keyed by their target identifier."""

    labels: dict[str, str] = {}
    for label, target in MARKDOWN_LINK_RE.findall(section):
        match = re.fullmatch(r"findings/([FS]-\d{3})\.md", target)
        if match is not None:
            labels[match.group(1)] = label.strip()
    return labels


def _lint_lean_dossier(
    identifier: str,
    path: Path,
    linked_label: str,
    errors: list[str],
) -> None:
    """Lint one published Lean v1 dossier without legacy ledger/digest fields."""

    text = _read_utf8(path)
    source = f"findings/{path.name}"
    title = _first_nonempty_line(text)
    expected_title = re.compile(rf"^# {re.escape(identifier)}:\s+\S.*$")
    if title is None or expected_title.fullmatch(title) is None:
        errors.append(f"{source}: title must begin '# {identifier}: '")
    elif linked_label != title.removeprefix("# "):
        errors.append(f"{source}: title does not exactly match its report.md link label")
    if PLACEHOLDER_RE.search(text):
        errors.append(f"{source}: contains an unreplaced template placeholder")

    if identifier.startswith("F-"):
        required = FINDING_SECTIONS
        forbidden = (
            "Conflicting Specification Anchors",
            "Applicability Overlap",
            "Precedence Search",
            "Specification Conflict Chain",
        )
    else:
        required = CONFLICT_SECTIONS
        forbidden = (
            "Root Cause",
            "Specification Evidence",
            "Implementation and Probe Evidence",
            "Contradiction Chain",
            "Counter-Search",
        )
    _require_sections(text, required, source, errors)
    _require_heading_order(text, required, source, errors)
    for heading in forbidden:
        if _heading_count(text, heading):
            errors.append(f"{source}: {identifier[0]} dossier contains other-type section '## {heading}'")

    if identifier.startswith("F-"):
        _lint_implementation_evidence(
            _section_body(text, "Implementation and Probe Evidence"),
            minimum=1,
            source_name=source,
            errors=errors,
        )

    review_span = _section_span(text, "Adversarial Review")
    review = review_span[2] if review_span is not None else ""
    outcomes = OUTCOME_RE.findall(review)
    if outcomes != ["Supported"]:
        errors.append(f"{source}: published dossier reviewer Outcome must be exactly Supported")
    review_modes = REVIEW_MODE_RE.findall(review)
    if review_modes not in [["Fresh native"], ["Serial falsification"]]:
        errors.append(f"{source}: Adversarial Review Mode must be exactly Fresh native or Serial falsification")
    if review_span is not None:
        outside_review = text[: review_span[0]] + text[review_span[1] :]
        if OUTCOME_RE.search(outside_review):
            errors.append(f"{source}: reviewer Outcome must appear only in '## Adversarial Review'")


def _lean_repository(report: str) -> Path | None:
    repositories = PINNED_REPOSITORY_PATH_RE.findall(_section_body(report, "Pinned Inputs"))
    if not repositories:
        repositories = re.findall(
            r"^- Repository:\s+`([^`\n]+)`\s*$",
            _section_body(report, "Pinned Inputs"),
            re.MULTILINE,
        )
    if len(repositories) != 1:
        return None
    repository = Path(repositories[0]).expanduser().absolute()
    return repository if repository.is_dir() else None


def _validate_lean_repo_anchor(
    value: str,
    *,
    source: str,
    field_name: str,
    repository: Path | None,
    errors: list[str],
) -> bool:
    match = re.fullmatch(r"([^:`\\][^:`\\]*):([1-9]\d*)", value)
    if match is None:
        errors.append(f"{source}: {field_name} must be a repository-relative path:line anchor")
        return False
    path_value = match.group(1)
    if _is_unsafe_repository_path(path_value):
        errors.append(f"{source}: {field_name} must use a safe repository-relative path")
        return False
    if repository is None:
        return True
    path = repository.joinpath(*PurePosixPath(path_value).parts)
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            errors.append(f"{source}: {field_name} source is not a regular file: {path_value}")
            return False
        line_count = len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError) as exc:
        errors.append(f"{source}: cannot replay {field_name} {path_value}: {exc}")
        return False
    if int(match.group(2)) > line_count:
        errors.append(f"{source}: {field_name} line is outside {path_value}")
        return False
    return True


def _validate_lean_evidence_anchor(
    value: str,
    *,
    packet_specs: set[str],
    source: str,
    field_name: str,
    repository: Path | None,
    errors: list[str],
) -> bool:
    specification = CANONICAL_SPECIFICATION_ANCHOR_RE.fullmatch(value)
    if specification is not None:
        if specification.group("source") not in packet_specs:
            errors.append(f"{source}: {field_name} specification anchor must belong to the packet")
            return False
        return True
    return _validate_lean_repo_anchor(
        value,
        source=source,
        field_name=field_name,
        repository=repository,
        errors=errors,
    )


def _lint_lean_receipts(
    report: str,
    errors: list[str],
    *,
    complete: bool,
) -> None:
    """Validate finalized Lean receipt structure and Complete-only closure gates."""

    pinned_body = _section_body(report, "Pinned Inputs")
    pinned_list = re.findall(
        r"^- Specification `(SPEC-\d{3})`:\s+\S.*$",
        pinned_body,
        re.MULTILINE,
    )
    pinned_specs = set(pinned_list)
    repository = _lean_repository(report)
    published_findings = set(re.findall(r"\(findings/(F-\d{3})\.md\)", report))
    card_findings: set[str] = set()
    global_capability_ids: set[str] = set()
    if not pinned_list:
        errors.append(
            "report.md: finalized Lean v1 Pinned Inputs must declare at least one '- Specification `SPEC-xxx`: ...'"
        )
    for duplicate in sorted(identifier for identifier in pinned_specs if pinned_list.count(identifier) > 1):
        errors.append(f"report.md: Lean v1 has duplicate pinned specification {duplicate}")

    coverage = _section_body(report, "Coverage")
    source_risk_lines = re.findall(
        r"^- Source risk lanes:\s*(\S.*)$",
        coverage,
        re.MULTILINE,
    )
    source_risk_lanes: dict[str, set[str]] = {}
    if len(source_risk_lines) != 1:
        errors.append("report.md: Coverage requires exactly one '- Source risk lanes: ...' field")
    else:
        for entry in source_risk_lines[0].split("; "):
            match = re.fullmatch(
                r"(SPEC-\d{3})=(none|[a-z-]+(?:,[a-z-]+)*)",
                entry,
            )
            if match is None:
                errors.append(
                    "report.md: Source risk lanes must use SPEC-xxx=lane[,lane] entries separated by semicolons"
                )
                continue
            spec_id = match.group(1)
            if spec_id in source_risk_lanes:
                errors.append(f"report.md: duplicate Source risk lanes entry {spec_id}")
                continue
            lane_value = match.group(2)
            lanes = set() if lane_value == "none" else set(lane_value.split(","))
            if lane_value != "none" and len(lanes) != len(lane_value.split(",")):
                errors.append(f"report.md: Source risk lanes repeats a lane for {spec_id}")
            invalid = lanes - LEAN_RISK_LANES
            if invalid:
                errors.append(
                    f"report.md: Source risk lanes for {spec_id} has invalid lanes: " + ", ".join(sorted(invalid))
                )
            source_risk_lanes[spec_id] = lanes
        declared_specs = set(source_risk_lanes)
        for missing in sorted(pinned_specs - declared_specs):
            errors.append(f"report.md: Source risk lanes is missing pinned {missing}")
        for extra in sorted(declared_specs - pinned_specs):
            errors.append(f"report.md: Source risk lanes references unpinned {extra}")
    receipts = _subsection_body(
        coverage,
        "Signature Triad Receipts",
        "report.md: Coverage",
        errors,
    )
    packets = _ledger_blocks(
        receipts,
        r"P-\d{3}",
        "Signature Triad Receipts",
        errors,
    )
    if not packets:
        errors.append("report.md: finalized Lean v1 requires at least one Signature Triad packet")
    specification_owners: dict[str, list[str]] = {}
    packet_lane_union: dict[str, set[str]] = {spec_id: set() for spec_id in pinned_specs}
    ownership_pairs: dict[tuple[frozenset[str], str], str] = {}
    for packet_id, block in packets:
        source = f"report.md: Signature Triad Receipts {packet_id}"
        specifications = _field_value(block, "Specifications", source, errors)
        mechanism = _field_value(block, "Mechanism", source, errors)
        risk_lanes = _field_value(block, "Risk lanes", source, errors)
        anchors = _field_value(block, "Grounding anchors", source, errors)
        terms = _field_value(block, "Grounded terms", source, errors)
        normative_signals = _field_value(block, "Normative signals", source, errors)
        capability_obligations = _field_value(block, "Capability obligations", source, errors)
        integration_scope = _field_value(block, "Integration scope", source, errors)
        core_scope = _field_value(block, "Core scope", source, errors)
        state = _field_value(block, "State", source, errors)

        seam_lines = [line.strip() for line in block.splitlines() if line.strip().startswith("- Seam connection:")]
        seam_pattern = re.compile(
            r"^- Seam connection: integration=`(?P<integration>[^`]+)`; "
            r"core=`(?P<core>[^`]+)`; observable=(?P<observable>[^;\n]+); "
            r"relationship=(?P<relationship>\S.*)$"
        )
        if len(seam_lines) != 1:
            errors.append(f"{source}: requires exactly one Seam connection record")
        else:
            seam_match = seam_pattern.fullmatch(seam_lines[0])
            if seam_match is None:
                errors.append(f"{source}: malformed Seam connection record")
            else:
                _validate_lean_repo_anchor(
                    seam_match.group("integration"),
                    source=source,
                    field_name="Seam integration",
                    repository=repository,
                    errors=errors,
                )
                _validate_lean_repo_anchor(
                    seam_match.group("core"),
                    source=source,
                    field_name="Seam core",
                    repository=repository,
                    errors=errors,
                )

        packet_specs: list[str] = []
        if specifications is not None:
            if (
                re.fullmatch(
                    r"`SPEC-\d{3}`(?:; `SPEC-\d{3}`)*",
                    specifications,
                )
                is None
            ):
                errors.append(
                    f"{source}: Specifications must be one or more backticked "
                    "SPEC-xxx identifiers separated by semicolons"
                )
            else:
                packet_specs = re.findall(r"`(SPEC-\d{3})`", specifications)
                if len(packet_specs) != len(set(packet_specs)):
                    errors.append(f"{source}: Specifications must not repeat an identifier")
                for spec_id in set(packet_specs):
                    if spec_id not in pinned_specs:
                        errors.append(f"{source}: Specifications references unpinned {spec_id}")
                    specification_owners.setdefault(spec_id, []).append(packet_id)

        packet_lanes: set[str] = set()
        if risk_lanes is not None:
            lane_values = risk_lanes.split(",")
            packet_lanes = set(lane_values)
            if len(packet_lanes) != len(lane_values):
                errors.append(f"{source}: Risk lanes must not repeat a lane")
            invalid = packet_lanes - LEAN_RISK_LANES
            if invalid:
                errors.append(f"{source}: Risk lanes has invalid lanes: " + ", ".join(sorted(invalid)))
            if not packet_lanes:
                errors.append(f"{source}: Risk lanes must be non-empty")
            for spec_id in set(packet_specs) & pinned_specs:
                packet_lane_union[spec_id].update(packet_lanes)

        if mechanism in NONE_MARKERS:
            errors.append(f"{source}: Mechanism must be non-empty")
        if packet_specs and mechanism not in NONE_MARKERS:
            ownership = (frozenset(packet_specs), mechanism or "")
            previous = ownership_pairs.get(ownership)
            if previous is not None:
                errors.append(f"{source}: duplicates (Specifications set, Mechanism) ownership from {previous}")
            else:
                ownership_pairs[ownership] = packet_id

        for label, value in (
            ("Grounding anchors", anchors),
            ("Grounded terms", terms),
            ("Integration scope", integration_scope),
            ("Core scope", core_scope),
        ):
            if value in NONE_MARKERS:
                errors.append(f"{source}: {label} must be non-empty")
        if normative_signals is not None:
            signal_match = re.fullmatch(r"returned=(\d+); inspected=(\d+)", normative_signals)
            if signal_match is None:
                errors.append(f"{source}: Normative signals must be 'returned=N; inspected=N'")
            elif int(signal_match.group(2)) > int(signal_match.group(1)):
                errors.append(f"{source}: Normative signals inspected cannot exceed returned")
            elif complete and signal_match.group(1) != signal_match.group(2):
                errors.append(f"{source}: Normative signals must inspect every returned result")
        if capability_obligations is not None:
            obligation_counts: tuple[int, int, int, int, int] | None = None
            obligation_match = re.fullmatch(
                r"implemented=(\d+); delegated=(\d+); gap=(\d+); "
                r"out-of-scope=(\d+); uncertain=(\d+)",
                capability_obligations,
            )
            if obligation_match is None:
                errors.append(
                    f"{source}: Capability obligations must be "
                    "'implemented=N; delegated=N; gap=N; out-of-scope=N; uncertain=N'"
                )
            else:
                obligation_counts = tuple(int(value) for value in obligation_match.groups())
                if sum(obligation_counts) < 1:
                    errors.append(f"{source}: Capability obligations total must be at least 1")
                if complete and obligation_counts[-1] != 0:
                    errors.append(f"{source}: Complete Capability obligations uncertain must be 0")
        if anchors not in NONE_MARKERS:
            anchor_specs = re.findall(
                r"`(SPEC-\d{3})/M-\d{3}:[1-9]\d*(?:-[1-9]\d*)?`",
                anchors or "",
            )
            if not anchor_specs:
                errors.append(f"{source}: Grounding anchors must contain canonical `SPEC-xxx/M-xxx:N[-M]` anchors")
            elif set(anchor_specs) != set(packet_specs):
                errors.append(f"{source}: Grounding anchors must cover exactly the Specifications in this packet")
        if integration_scope is not None and core_scope is not None and integration_scope == core_scope:
            errors.append(f"{source}: Integration scope and Core scope must be different")
        if complete:
            if state != "complete":
                errors.append(f"{source}: State must be complete")
        elif state not in {"complete", "unfinished"}:
            errors.append(f"{source}: State must be complete or unfinished")

        pass_records: dict[str, tuple[int, int, str]] = {}
        pass_lines = [line.strip() for line in block.splitlines() if line.strip().startswith("- Pass ")]
        pass_pattern = re.compile(
            r"^- Pass (?P<name>[a-z-]+/[a-z-]+): "
            r"returned=(?P<returned>\d+); "
            r"inspected=(?P<inspected>\d+); "
            r"disposition=(?P<disposition>\S.*)$"
        )
        for line in pass_lines:
            match = pass_pattern.fullmatch(line)
            if match is None:
                errors.append(f"{source}: malformed Signature Triad pass record")
                continue
            name = match.group("name")
            if name not in LEAN_SIGNATURE_PASSES:
                errors.append(f"{source}: unexpected Signature Triad pass {name}")
                continue
            if name in pass_records:
                errors.append(f"{source}: duplicate Signature Triad pass {name}")
                continue
            returned = int(match.group("returned"))
            inspected = int(match.group("inspected"))
            disposition = match.group("disposition").strip()
            pass_records[name] = (returned, inspected, disposition)
            if disposition not in {
                "candidate-bearing",
                "satisfying",
                "out-of-scope",
            }:
                errors.append(
                    f"{source}: pass {name} disposition must be candidate-bearing, satisfying, or out-of-scope"
                )
        for missing in sorted(LEAN_SIGNATURE_PASSES - set(pass_records)):
            errors.append(f"{source}: missing Signature Triad pass {missing}")

        card_counts = {name: 0 for name in LEAN_SIGNATURE_PASSES}
        priority_card_counts = {name: 0 for name in LEAN_SIGNATURE_PASSES}
        packet_lead_ids: set[str] = set()
        card_outcomes = {
            signature: {
                outcome: 0
                for outcome in (
                    "finding",
                    "contradicted",
                    "satisfying",
                    "out-of-scope",
                    "uncertain",
                )
            }
            for signature in LEAN_SIGNATURES
        }
        card_pass_outcomes = {name: set() for name in LEAN_SIGNATURE_PASSES}
        supported_candidates: set[tuple[str, str]] = set()
        closure_lines = [line.strip() for line in block.splitlines() if line.strip().startswith("- Closure card:")]
        closure_card_pattern = re.compile(
            r"^- Closure card: leads=(?P<leads>(?:`L-[0-9a-f]{12}`(?:, `L-[0-9a-f]{12}`){0,2})|(?:`D-\d{3}`(?:, `D-\d{3}`){0,2})); "
            r"signature=(?P<signature>boundary|dispatch-state|capability); "
            r"scope=(?P<scope>integration|core); "
            r"markers=(?P<markers>[a-z-]+(?:,[a-z-]+)*); "
            r"outcome=(?P<outcome>finding|contradicted|satisfying|out-of-scope|uncertain); "
            r"spec=`(?P<spec>SPEC-\d{3}/M-\d{3}:[1-9]\d*(?:-[1-9]\d*)?)`; "
            r"implementation=`(?P<implementation>[^`]+)`; "
            r"candidate=(?P<candidate>`C-\d{3}`|None); "
            r"dossier=(?P<dossier>`F-\d{3}`|None); "
            r"review=(?P<review>Supported|Contradicted|Insufficient|None); "
            r"counterevidence=(?P<counterevidence>`[^`]+`|None); "
            r"exclusion=(?P<exclusion>`[^`]+`|None); "
            r"basis=(?P<basis>[^;\n]+); witnesses=(?P<witnesses>\S.*)$"
        )
        allowed_markers = {
            "none",
            "explicit-omission",
            "finite-boundary",
            "dispatch-diversion",
            "state-timing",
        }
        for line in closure_lines:
            match = closure_card_pattern.fullmatch(line)
            if match is None:
                errors.append(f"{source}: malformed Closure card record")
                continue
            lead_ids = re.findall(r"`([LD]-(?:[0-9a-f]{12}|\d{3}))`", match.group("leads"))
            derived = lead_ids[0].startswith("D-")
            for lead_id in lead_ids:
                if lead_id in packet_lead_ids:
                    errors.append(f"{source}: duplicate packet lead ID {lead_id}")
                packet_lead_ids.add(lead_id)
            signature = match.group("signature")
            pass_name = f"{signature}/{match.group('scope')}"
            outcome = match.group("outcome")
            card_counts[pass_name] += len(lead_ids)
            if not derived:
                priority_card_counts[pass_name] += len(lead_ids)
            card_outcomes[signature][outcome] += len(lead_ids)
            card_pass_outcomes[pass_name].add(outcome)

            marker_list = match.group("markers").split(",")
            if len(marker_list) != len(set(marker_list)) or not set(marker_list) <= allowed_markers:
                errors.append(f"{source}: Closure card has invalid or duplicate markers")
            if "none" in marker_list and len(marker_list) != 1:
                errors.append(f"{source}: Closure card marker none must stand alone")
            if "explicit-omission" in marker_list and outcome == "satisfying":
                errors.append(f"{source}: explicit-omission Closure card cannot be satisfying")
            if derived and outcome not in {"finding", "contradicted", "uncertain"}:
                errors.append(
                    f"{source}: Derived lead Closure card outcome must be finding, contradicted, or uncertain"
                )

            spec_anchor = match.group("spec")
            spec_match = CANONICAL_SPECIFICATION_ANCHOR_RE.fullmatch(spec_anchor)
            if spec_match is None or spec_match.group("source") not in set(packet_specs):
                errors.append(f"{source}: Closure card spec anchor must belong to the packet")
            _validate_lean_repo_anchor(
                match.group("implementation"),
                source=source,
                field_name="Closure card implementation",
                repository=repository,
                errors=errors,
            )

            candidate = match.group("candidate").strip("`")
            dossier = match.group("dossier").strip("`")
            review = match.group("review")
            counter = match.group("counterevidence")
            exclusion = match.group("exclusion")
            witnesses = match.group("witnesses")
            if outcome == "finding":
                if candidate == "None" or dossier == "None" or review != "Supported":
                    errors.append(f"{source}: finding Closure card requires candidate, F dossier, and Supported review")
                else:
                    supported_candidates.add((candidate, dossier))
                    card_findings.add(dossier)
                    if dossier not in published_findings:
                        errors.append(f"{source}: finding Closure card references unpublished {dossier}")
            elif outcome == "contradicted":
                if candidate == "None" or dossier != "None" or review != "Contradicted" or counter == "None":
                    errors.append(
                        f"{source}: contradicted Closure card requires candidate, Contradicted review, counterevidence, and no dossier"
                    )
            elif outcome == "uncertain":
                if candidate == "None" or dossier != "None" or review != "Insufficient":
                    errors.append(
                        f"{source}: uncertain Closure card requires candidate, Insufficient review, and no dossier"
                    )
                if complete:
                    errors.append(f"{source}: Complete Closure card cannot be uncertain")
            elif candidate != "None" or dossier != "None" or review != "None":
                errors.append(f"{source}: {outcome} Closure card cannot publish candidate, dossier, or review")

            if counter != "None":
                _validate_lean_evidence_anchor(
                    counter.strip("`"),
                    packet_specs=set(packet_specs),
                    source=source,
                    field_name="Closure card counterevidence",
                    repository=repository,
                    errors=errors,
                )
            if outcome == "out-of-scope":
                if exclusion == "None":
                    errors.append(f"{source}: out-of-scope Closure card requires an exclusion anchor")
                else:
                    _validate_lean_evidence_anchor(
                        exclusion.strip("`"),
                        packet_specs=set(packet_specs),
                        source=source,
                        field_name="Closure card exclusion",
                        repository=repository,
                        errors=errors,
                    )
            elif exclusion != "None":
                errors.append(f"{source}: only out-of-scope Closure cards may use exclusion")

            if outcome == "satisfying" and "finite-boundary" in marker_list:
                if re.fullmatch(r"below=\S[^|]*\|at=\S[^|]*\|above=\S[^|]*\|narrowing=\S.*", witnesses) is None:
                    errors.append(
                        f"{source}: finite-boundary satisfying card requires below/at/above/narrowing witnesses"
                    )
            if outcome == "satisfying" and "dispatch-diversion" in marker_list:
                if re.fullmatch(r"variants=\S[^|]*\|destinations=\S.*", witnesses) is None:
                    errors.append(
                        f"{source}: dispatch-diversion satisfying card requires variants/destinations witnesses"
                    )
            if outcome == "satisfying" and "state-timing" in marker_list:
                if re.fullmatch(r"timeline=\S.*", witnesses) is None:
                    errors.append(f"{source}: state-timing satisfying card requires a timeline witness")

        for pass_name, (returned, inspected, disposition) in pass_records.items():
            if card_counts[pass_name] != inspected:
                errors.append(
                    f"{source}: pass {pass_name} inspected must equal Closure card lead count {card_counts[pass_name]}"
                )
            priority_count = priority_card_counts[pass_name]
            if priority_count > returned:
                errors.append(f"{source}: pass {pass_name} has more priority L cards than returned")
            elif complete and priority_count != returned:
                errors.append(f"{source}: Complete pass {pass_name} priority L-card count must equal returned")
            outcomes = card_pass_outcomes[pass_name]
            candidate_outcomes = {"finding", "contradicted", "uncertain"}
            if disposition == "candidate-bearing" and not (outcomes & candidate_outcomes):
                errors.append(f"{source}: candidate-bearing pass {pass_name} requires a candidate Closure card")
            if disposition == "satisfying" and (
                outcomes - {"satisfying", "out-of-scope"} or "satisfying" not in outcomes
            ):
                errors.append(
                    f"{source}: satisfying pass {pass_name} requires at least one "
                    "satisfying card and no candidate cards"
                )
            if disposition == "out-of-scope" and outcomes - {"out-of-scope"}:
                errors.append(f"{source}: out-of-scope pass {pass_name} has other Closure cards")

        empty_signature_lines = [
            line.strip() for line in block.splitlines() if line.strip().startswith("- Empty signature ")
        ]
        empty_signature_pattern = re.compile(
            r"^- Empty signature (?P<signature>boundary|dispatch-state|capability): "
            r"basis=(?P<basis>\S.*)$"
        )
        empty_signatures: set[str] = set()
        for line in empty_signature_lines:
            match = empty_signature_pattern.fullmatch(line)
            if match is None:
                errors.append(f"{source}: malformed Empty signature record")
                continue
            signature = match.group("signature")
            if signature in empty_signatures:
                errors.append(f"{source}: duplicate Empty signature {signature}")
            empty_signatures.add(signature)
        for signature in LEAN_SIGNATURES:
            returned_total = sum(
                pass_records.get(f"{signature}/{scope}", (0, 0, ""))[0] for scope in ("integration", "core")
            )
            card_total = sum(card_outcomes[signature].values())
            derived_total = sum(
                card_counts[f"{signature}/{scope}"] - priority_card_counts[f"{signature}/{scope}"]
                for scope in ("integration", "core")
            )
            if returned_total == 0:
                if card_total == 0 and signature not in empty_signatures:
                    errors.append(
                        f"{source}: zero-return signature {signature} with no Derived "
                        "cards requires exactly one Empty signature explanation"
                    )
                elif card_total != derived_total:
                    errors.append(f"{source}: zero-return signature {signature} may contain only Derived lead cards")
                elif card_total and signature in empty_signatures:
                    errors.append(
                        f"{source}: zero-return signature {signature} with Derived cards cannot use Empty signature"
                    )
            else:
                if card_total == 0:
                    errors.append(f"{source}: signature {signature} requires at least one Closure card")
                if signature in empty_signatures:
                    errors.append(f"{source}: non-empty signature {signature} cannot use Empty signature")

        closure_records: dict[str, tuple[int, int, int, int, int]] = {}
        closure_lines = [line.strip() for line in block.splitlines() if line.strip().startswith("- Lead closure ")]
        closure_pattern = re.compile(
            r"^- Lead closure (?P<signature>[a-z-]+): "
            r"finding=(?P<finding>\d+); "
            r"contradicted=(?P<contradicted>\d+); "
            r"satisfying=(?P<satisfying>\d+); "
            r"out-of-scope=(?P<out_of_scope>\d+); "
            r"uncertain=(?P<uncertain>\d+)$"
        )
        for line in closure_lines:
            match = closure_pattern.fullmatch(line)
            if match is None:
                errors.append(f"{source}: malformed Lead closure record")
                continue
            signature = match.group("signature")
            if signature not in LEAN_SIGNATURES:
                errors.append(f"{source}: unexpected Lead closure {signature}")
                continue
            if signature in closure_records:
                errors.append(f"{source}: duplicate Lead closure {signature}")
                continue
            counts = tuple(
                int(match.group(field))
                for field in (
                    "finding",
                    "contradicted",
                    "satisfying",
                    "out_of_scope",
                    "uncertain",
                )
            )
            closure_records[signature] = counts
            expected_counts = card_outcomes[signature]
            derived = tuple(
                expected_counts[name]
                for name in (
                    "finding",
                    "contradicted",
                    "satisfying",
                    "out-of-scope",
                    "uncertain",
                )
            )
            if counts != derived:
                errors.append(f"{source}: Lead closure {signature} must equal Closure card outcomes")
            integration = pass_records.get(f"{signature}/integration")
            core = pass_records.get(f"{signature}/core")
            if integration is not None and core is not None:
                inspected = integration[1] + core[1]
                if sum(counts) != inspected:
                    errors.append(
                        f"{source}: Lead closure {signature} total must equal "
                        f"integration/core inspected total {inspected}"
                    )
            if complete and counts[-1] != 0:
                errors.append(f"{source}: Complete Lead closure {signature} uncertain must be 0")
        for missing in LEAN_SIGNATURES:
            if missing not in closure_records:
                errors.append(f"{source}: missing Lead closure {missing}")

        capability_lines = [
            line.strip() for line in block.splitlines() if line.strip().startswith("- Capability card:")
        ]
        capability_pattern = re.compile(
            r"^- Capability card: id=`(?P<id>K-\d{3})`; "
            r"status=(?P<status>implemented|delegated|gap|out-of-scope|uncertain); "
            r"spec=`(?P<spec>SPEC-\d{3}/M-\d{3}:[1-9]\d*(?:-[1-9]\d*)?)`; "
            r"repository=(?P<repository>`[^`]+`|None); "
            r"candidate=(?P<candidate>`C-\d{3}`|None); "
            r"dossier=(?P<dossier>`F-\d{3}`|None); "
            r"review=(?P<review>Supported|Insufficient|None); "
            r"exclusion=(?P<exclusion>`[^`]+`|None); basis=(?P<basis>\S.*)$"
        )
        capability_counts = {name: 0 for name in ("implemented", "delegated", "gap", "out-of-scope", "uncertain")}
        for line in capability_lines:
            match = capability_pattern.fullmatch(line)
            if match is None:
                errors.append(f"{source}: malformed Capability card record")
                continue
            capability_id = match.group("id")
            if capability_id in global_capability_ids:
                errors.append(f"{source}: duplicate global capability ID {capability_id}")
            global_capability_ids.add(capability_id)
            status = match.group("status")
            capability_counts[status] += 1
            spec_match = CANONICAL_SPECIFICATION_ANCHOR_RE.fullmatch(match.group("spec"))
            if spec_match is None or spec_match.group("source") not in set(packet_specs):
                errors.append(f"{source}: Capability card spec anchor must belong to the packet")
            repo_evidence = match.group("repository")
            candidate = match.group("candidate").strip("`")
            dossier = match.group("dossier").strip("`")
            review = match.group("review")
            exclusion = match.group("exclusion")
            if status in {"implemented", "delegated"}:
                if repo_evidence == "None":
                    errors.append(f"{source}: {status} Capability card requires repository evidence")
                else:
                    _validate_lean_repo_anchor(
                        repo_evidence.strip("`"),
                        source=source,
                        field_name="Capability card repository",
                        repository=repository,
                        errors=errors,
                    )
            if status == "gap":
                if candidate == "None" or dossier == "None" or review != "Supported":
                    errors.append(f"{source}: gap Capability card requires Supported candidate and F dossier")
                elif (candidate, dossier) not in supported_candidates:
                    errors.append(f"{source}: gap Capability card must reference a finding Closure card")
            elif status == "out-of-scope":
                if exclusion == "None":
                    errors.append(f"{source}: out-of-scope Capability card requires exclusion")
                else:
                    _validate_lean_evidence_anchor(
                        exclusion.strip("`"),
                        packet_specs=set(packet_specs),
                        source=source,
                        field_name="Capability card exclusion",
                        repository=repository,
                        errors=errors,
                    )
            elif status == "uncertain":
                if candidate == "None" or dossier != "None" or review != "Insufficient":
                    errors.append(f"{source}: uncertain Capability card requires Insufficient candidate")
                if complete:
                    errors.append(f"{source}: Complete Capability card cannot be uncertain")
            if status not in {"gap", "uncertain"} and (candidate != "None" or dossier != "None" or review != "None"):
                errors.append(f"{source}: {status} Capability card cannot publish candidate, dossier, or review")
            if status != "out-of-scope" and exclusion != "None":
                errors.append(f"{source}: only out-of-scope Capability cards may use exclusion")

        if capability_obligations is not None and obligation_counts is not None:
            derived_obligations = tuple(
                capability_counts[name] for name in ("implemented", "delegated", "gap", "out-of-scope", "uncertain")
            )
            if obligation_counts != derived_obligations:
                errors.append(f"{source}: Capability obligations must equal Capability card statuses")

        challenge_lines = [
            line.strip() for line in block.splitlines() if line.strip().startswith("- Zero-finding challenge:")
        ]
        challenge_pattern = re.compile(
            r"^- Zero-finding challenge: evidence=`(?P<evidence>[^`]+)`; basis=(?P<basis>\S.*)$"
        )
        if (
            card_outcomes["boundary"]["finding"]
            + card_outcomes["dispatch-state"]["finding"]
            + card_outcomes["capability"]["finding"]
            == 0
            and len(challenge_lines) != 1
        ):
            errors.append(f"{source}: zero-finding packet requires exactly one Zero-finding challenge")
        for line in challenge_lines:
            match = challenge_pattern.fullmatch(line)
            if match is None:
                errors.append(f"{source}: malformed Zero-finding challenge record")
            else:
                _validate_lean_evidence_anchor(
                    match.group("evidence"),
                    packet_specs=set(packet_specs),
                    source=source,
                    field_name="Zero-finding challenge evidence",
                    repository=repository,
                    errors=errors,
                )

    for spec_id in sorted(pinned_specs):
        owners = specification_owners.get(spec_id, [])
        if not owners:
            errors.append(
                f"report.md: pinned specification {spec_id} must be covered by at least one Signature Triad packet"
            )
        missing_lanes = source_risk_lanes.get(spec_id, set()) - packet_lane_union.get(spec_id, set())
        if missing_lanes:
            errors.append(
                f"report.md: packets for {spec_id} do not cover Source risk lanes: " + ", ".join(sorted(missing_lanes))
            )
    for dossier in sorted(published_findings - card_findings):
        errors.append(f"report.md: published finding {dossier} has no finding Closure card")

    if complete:
        validation = _section_body(report, "Validation")
        for line in (
            "- Result: Passed",
            "- Inventory verification: Passed",
            "- Report lint: Passed",
        ):
            if validation.splitlines().count(line) != 1:
                errors.append(f"report.md: Lean v1 Complete Validation requires exactly one {line!r}")


def _lint_lean_report(
    report: str,
    findings_dir: Path,
    errors: list[str],
) -> list[str]:
    """Validate the deliberately small, marker-selected Lean v1 contract."""

    if report.splitlines().count(LEAN_V1_MARKER) != 1:
        errors.append("report.md: Lean v1 report requires exactly one '- Contract: Lean v1'")
    _require_sections(report, REPORT_SECTIONS, "report.md", errors)
    _require_report_order(report, errors)
    _require_status(report, errors)
    statuses = re.findall(
        r"^-\s*Result:\s*(Complete|Partial)\s*$",
        report,
        re.MULTILINE,
    )
    validation_lines = _section_body(report, "Validation").splitlines()
    if statuses == ["Complete"]:
        _lint_lean_receipts(report, errors, complete=True)
        limitations = _section_body(report, "Limitations")
        if re.search(
            r"(?i)(?:\binitial\s+partial\b|\bthis\s+(?:initial\s+)?partial\s+report\b|"
            r"\b(?:report|audit|procedure)\s+(?:is|remains)\s+(?:partial|unfinished|not completed)\b)",
            limitations,
        ):
            errors.append("report.md: Complete status contradicts the declared Limitations state")
    elif statuses == ["Partial"] and validation_lines.count("- Report lint: Passed") == 1:
        _lint_lean_receipts(report, errors, complete=False)

    finding_count, finding_section = _parse_count_section(report, "Findings", errors)
    conflict_count, conflict_section = _parse_count_section(report, "Specification Conflicts", errors)
    finding_links = _parse_dossier_links(finding_section, "F", "Findings", errors)
    conflict_links = _parse_dossier_links(conflict_section, "S", "Specification Conflicts", errors)
    if finding_count is not None and finding_count != len(finding_links):
        errors.append(
            f"report.md: Findings count is {finding_count}, but {len(finding_links)} dossier links were found"
        )
    if conflict_count is not None and conflict_count != len(conflict_links):
        errors.append(
            f"report.md: Specification Conflicts count is {conflict_count}, but "
            f"{len(conflict_links)} dossier links were found"
        )

    dossiers = _inventory_dossiers(findings_dir, errors)
    linked = set(finding_links) | set(conflict_links)
    present = set(dossiers)
    for missing in sorted(linked - present):
        errors.append(f"report.md: linked dossier findings/{missing}.md is missing")
    for orphan in sorted(present - linked):
        errors.append(f"findings/{orphan}.md: dossier is not linked from report.md")

    labels = {
        **_lean_link_labels(finding_section),
        **_lean_link_labels(conflict_section),
    }
    for identifier in sorted(linked & present):
        _lint_lean_dossier(
            identifier,
            dossiers[identifier],
            labels.get(identifier, ""),
            errors,
        )
    _replay_implementation_evidence(
        report,
        dossiers,
        errors,
        allow_lean_repository=True,
    )
    return errors


def lint_report(report_dir: Path) -> list[str]:
    """Return structural contract errors found under ``report_dir``."""

    if not report_dir.exists():
        raise ToolError(f"report directory does not exist: {report_dir}")
    if not report_dir.is_dir():
        raise ToolError(f"report path is not a directory: {report_dir}")

    report_path = report_dir / "report.md"
    findings_dir = report_dir / "findings"
    errors: list[str] = []
    try:
        root_entries = sorted(report_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ToolError(f"cannot list report directory {report_dir}: {exc}") from exc
    for entry in root_entries:
        if entry.name not in {"report.md", "findings"}:
            errors.append(f"report directory: unexpected formal artifact {entry.name!r}")
    if not report_path.is_file():
        errors.append("report.md: required report file is missing")
    if not findings_dir.is_dir():
        errors.append("findings/: required dossier directory is missing")
    if errors and (not report_path.is_file() or not findings_dir.is_dir()):
        return errors

    report = _read_utf8(report_path)
    if _first_nonempty_line(report) != REPORT_TITLE:
        errors.append(f"report.md: first heading must be '{REPORT_TITLE}'")
    if PLACEHOLDER_RE.search(report):
        errors.append("report.md: contains an unreplaced template placeholder")
    if LEAN_V1_MARKER in report.splitlines():
        return _lint_lean_report(report, findings_dir, errors)
    _require_sections(report, REPORT_SECTIONS, "report.md", errors)
    _require_report_order(report, errors)
    _require_status(report, errors)
    _lint_probe_contract(report, errors)
    ledger = _lint_coverage_contract(report, errors)
    pinned_spec_ids = {item[0] for item in PINNED_SPEC_RE.findall(_section_body(report, "Pinned Inputs"))}

    finding_count, finding_section = _parse_count_section(
        report,
        "Findings",
        errors,
    )
    conflict_count, conflict_section = _parse_count_section(
        report,
        "Specification Conflicts",
        errors,
    )
    finding_links = _parse_dossier_links(
        finding_section,
        "F",
        "Findings",
        errors,
    )
    conflict_links = _parse_dossier_links(
        conflict_section,
        "S",
        "Specification Conflicts",
        errors,
    )

    ledger_findings = set(ledger.finding_requirements)
    linked_findings = set(finding_links)
    if ledger_findings != linked_findings:
        errors.append("report.md: Finding links do not match inconsistent Requirement dispositions")
    ledger_conflicts = set(ledger.conflict_requirements)
    linked_conflicts = set(conflict_links)
    if ledger_conflicts != linked_conflicts:
        errors.append("report.md: Specification Conflict links do not match terminal Requirement dispositions")

    if finding_count is not None and finding_count != len(finding_links):
        errors.append(
            f"report.md: Findings count is {finding_count}, but {len(finding_links)} dossier links were found"
        )
    if conflict_count is not None and conflict_count != len(conflict_links):
        errors.append(
            f"report.md: Specification Conflicts count is {conflict_count}, "
            f"but {len(conflict_links)} dossier links were found"
        )

    dossiers = _inventory_dossiers(findings_dir, errors)
    linked = set(finding_links) | set(conflict_links)
    present = set(dossiers)
    for missing in sorted(linked - present):
        errors.append(f"report.md: linked dossier findings/{missing}.md is missing")
    for orphan in sorted(present - linked):
        errors.append(f"findings/{orphan}.md: dossier is not linked from report.md")

    for identifier, path in sorted(dossiers.items()):
        expected_requirements = (
            ledger.finding_requirements.get(identifier, set())
            if identifier.startswith("F-")
            else ledger.conflict_requirements.get(identifier, set())
        )
        _lint_dossier(
            identifier,
            path,
            pinned_spec_ids,
            ledger.pinned_members,
            expected_requirements,
            errors,
        )
    _replay_implementation_evidence(report, dossiers, errors)
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Spec-Audit Markdown report directory, compute one "
            "reviewer-bound dossier digest, or emit a content-bound serial-review "
            "marker."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--report",
        type=Path,
        help="directory containing report.md and findings/",
    )
    mode.add_argument(
        "--candidate-digest",
        type=Path,
        metavar="DOSSIER",
        help=("validate one Pending F/S draft, then print its reviewer-bound SHA-256"),
    )
    mode.add_argument(
        "--candidate-review-start",
        type=Path,
        metavar="DOSSIER",
        help="validate one Pending F/S draft and emit its bound review-start marker",
    )
    mode.add_argument(
        "--candidate-review-complete",
        type=Path,
        metavar="DOSSIER",
        help=("validate one Pending F/S draft and emit its bound review-completion marker"),
    )
    mode.add_argument(
        "--candidate-finalize-supported",
        type=Path,
        metavar="DOSSIER",
        help=("atomically replace only Pending Outcome and Digest review fields after a Supported completion"),
    )
    parser.add_argument(
        "--outcome",
        choices=TERMINAL_REVIEW_OUTCOMES,
        help="terminal outcome required only with --candidate-review-complete",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.candidate_review_complete is not None and args.outcome is None:
        parser.error("--candidate-review-complete requires --outcome")
    if args.outcome is not None and args.candidate_review_complete is None:
        parser.error("--outcome requires --candidate-review-complete")
    try:
        candidate_path = (
            args.candidate_digest
            or args.candidate_review_start
            or args.candidate_review_complete
            or args.candidate_finalize_supported
        )
        if candidate_path is not None:
            errors, digest = lint_candidate_dossier(candidate_path)
            if errors:
                for error in errors:
                    print(f"contract error: {error}", file=sys.stderr)
                return 1
            assert digest is not None
            if args.candidate_digest is not None:
                print(digest)
            elif args.candidate_review_start is not None:
                _write_review_receipt(
                    candidate_path,
                    digest=digest,
                    phase="started",
                )
                print(f"SERIAL_REVIEW_START {candidate_path.stem} {digest}")
            elif args.candidate_review_complete is not None:
                assert args.outcome is not None
                _require_review_receipt(
                    candidate_path,
                    digest=digest,
                    phase="started",
                )
                _write_review_receipt(
                    candidate_path,
                    digest=digest,
                    phase="completed",
                    outcome=args.outcome,
                )
                print(f"SERIAL_REVIEW_COMPLETE {candidate_path.stem} {digest} {args.outcome}")
            else:
                _require_review_receipt(
                    candidate_path,
                    digest=digest,
                    phase="completed",
                    outcome="Supported",
                )
                finalize_candidate_supported(candidate_path, digest)
                try:
                    _review_receipt_path(candidate_path).unlink()
                except OSError as exc:
                    raise ToolError(f"cannot consume candidate review receipt after finalization: {exc}") from exc
                print(f"FINALIZED_CANDIDATE {candidate_path.stem} {digest}")
            return 0
        errors = lint_report(args.report)
    except ContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 1
    except ToolError as exc:
        print(f"tool error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Defensive CLI boundary: never mislabel as valid.
        print(f"tool error: unexpected linter failure: {exc}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"contract error: {error}", file=sys.stderr)
        return 1

    print(f"valid Spec-Audit report: {args.report / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
