#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# ruff: noqa: F402
# pylint: disable=W0621

"""GraphCommand envelope and PatchTask composite for the LKB Plan Graph.

Spec: §5.10 (revision / snapshot / idempotency) and §6.1 (TaskV2 mapping).

This module is pure — no imports of ToolContext, Task-v2, or runtime state.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

from lkb.error_codes import LkbErrorCode
from lkb.graph_types import RevisionVector
from lkb.refs import NodeRef

# ── GraphCommand ─────────────────────────────────────────────────────────────

CommandDecision = Literal["committed", "denied"]


class FrozenDict(dict[str, Any]):
    """JSON-serializable immutable mapping used by command envelopes."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("command input is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, _memo: dict[int, Any]) -> "FrozenDict":
        return self


def _freeze_json(value: Any, *, path: str = "$") -> Any:
    """Validate and freeze a value using the strict JSON data model."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, list):
        return tuple(_freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, dict):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string JSON object key {key!r}")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return FrozenDict(frozen)
    raise TypeError(f"{path} contains non-JSON value of type {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class GraphCommand:
    """Write-command envelope (spec §5.10).

    Every mutation to a board goes through a GraphCommand.  The envelope
    carries everything needed for idempotency (``command_id`` +
    ``request_hash``), optimistic concurrency (``expected_*_revision``),
    audit (``actor`` + ``reason``), and routing (``kind``).
    """

    command_id: str
    board_id: str
    actor: str
    kind: str
    request_hash: str = ""
    primary_subject_ref: NodeRef | None = None
    subject_refs: tuple[NodeRef, ...] = ()
    expected_revision_vector: RevisionVector | None = None
    expected_node_revision: int | None = None
    expected_store_revision: int | None = None
    reason: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    # Trusted roles asserted for the actor by the host (e.g. "admin",
    # "operator").  These are NOT model-supplied free text - the host
    # derives them from the authenticated session/agent context. Override
    # authorization (force_override_roles) compares these roles, never the
    # bare actor id, so an agent named "admin" gains no implicit privilege.
    roles: tuple[str, ...] = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.primary_subject_ref is not None and not isinstance(self.primary_subject_ref, NodeRef):
            raise TypeError("primary_subject_ref must be a NodeRef")
        if any(not isinstance(ref, NodeRef) for ref in self.subject_refs):
            raise TypeError("subject_refs must contain only NodeRef values")
        subjects = tuple(sorted(set(self.subject_refs), key=NodeRef.to_str))
        object.__setattr__(self, "subject_refs", subjects)
        object.__setattr__(self, "payload", _freeze_json(self.payload, path="$.payload"))
        if isinstance(self.roles, str) or not isinstance(self.roles, (list, tuple)):
            raise TypeError("roles must be a tuple or list of role names")
        if any(not isinstance(role, str) for role in self.roles):
            raise TypeError("roles must contain only strings")
        roles = tuple(dict.fromkeys(role.strip() for role in self.roles if role.strip()))
        object.__setattr__(self, "roles", roles)
        if self.expected_revision_vector is not None:
            frozen_revisions = _freeze_json(
                self.expected_revision_vector.revisions,
                path="$.expected_revision_vector",
            )
            if any(
                isinstance(revision, bool) or not isinstance(revision, int) or revision < 0
                for revision in frozen_revisions.values()
            ):
                raise ValueError("expected_revision_vector values must be non-negative integers")
            object.__setattr__(
                self,
                "expected_revision_vector",
                RevisionVector(revisions=frozen_revisions),
            )
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())
        if not isinstance(self.request_hash, str):
            raise TypeError("request_hash must be a string")
        if not self.request_hash:
            object.__setattr__(self, "request_hash", compute_request_hash(self))


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of executing a :class:`GraphCommand`.

    Uses command-centric fields shared by the adapter and repository.
    """

    decision: CommandDecision
    command_id: str
    revision_vector: RevisionVector | None = None
    validation_run_id: str | None = None
    error_code: LkbErrorCode | None = None
    reason: str | None = None
    derived_facts: tuple[str, ...] = ()
    # ClaimTask surfaces the created Claim id. DeleteTask/cascade
    # surfaces task refs whose dependency relationships changed.
    claim_id: str | None = None
    affected_refs: tuple[str, ...] = ()

    @property
    def committed(self) -> bool:
        return self.decision == "committed"


def compute_request_hash(command: GraphCommand) -> str:
    """Compute the canonical request hash for a command.

    Per spec §5.10: the hash must cover the *normalized* command, board ID,
    expected revisions, and trusted actor — but NOT ``command_id`` (which
    is the idempotency key chosen by the caller) and NOT ``created_at``.

    The hash is a SHA-256 hex digest of a canonical JSON string produced
    from a stable field order.
    """
    canonical = _build_canonical_command_dict(command)
    raw = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_request_hash(command: GraphCommand) -> bool:
    """Recompute and constant-time validate a command at execution time."""
    if not isinstance(command.request_hash, str):
        return False
    try:
        return secrets.compare_digest(command.request_hash, compute_request_hash(command))
    except (TypeError, ValueError):
        return False


def _build_canonical_command_dict(command: GraphCommand) -> dict[str, Any]:
    """Build the canonical dict used for request_hash computation.

    Excludes ``command_id``, ``request_hash`` itself, and ``created_at``.
    Normalizes NodeRefs to their string form.
    """
    primary: str | None = str(command.primary_subject_ref) if command.primary_subject_ref is not None else None
    subjects = sorted({str(ref) for ref in command.subject_refs})
    rev_vec: dict[str, int] | None = (
        command.expected_revision_vector.to_dict() if command.expected_revision_vector is not None else None
    )
    return {
        "board_id": command.board_id,
        "actor": command.actor,
        "kind": command.kind,
        "primary_subject_ref": primary,
        "subject_refs": subjects,
        "expected_revision_vector": rev_vec,
        "expected_node_revision": command.expected_node_revision,
        "expected_store_revision": command.expected_store_revision,
        "reason": command.reason,
        "payload": command.payload,
        "roles": list(command.roles),
    }


# ── PatchTask ────────────────────────────────────────────────────────────────


DependencyOperation = Literal["add", "remove"]
DependencyField = Literal["blocks", "blocked_by"]


@dataclass(frozen=True, slots=True)
class DependencyIntent:
    """One dependency mutation in canonical edge orientation.

    ``dependent`` always depends on ``prerequisite``. ``source_field`` keeps
    the Task-v2 spelling that produced the intent for lossless projection.
    """

    operation: DependencyOperation
    dependent: NodeRef
    prerequisite: NodeRef
    source_field: DependencyField

    def other_endpoint(self, task_ref: NodeRef) -> NodeRef:
        if task_ref == self.dependent:
            return self.prerequisite
        if task_ref == self.prerequisite:
            return self.dependent
        raise ValueError(f"{task_ref} is not an endpoint of this dependency intent")


@dataclass(slots=True, init=False)
class PatchTask:
    """Composite command: one TaskUpdate decomposed into sub-intents.

    Captures every sub-intent present in a mixed TaskUpdate payload so
    validation and mutation can commit atomically.
    """

    task_ref: NodeRef
    field_updates: dict[str, Any] = field(default_factory=dict)
    status_target: str | None = None
    owner_target: str | None = None
    dependency_intents: tuple["DependencyIntent", ...] = ()
    metadata_updates: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        task_ref: NodeRef,
        field_updates: dict[str, Any] | None = None,
        status_target: str | None = None,
        owner_target: str | None = None,
        dependency_intents: Iterable[DependencyIntent] = (),
        metadata_updates: dict[str, Any] | None = None,
        *,
        add_dependencies: Iterable[NodeRef] = (),
        remove_dependencies: Iterable[NodeRef] = (),
    ) -> None:
        """Create a patch, accepting the legacy endpoint-only arguments."""
        intents = list(dependency_intents)
        intents.extend(DependencyIntent("add", task_ref, ref, "blocked_by") for ref in add_dependencies)
        intents.extend(DependencyIntent("remove", task_ref, ref, "blocked_by") for ref in remove_dependencies)
        self.task_ref = task_ref
        self.field_updates = dict(field_updates or {})
        self.status_target = status_target
        self.owner_target = owner_target
        self.dependency_intents = tuple(intents)
        self.metadata_updates = dict(metadata_updates or {})

    # -- intent detection --

    @property
    def add_dependencies(self) -> tuple[NodeRef, ...]:
        """Legacy endpoint-only projection of dependency additions."""
        return tuple(
            intent.other_endpoint(self.task_ref) for intent in self.dependency_intents if intent.operation == "add"
        )

    @property
    def remove_dependencies(self) -> tuple[NodeRef, ...]:
        """Legacy endpoint-only projection of dependency removals."""
        return tuple(
            intent.other_endpoint(self.task_ref) for intent in self.dependency_intents if intent.operation == "remove"
        )

    @property
    def has_field_updates(self) -> bool:
        return bool(self.field_updates)

    @property
    def has_status_change(self) -> bool:
        return self.status_target is not None

    @property
    def has_owner_change(self) -> bool:
        return self.owner_target is not None

    @property
    def has_add_dependencies(self) -> bool:
        return len(self.add_dependencies) > 0

    @property
    def has_remove_dependencies(self) -> bool:
        return len(self.remove_dependencies) > 0

    @property
    def has_metadata_updates(self) -> bool:
        return bool(self.metadata_updates)

    @property
    def is_empty(self) -> bool:
        return not any(
            [
                self.has_field_updates,
                self.has_status_change,
                self.has_owner_change,
                self.has_add_dependencies,
                self.has_remove_dependencies,
                self.has_metadata_updates,
            ]
        )

    @property
    def sub_intent_count(self) -> int:
        """Number of distinct sub-intent categories present."""
        return sum(
            [
                self.has_field_updates,
                self.has_status_change,
                self.has_owner_change,
                self.has_add_dependencies,
                self.has_remove_dependencies,
                self.has_metadata_updates,
            ]
        )

    # -- decomposition from a TaskUpdate payload --

    @classmethod
    def decompose(
        cls,
        tool_input: dict[str, Any],
        task_ref: NodeRef,
    ) -> "PatchTask":
        """Decompose a TaskUpdate ``tool_input`` dict into a PatchTask.

        All sub-intents present in the payload are captured and applied atomically.

        Parameters
        ----------
        tool_input:
            The raw TaskUpdate tool input dict.  Expected keys include
            ``taskId`` (ignored — use ``task_ref``), ``subject``,
            ``description``, ``activeForm``, ``status``, ``owner``,
            ``addBlocks``, ``removeBlocks``, ``addBlockedBy``,
            ``removeBlockedBy``, ``metadata``.  Unknown keys are ignored.
        task_ref:
            The :class:`NodeRef` of the task being patched.
        """
        # Field updates (subject / description / activeForm)
        field_updates: dict[str, Any] = {}
        for field in ("subject", "description", "activeForm"):
            if field in tool_input:
                field_updates[field] = tool_input[field]

        # Status change
        status_target: str | None = None
        status_val = tool_input.get("status")
        if isinstance(status_val, str) and status_val:
            status_target = status_val

        # Owner change
        owner_target: str | None = None
        owner_val = tool_input.get("owner")
        if isinstance(owner_val, str):
            owner_target = owner_val
        elif owner_val is not None and owner_val is not ...:
            # Non-string truthy owner — coerce for forward compatibility.
            owner_target = str(owner_val)

        # Preserve Task-v2 field direction while normalizing every relation to
        # canonical ``dependent depends_on prerequisite`` endpoints.
        dependency_intents: list[DependencyIntent] = []
        dependency_intents.extend(_dependency_intents(tool_input.get("addBlocks"), task_ref, "add", "blocks"))
        dependency_intents.extend(_dependency_intents(tool_input.get("addBlockedBy"), task_ref, "add", "blocked_by"))
        dependency_intents.extend(_dependency_intents(tool_input.get("removeBlocks"), task_ref, "remove", "blocks"))
        dependency_intents.extend(
            _dependency_intents(tool_input.get("removeBlockedBy"), task_ref, "remove", "blocked_by")
        )

        # Metadata updates
        metadata_updates: dict[str, Any] = {}
        meta = tool_input.get("metadata")
        if isinstance(meta, dict) and meta:
            metadata_updates = dict(meta)

        return cls(
            task_ref=task_ref,
            field_updates=field_updates,
            status_target=status_target,
            owner_target=owner_target,
            dependency_intents=tuple(dependency_intents),
            metadata_updates=metadata_updates,
        )


def _coerce_dep_refs(
    raw: Any,
    from_ref: NodeRef,
    direction: Literal["blocks", "blocked_by"],
) -> list[NodeRef]:
    """Coerce a raw dependency value (string or list of strings) into
    a list of :class:`NodeRef`.

    ``direction`` indicates the edge direction from ``from_ref``'s
    perspective — ``blocks`` means the other node is downstream
    (from_ref blocks other), ``blocked_by`` means the other node is
    upstream (from_ref depends on other).  The returned list always
    contains the *other* endpoint(s); direction is preserved by the
    caller.
    """
    if raw is None:
        return []
    ids: list[str]
    if isinstance(raw, str):
        ids = [raw] if raw else []
    elif isinstance(raw, (list, tuple)):
        ids = [str(x) for x in raw if str(x)]
    else:
        return []

    refs: list[NodeRef] = []
    for tid in ids:
        # Construct a peer NodeRef in the same graph and kind as from_ref.
        # Phase 2/3 will validate these actually exist; Phase 1 just
        # preserves the identity in normalized form.
        refs.append(NodeRef(graph=from_ref.graph, kind=from_ref.kind, id=tid))
    return refs


def _dependency_intents(
    raw: Any,
    task_ref: NodeRef,
    operation: DependencyOperation,
    direction: DependencyField,
) -> list[DependencyIntent]:
    """Normalize a Task-v2 dependency field to canonical endpoints."""
    peers = _coerce_dep_refs(raw, task_ref, direction)
    if direction == "blocked_by":
        return [DependencyIntent(operation, task_ref, peer, direction) for peer in peers]
    return [DependencyIntent(operation, peer, task_ref, direction) for peer in peers]
