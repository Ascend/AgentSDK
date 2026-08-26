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

# AgentSDK publishes this standalone src-layout package across independent Parts; the complete
# ClawCodex source and focused tests validate imports and dynamic patterns during migration.
# The target hook also enables legacy default diagnostics beyond its declared high-value set.
# pylint: disable=C0207,E0402,E0611,W0706

"""JSON board persistence, atomic mutation, and recovery operations."""

from __future__ import annotations

import json
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable

from .atomic_file import atomic_write_json
from . import audit_compaction as audit
from .commands import CommandResult
from .error_codes import coerce_error_code
from .graph_types import Board, GraphSnapshot, RevisionVector
from .json_store_models import (
    BoardEnvelope,
    BoardNotFoundError,
    BoardRecoveryWarning,
    BoardSchemaTooNewError,
    BoardStoreCorruptError,
    BoardTombstonedError,
    CURRENT_SCHEMA_VERSION,
    IdempotencyKeyReusedError,
    STORE_FORMAT,
    StaleRevisionError,
    validate_envelope_schema,
    verify_payload_hash,
    payload_hash,
    set_payload_hash,
)
from .json_store_helpers import build_command_audit_fields, build_graph_content


class _PayloadHashMismatchError(ValueError):
    """Deterministic integrity failure; never retry a known-bad snapshot."""


class JsonBoardStore:
    """Crash-consistent JSON board store (spec §7.3 – §7.6, §7.12).

    Responsibilities:
    * Load / validate board envelopes from disk
    * Execute atomic mutations under a BoardFileLock (two-phase)
    * Idempotency via ``processedCommands`` (LKB-STORE-004/005)
    * Revision CAS via ``expected_revision_vector`` (LKB-STORE-006/007)
    * Corruption recovery via .bak rotation (spec §7.12)

    Parameters
    ----------
    board_dir:
        Path to the board directory (contains board.json, .lock, etc.).
    board_id:
        Expected board_id — re-validated on every load (LKB-STORE-028).
    lock:
        A ``BoardFileLock`` instance (or compatible context manager)
        that provides exclusive cross-process + thread access.
    home:
        Legacy optional LKB storage root.  New callers should prefer
        ``lkb_root`` so the meaning is explicit.
    lkb_root:
        Optional explicit LKB storage root.  Use this for relocated stores;
        it takes precedence over ``home`` and avoids deriving ownership from
        the board directory layout.
    failpoint:
        Optional ``Failpoint`` for crash-injection testing.
    """

    def __init__(
        self,
        board_dir: Path | str,
        *,
        board_id: str,
        lock: Any,
        home: Path | None = None,
        lkb_root: Path | None = None,
        failpoint: Any | None = None,
    ) -> None:
        self._board_dir = Path(board_dir)
        self._board_id = board_id
        self._lock = lock
        self._home = home
        self._lkb_root = lkb_root
        self._failpoint = failpoint

        self._board_json = self._board_dir / "board.json"
        self._board_json_bak = self._board_dir / "board.json.bak"
        self._tmp_dir = self._board_dir / ".tmp"
        self._quarantine_dir = self._board_dir / "quarantine"

    @property
    def board_dir(self) -> Path:
        """Directory containing this board's authoritative JSON state."""
        return self._board_dir

    def _tombstone_path(self) -> Path:
        from .board_resolver import safe_board_id

        # Keep the historical ``home`` keyword as an LKB-root alias.  New
        # callers use ``lkb_root`` so relocated stores never depend on board
        # directory shape.
        lkb_root = (
            self._lkb_root
            if self._lkb_root is not None
            else self._home
            if self._home is not None
            else self._board_dir.parent.parent
        )
        return lkb_root / "tombstones" / f"{safe_board_id(self._board_id)}.json"

    def _assert_not_tombstoned(self) -> None:
        marker = self._tombstone_path()
        if marker.is_file():
            from .lifecycle import read_tombstone

            read_tombstone(marker, expected_board_id=self._board_id)
            raise BoardTombstonedError(self._board_id, marker)

    # ── public read API ───────────────────────────────────────────────

    def load(self) -> BoardEnvelope:
        """Load the primary, or explicitly recover an invalid primary.

        A primary that is readable and hash-valid is authoritative (spec
        §7.12).  Its revision chain is then checked against any backup: a
        same-revision fork or a rollback raises ``BoardStoreCorruptError``
        and must *not* be silently recovered from.  Only an unreadable or
        hash-invalid primary falls through to ``.bak`` recovery.
        """
        self._assert_not_tombstoned()
        self._migrate_primary_if_needed()
        try:
            primary = self._read_valid_envelope(self._board_json)
        except BoardSchemaTooNewError:
            raise
        except (OSError, json.JSONDecodeError, ValueError):
            # Primary unreadable or hash-invalid - attempt .bak recovery.
            recovered = self._try_recover_from_backup()
            if recovered is not None:
                self._assert_not_tombstoned()
                return recovered
            raise BoardStoreCorruptError(
                f"Both {self._board_json} and {self._board_json_bak} are "
                f"corrupt or invalid for board {self._board_id!r}"
            ) from None
        # Primary is valid; validate its chain against the backup.  A fork
        # or rollback detected here must propagate, not trigger recovery.
        self._validate_primary_chain(primary)
        self._assert_not_tombstoned()
        if audit.needs_compaction(primary):
            warnings.warn(
                "LKB Audit is above its soft limit; run `/lkb board compact`.",
                audit.AuditSizeWarning,
                stacklevel=2,
            )
        return primary

    def read_snapshot(self) -> GraphSnapshot:
        """Read and validate only the authoritative primary snapshot."""
        self._assert_not_tombstoned()
        self._migrate_primary_if_needed()
        last_error: Exception | None = None
        envelope: BoardEnvelope | None = None
        for attempt in range(3):
            try:
                envelope = self._read_valid_envelope(self._board_json)
                break
            except BoardSchemaTooNewError:
                raise
            except _PayloadHashMismatchError as exc:
                raise BoardStoreCorruptError(
                    f"Primary snapshot for board {self._board_id!r} has a payload hash mismatch"
                ) from exc
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                # Transient: may have caught an in-progress atomic write.
                last_error = exc
            if attempt < 2:
                time.sleep(0.005 * (attempt + 1))
        if envelope is None:
            raise BoardStoreCorruptError(
                f"Cannot read valid primary snapshot for board {self._board_id!r}: {last_error}"
            )
        # Chain validation is not transient - a fork or rollback must
        # surface immediately rather than be retried or masked.
        self._validate_primary_chain(envelope)
        snapshot = envelope.build_graph_snapshot()
        self._assert_not_tombstoned()
        return snapshot

    def header(self) -> dict[str, Any]:
        """Return a cheap header dict for board listings.

        Parses the JSON document and extracts only the summary fields
        required by board listings.  It does not validate or materialize a
        ``BoardEnvelope``.
        """
        data = self._read_json_file(self._board_json)
        board = data.get("board", {}) if isinstance(data, dict) else {}
        return {
            "board_id": board.get("board_id", ""),
            "display_name": board.get("display_name", ""),
            "schema_version": data.get("schemaVersion", 0) if isinstance(data, dict) else 0,
            "store_revision": data.get("storeRevision", 0) if isinstance(data, dict) else 0,
            "lifecycle_state": (data.get("lifecycle", {}) or {}).get("state", "active")
            if isinstance(data, dict)
            else "unknown",
        }

    def exists(self) -> bool:
        """Return True if board.json exists (regardless of validity)."""
        return self._board_json.is_file()

    # ── public write API ──────────────────────────────────────────────

    def execute_atomic(
        self,
        board_id: str,
        command_id: str,
        request_hash: str,
        expected_revision_vector: RevisionVector | None,
        mutate: Callable[[BoardEnvelope], tuple[BoardEnvelope, CommandResult]],
        *,
        expected_store_revision: int | None = None,
        actor: str,
        reason: str | None = None,
        lifecycle_operation: bool = False,
        audit_maintenance: bool = False,
        audit_context: dict[str, Any] | None = None,
    ) -> CommandResult:
        """Execute *mutate* atomically against the board (spec §7.6).

        Two-phase protocol:
          1. Acquire the board lock.
          2. Re-read and validate board.json.
          3. Check idempotency (processedCommands):
             - same command_id + same request_hash → return cached result
             - same command_id + different hash → IdempotencyKeyReusedError
          4. Check expected store/graph revision CAS guards.
          5. Call *mutate(envelope)* → (candidate, result).
          6. Validate candidate (schema + payload hash + board invariants).
          7. Bump store_revision + modified graph revisions.
          8. Append command result + validation + commit audit to candidate.
          9. Atomic write with .bak rotation.
         10. Return result.

        Parameters
        ----------
        board_id:
            Must match the store's board_id (defence-in-depth).
        command_id:
            Idempotency key.
        request_hash:
            Hash of the command payload (for idempotency check).
        expected_revision_vector:
            If provided, all graph revisions in the vector must match
            the current state (CAS check).  Graphs not mentioned in the
            vector are not checked.
        expected_store_revision:
            If provided, the board's store revision must match exactly.
        mutate:
            Callable receiving the current BoardEnvelope and returning
            ``(modified_envelope, CommandResult)``.  The envelope is a
            deep copy so the callback can mutate it freely.

        Returns
        -------
        CommandResult
            The result from *mutate* (possibly enriched with revision
            info on successful commit).

        Raises
        ------
        IdempotencyKeyReusedError
            command_id reused with different request_hash.
        StaleRevisionError
            expected_revision_vector doesn't match current state.
        BoardStoreCorruptError
            Board is unreadable and unrecoverable.
        """
        if board_id != self._board_id:
            raise ValueError(f"board_id mismatch: store owns {self._board_id!r}, command targets {board_id!r}")

        self._assert_not_tombstoned()
        with self._lock:
            # Re-check after serialization: a purge may have published the
            # permanent marker while this caller was waiting for the lock.
            self._assert_not_tombstoned()
            # Step 2: re-read + validate (with .bak recovery)
            envelope = self._load_locked()

            # Step 3: idempotency check
            existing = envelope.processed_commands.get(command_id)
            if existing is not None:
                stored_hash = existing.get("request_hash", "")
                if stored_hash == request_hash:
                    # LKB-STORE-004: same id + same hash → cached result
                    decision = existing.get("decision", "committed")
                    rev_vec_dict = existing.get("revision_vector")
                    rev_vec = RevisionVector(revisions=dict(rev_vec_dict)) if isinstance(rev_vec_dict, dict) else None
                    return CommandResult(
                        decision=decision,  # type: ignore[arg-type]
                        command_id=command_id,
                        revision_vector=rev_vec,
                        validation_run_id=existing.get("validation_run_id"),
                        error_code=coerce_error_code(existing.get("error_code")),
                        reason=existing.get("reason"),
                        derived_facts=tuple(existing.get("derived_facts", ())),
                        claim_id=existing.get("claim_id"),
                        affected_refs=tuple(existing.get("affected_refs", ())),
                    )
                else:
                    # LKB-STORE-005: same id, different hash → reuse error
                    raise IdempotencyKeyReusedError(command_id, stored_hash, request_hash)

            audit_bytes = audit.active_audit_size(envelope)
            if audit.needs_compaction(envelope):
                warnings.warn(
                    "LKB Audit is above its soft limit; run `/lkb board compact`.",
                    audit.AuditSizeWarning,
                    stacklevel=2,
                )
            if audit_bytes >= audit.AUDIT_HARD_MAX_BYTES and not audit_maintenance:
                raise audit.AuditSizeLimitError(
                    "LKB Audit reached the hard limit; run `/lkb board compact` before issuing more writes."
                )

            if not lifecycle_operation:
                from .lifecycle import ordinary_write_denial_reason

                denial = ordinary_write_denial_reason(envelope)
                if denial is not None:
                    raise PermissionError(denial)

            # Step 4: expected revision CAS checks
            if expected_store_revision is not None and envelope.store_revision != expected_store_revision:
                current = envelope.current_revision_vector()
                raise StaleRevisionError(
                    board_id,
                    expected_revision_vector or RevisionVector(),
                    current,
                    reason=(f"store revision: expected {expected_store_revision}, got {envelope.store_revision}"),
                )
            if expected_revision_vector is not None:
                current = envelope.current_revision_vector()
                for gid, expected_rev in expected_revision_vector.revisions.items():
                    actual_rev = current.get(gid)
                    if actual_rev != expected_rev:
                        raise StaleRevisionError(
                            board_id,
                            expected_revision_vector,
                            current,
                            reason=f"graph {gid!r}: expected rev {expected_rev}, got {actual_rev}",
                        )

            # Capture pre-mutation state for comparison.
            pre_store_rev = envelope.store_revision
            previous_hash = envelope.integrity.get("payloadHash", "")

            # Step 5: mutate
            candidate, result = mutate(envelope.clone())
            if not isinstance(candidate, BoardEnvelope):
                raise TypeError("mutate must return a BoardEnvelope candidate")
            if not isinstance(result, CommandResult):
                raise TypeError("mutate must return a CommandResult")

            # Step 6: validate candidate
            validate_envelope_schema(candidate.to_dict(), board_id=board_id)
            # Recompute payload hash (with previous chain)
            set_payload_hash(candidate, previous_hash=previous_hash)

            # Step 7: compute graph revisions from exact per-graph content.
            # This is the only graph-revision authority; count heuristics can
            # miss same-cardinality replacements and are therefore unsafe.
            pre_content = build_graph_content(envelope)
            post_content = build_graph_content(candidate)
            for gid, graph in candidate.graphs.items():
                old_revision = int(envelope.graphs.get(gid, {}).get("revision", 0))
                graph["revision"] = old_revision + (1 if pre_content.get(gid) != post_content.get(gid) else 0)

            # Bump store_revision by 1
            candidate.store_revision = pre_store_rev + 1
            candidate.board["store_revision"] = candidate.store_revision

            # Step 8: append command + audit to processedCommands
            rev_vec = candidate.current_revision_vector()
            entry: dict[str, Any] = {
                "command_id": command_id,
                "request_hash": request_hash,
                "decision": result.decision,
                "actor": actor,
                "store_revision": candidate.store_revision,
                "revision_vector": rev_vec.to_dict(),
                "validation_run_id": result.validation_run_id,
                "reason": result.reason,
                "derived_facts": list(result.derived_facts),
            }
            if result.error_code:
                entry["error_code"] = str(result.error_code)
            if result.claim_id:
                entry["claim_id"] = result.claim_id
            if result.affected_refs:
                entry["affected_refs"] = list(result.affected_refs)
            if reason is not None:
                entry["audit_reason"] = reason
            candidate.processed_commands[command_id] = entry

            # Append to event log (spec §6.10 — MUST include: event_id,
            # board_id, store_revision, command_id, actor, timestamp,
            # subject_ref, decision, rule/reason, input snapshot hash,
            # validation_run_id, affected_refs).
            subject_ref_val = ""
            affected_refs_val: list[str] = []
            input_snapshot_hash = previous_hash or ""
            if audit_context:
                subject_ref_val = str(audit_context.get("subject_ref") or "")
                affected = audit_context.get("affected_refs")
                if affected:
                    affected_refs_val = [str(r) for r in affected]
                # Issue #9: input_snapshot_hash is the GraphSnapshot hash the
                # validator read (supplied by the application service via
                # audit_context), NOT the previous Board payload hash.
                input_snapshot_hash = str(audit_context.get("input_snapshot_hash") or input_snapshot_hash)
            if not affected_refs_val and result.affected_refs:
                affected_refs_val = list(result.affected_refs)
            rule_val = str((audit_context or {}).get("rule") or "")
            # Issue #9: an independent ``command_received`` event records
            # that the command was accepted for execution (spec §6.10 lists
            # it among the events every command must produce).  It carries
            # the same MUST fields as command_executed so the audit schema is
            # uniform.  It is distinct from ``command_executed`` (the outcome)
            # and from the ``processedCommands`` map (the idempotency receipt).
            common_audit_fields = build_command_audit_fields(
                board_id=board_id,
                command_id=command_id,
                result=result,
                actor=actor,
                store_revision=candidate.store_revision,
                revision_vector=rev_vec,
                input_snapshot_hash=input_snapshot_hash,
                validation_run_id=result.validation_run_id,
                subject_ref=subject_ref_val,
                affected_refs=affected_refs_val,
                rule=rule_val,
            )
            candidate.events.append(
                {
                    **common_audit_fields,
                    "type": "command_received",
                    "event_id": f"E-{uuid.uuid4().hex[:16]}",
                    "request_hash": request_hash,
                    "reason": reason or "",
                }
            )
            event: dict[str, Any] = {
                **common_audit_fields,
                "type": "command_executed",
                "event_id": f"E-{uuid.uuid4().hex[:16]}",
                "reason": result.reason or "",
            }
            candidate.events.append(event)

            # Issue #9: override / invalidation custom events appended by
            # domain handlers were stamped with the PRE-bump store_revision
            # (the candidate revision is only advanced above).  Patch every
            # command-scoped event to the authoritative post-bump revision
            # so the audit chain is internally consistent.
            for ev in candidate.events:
                if ev.get("command_id") == command_id and ev is not event:
                    ev["store_revision"] = candidate.store_revision

            if audit.active_audit_size(candidate) > audit.AUDIT_HARD_MAX_BYTES and not audit_maintenance:
                raise audit.AuditSizeLimitError(
                    "LKB Audit write would exceed the hard limit; run `/lkb board compact` and retry."
                )

            # Re-hash after all mutations
            set_payload_hash(candidate, previous_hash=previous_hash)

            # Final schema/decode/invariant and serialization validation.
            candidate_data = candidate.to_dict()
            validate_envelope_schema(candidate_data, board_id=board_id)
            if not verify_payload_hash(candidate_data):
                raise ValueError("candidate payload hash does not verify")
            json.dumps(candidate_data, sort_keys=True, ensure_ascii=False)

            # Step 9: atomic write
            self._write_atomic(candidate)

            # Step 10: return result (with final revision vector)
            if result.committed:
                return CommandResult(
                    decision=result.decision,
                    command_id=command_id,
                    revision_vector=rev_vec,
                    validation_run_id=result.validation_run_id,
                    error_code=result.error_code,
                    reason=result.reason,
                    derived_facts=result.derived_facts,
                    claim_id=result.claim_id,
                    affected_refs=result.affected_refs,
                )
            return result

    def compact_audit(self, board_id: str, *, actor: str) -> audit.AuditCompactionResult:
        """Manually move old Audit material to an immutable history segment."""
        before = self.load()
        bytes_before = audit.active_audit_size(before)
        initial_plan = audit.build_compaction_plan(before)
        if initial_plan.empty:
            return audit.AuditCompactionResult(None, None, 0, 0, 0, bytes_before, bytes_before)

        command_id = f"audit-compact-{uuid.uuid4().hex[:12]}"
        request_hash = f"audit-compact:{before.store_revision}:{before.integrity.get('payloadHash', '')}"
        summary: dict[str, Any] = {}

        def mutate(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
            plan = audit.build_compaction_plan(envelope)
            previous_segment_hash = ""
            if envelope.history_segments:
                previous_segment_hash = str(envelope.history_segments[-1].get("sha256") or "")
            manifest = audit.write_history_segment(
                self._board_dir,
                plan,
                previous_segment_hash=previous_segment_hash,
            )
            envelope.events = envelope.events[len(plan.events) :]
            for command_key, record in plan.processed_commands:
                envelope.processed_commands[command_key] = audit.thin_processed_command(
                    record, str(manifest["segmentId"])
                )
            for run_id, _record in plan.validation_runs:
                envelope.validation_runs.pop(run_id, None)
            envelope.history_segments.append(manifest)
            summary.update(
                manifest=manifest,
                event_count=len(plan.events),
                processed_count=len(plan.processed_commands),
                validation_count=len(plan.validation_runs),
            )
            return envelope, CommandResult(
                decision="committed",
                command_id=command_id,
                reason=(f"compacted {len(plan.events)} events and {len(plan.processed_commands)} command summaries"),
            )

        self.execute_atomic(
            board_id,
            command_id,
            request_hash,
            None,
            mutate,
            expected_store_revision=before.store_revision,
            actor=actor,
            reason="manual audit compaction",
            audit_maintenance=True,
        )
        after = self.load()
        manifest = summary["manifest"]
        return audit.AuditCompactionResult(
            str(manifest["segmentId"]),
            str(manifest["file"]),
            int(summary["event_count"]),
            int(summary["processed_count"]),
            int(summary["validation_count"]),
            bytes_before,
            audit.active_audit_size(after),
        )

    # ── internal: read + validate ─────────────────────────────────────

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        """Read and parse a JSON file.  Raises on I/O or parse error."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_valid_envelope(self, path: Path) -> BoardEnvelope:
        data = self._read_json_file(path)
        validate_envelope_schema(data, board_id=self._board_id)
        if not verify_payload_hash(data):
            raise _PayloadHashMismatchError(f"{path} payload hash mismatch")
        return BoardEnvelope.from_dict(data)

    def _migrate_primary_if_needed(self) -> None:
        try:
            raw = self._read_json_file(self._board_json)
        except (OSError, json.JSONDecodeError):
            return
        schema_version = raw.get("schemaVersion", 0)
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            return
        if schema_version > CURRENT_SCHEMA_VERSION:
            raise BoardSchemaTooNewError(self._board_id, schema_version, CURRENT_SCHEMA_VERSION)
        if schema_version == CURRENT_SCHEMA_VERSION:
            return
        with self._lock:
            latest = self._read_json_file(self._board_json)
            latest_version = latest.get("schemaVersion", 0)
            if latest_version > CURRENT_SCHEMA_VERSION:
                raise BoardSchemaTooNewError(self._board_id, latest_version, CURRENT_SCHEMA_VERSION)
            if latest_version == CURRENT_SCHEMA_VERSION:
                return
            from .migrations import migrate_board_file

            migrate_board_file(
                self._board_json,
                expected_board_id=self._board_id,
                target_schema=CURRENT_SCHEMA_VERSION,
                failpoint=self._failpoint,
            )

    def _read_and_validate(self, path: Path) -> BoardEnvelope | None:
        try:
            return self._read_valid_envelope(path)
        except BoardSchemaTooNewError:
            raise
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def _validate_primary_chain(self, primary: BoardEnvelope) -> None:
        if not self._board_json_bak.is_file():
            return
        try:
            backup_raw = self._read_json_file(self._board_json_bak)
        except (OSError, json.JSONDecodeError):
            # A valid primary is authoritative.  A damaged optional backup is
            # diagnostic material, not a reason to make healthy state
            # unavailable.
            return
        backup_schema = backup_raw.get("schemaVersion", 0)
        if isinstance(backup_schema, int) and backup_schema < CURRENT_SCHEMA_VERSION:
            # A schema migration backup is intentionally from the previous
            # schema and is not part of the v1 revision hash chain.
            return
        backup = self._read_and_validate(self._board_json_bak)
        if backup is None:
            return
        if primary.store_revision == backup.store_revision:
            # ``atomic_write_json`` rotates the current authoritative
            # primary into .bak before replacing it.  A crash at
            # after_backup_before_replace therefore leaves two identical,
            # complete copies of the old revision.  This is a valid
            # pre-commit state, not a rollback or a broken revision chain.
            if (
                primary.integrity.get("payloadHash") == backup.integrity.get("payloadHash")
                and primary.to_dict() == backup.to_dict()
            ):
                return
            raise BoardStoreCorruptError("primary/backup have the same revision but different payloads")
        if primary.store_revision < backup.store_revision:
            raise BoardStoreCorruptError(
                "primary revision is older than backup (possible rollback): "
                f"{primary.store_revision} vs {backup.store_revision}"
            )
        if primary.store_revision > backup.store_revision + 1:
            # Backup rotation is best-effort historical context.  A valid
            # lower non-adjacent revision is merely stale and cannot refute
            # the authoritative primary.
            return
        if primary.integrity.get("previousPayloadHash") != backup.integrity.get("payloadHash"):
            raise BoardStoreCorruptError("primary previousPayloadHash does not match backup")

    def _backup_revision_is_explainable(self, backup: BoardEnvelope) -> bool:
        try:
            raw = self._read_json_file(self._board_json)
        except (OSError, json.JSONDecodeError):
            return True
        raw_board = raw.get("board")
        if isinstance(raw_board, dict) and raw_board.get("board_id") not in (
            None,
            self._board_id,
        ):
            return False
        raw_revision = raw.get("storeRevision")
        if isinstance(raw_revision, int):
            if raw_revision != backup.store_revision + 1:
                return False
            raw_integrity = raw.get("integrity")
            if isinstance(raw_integrity, dict):
                previous = raw_integrity.get("previousPayloadHash")
                if previous is not None and previous != backup.integrity.get("payloadHash"):
                    return False
        return True

    def _quarantine_copy(self, path: Path, reason: str) -> Path | None:
        if not path.exists():
            return None
        try:
            self._quarantine_dir.mkdir(parents=True, exist_ok=True)
            target = self._quarantine_dir / f"{path.name}.{time.time_ns()}.{reason}"
            shutil.copy2(path, target)
            return target
        except OSError:
            return None

    def _try_recover_from_backup(self) -> BoardEnvelope | None:
        with self._lock:
            try:
                primary = self._read_valid_envelope(self._board_json)
                self._validate_primary_chain(primary)
                return primary
            except BoardSchemaTooNewError:
                raise
            except (OSError, json.JSONDecodeError, ValueError, BoardStoreCorruptError):
                pass

            backup = self._read_and_validate(self._board_json_bak)
            if backup is None or not self._backup_revision_is_explainable(backup):
                return None

            recovered = backup.clone()
            recovered.store_revision = backup.store_revision + 1
            recovered.board["store_revision"] = recovered.store_revision
            recovered.events.append(
                {
                    "type": "store_recovered",
                    "actor": "json_store",
                    "reason": "primary invalid; restored from board.json.bak",
                    "recovered_from_store_revision": backup.store_revision,
                    "store_revision": recovered.store_revision,
                }
            )
            set_payload_hash(
                recovered,
                previous_hash=str(backup.integrity.get("payloadHash", "")),
            )
            recovered_data = recovered.to_dict()
            validate_envelope_schema(recovered_data, board_id=self._board_id)
            self._quarantine_copy(self._board_json, "primary-corrupt")
            atomic_write_json(
                self._board_json,
                recovered_data,
                backup_path=None,
                fsync_dir=True,
                failpoint=self._failpoint,
                payload_hash_key="payloadHash",
            )
            warnings.warn(
                f"Recovered board {self._board_id!r} from board.json.bak; the invalid primary was quarantined",
                BoardRecoveryWarning,
                stacklevel=2,
            )
            return recovered

    def _load_locked(self) -> BoardEnvelope:
        try:
            primary = self._read_valid_envelope(self._board_json)
            self._validate_primary_chain(primary)
            return primary
        except BoardSchemaTooNewError:
            raise
        except (OSError, json.JSONDecodeError, ValueError, BoardStoreCorruptError) as exc:
            raise BoardStoreCorruptError(
                f"{self._board_json} is invalid for board {self._board_id!r}; "
                "mutations do not implicitly recover from backup"
            ) from exc

    def _write_atomic(self, envelope: BoardEnvelope) -> None:
        """Atomically write *envelope* to board.json with .bak rotation.

        Delegates to atomic_file.atomic_write_json.  Failpoint-injectable.
        """
        self._board_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        data = envelope.to_dict()
        atomic_write_json(
            self._board_json,
            data,
            backup_path=self._board_json_bak,
            fsync_dir=True,
            failpoint=self._failpoint,
            payload_hash_key="integrity.payloadHash".split(".")[-1],
        )

    # ── initialization helper ─────────────────────────────────────────

    @classmethod
    def create_board(
        cls,
        board_dir: Path | str,
        *,
        board: Board,
        lock: Any,
        home: Path | None = None,
        lkb_root: Path | None = None,
        failpoint: Any | None = None,
    ) -> "JsonBoardStore":
        """Create a new board on disk and return a JsonBoardStore for it.

        Writes the genesis envelope (store_revision=0, empty graphs,
        initial payload hash).  If the board already exists, raises
        FileExistsError.
        """
        board_dir = Path(board_dir)
        store = cls(
            board_dir,
            board_id=board.board_id,
            lock=lock,
            home=home,
            lkb_root=lkb_root,
            failpoint=failpoint,
        )
        store._assert_not_tombstoned()

        envelope = BoardEnvelope(
            store_format=STORE_FORMAT,
            schema_version=CURRENT_SCHEMA_VERSION,
            store_revision=0,
            board={
                "board_id": board.board_id,
                "project_uri": board.project_uri,
                "display_name": board.display_name,
                "schema_version": board.schema_version,
                "store_revision": 0,
                "created_at": board.created_at,
                "updated_at": board.updated_at,
                "policy": board.policy.to_dict(),
            },
            lifecycle={},
        )
        from .lifecycle import genesis_lifecycle

        envelope.lifecycle = genesis_lifecycle(
            scope=("session" if board.project_uri.startswith("session:") else "project"),
            created_at=board.created_at,
            origin_project_uri=board.project_uri,
        )
        # Genesis hash — no previous
        set_payload_hash(envelope, previous_hash=None)

        # First creation participates in the same serialization order as
        # every later mutation.
        with lock:
            # The first check above avoids needless directory creation; this
            # in-lock check closes the race with tombstone publication.
            store._assert_not_tombstoned()
            board_dir.mkdir(parents=True, exist_ok=True)
            if store._board_json.exists():
                raise FileExistsError(f"Board already exists: {store._board_json}")
            store._write_atomic(envelope)
        return store


def validate_board_envelope(
    envelope: BoardEnvelope | dict[str, Any],
    *,
    board_id: str | None = None,
    verify_hash: bool = True,
) -> None:
    """Public Phase-2 schema/decode/invariant oracle."""
    data = envelope.to_dict() if isinstance(envelope, BoardEnvelope) else envelope
    validate_envelope_schema(data, board_id=board_id)
    if verify_hash and not verify_payload_hash(data):
        raise AssertionError("payload hash does not match envelope content")
    BoardEnvelope.from_dict(data).build_graph_snapshot()


# Backward-compatible internal spellings for unselected legacy consumers.
_validate_envelope_schema = validate_envelope_schema
_verify_payload_hash = verify_payload_hash


__all__ = [
    "BoardEnvelope",
    "BoardNotFoundError",
    "BoardRecoveryWarning",
    "BoardSchemaTooNewError",
    "BoardStoreCorruptError",
    "BoardTombstonedError",
    "CURRENT_SCHEMA_VERSION",
    "IdempotencyKeyReusedError",
    "JsonBoardStore",
    "STORE_FORMAT",
    "StaleRevisionError",
    "payload_hash",
    "set_payload_hash",
    "validate_envelope_schema",
    "validate_board_envelope",
    "verify_payload_hash",
]
