# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
"""parentUuid chain + walkChainBeforeParse 读取过滤。

Test coverage for the decoupled implementation:

* Write side (``extensions/agent/session_persist.py::_inject_parent_uuids``)
* Read side (``clawcodex_ext/agent/chain_filter.py``)
* Integration (``clawcodex_ext/agent/session.py::_load_from_enhanced_transcript``)

Acceptance scenarios:
1. New-format write produces a correct parentUuid chain.
2. ``/rewind`` → new messages form a branch on disk.
3. ``walkChainBeforeParse`` drops dead-branch lines above the
   ratio threshold.
4. ``--resume`` (via ``Session.load``) returns only the active chain.
5. Legacy transcripts (no ``parentUuid``) skip the filter and
   return every message unchanged.
6. ``walkChainBeforeParse`` gate short-circuits on small / low
   dead-branch-ratio transcripts.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Ensure repo root is on sys.path so ``extensions`` / ``clawcodex_ext``
# imports resolve when the test is run from anywhere in the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clawcodex_ext.agent.chain_filter import (  # noqa: E402
    ABS_SIZE_THRESHOLD,
    ChainFilterConfig,
    build_conversation_chain,
    filter_active_chain_messages,
    walk_chain_before_parse,
)
from extensions.agent.session_persist import _inject_parent_uuids  # noqa: E402


def _msg(uuid: str, parent_uuid, role: str, content: str) -> dict:
    """Build a chat message dict with the ``parentUuid`` field."""
    return {
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "role": role,
        "content": content,
    }


def _msg_json(uuid: str, parent_uuid, role: str, content: str) -> str:
    """Serialise a chat message dict to a JSONL string."""
    return json.dumps(_msg(uuid, parent_uuid, role, content), ensure_ascii=False)


def _make_session_init(session_id: str = "s-test") -> str:
    """Return a serialised ``session_init`` line."""
    return json.dumps(
        {
            "type": "session_init",
            "session_id": session_id,
            "provider": "test",
            "model": "test-model",
            "cwd": "/tmp",
            "created_at": "2026-06-22T00:00:00",
        },
        ensure_ascii=False,
    )


def _snapshot_line() -> str:
    return json.dumps(
        {
            "type": "session_snapshot",
            "cost": {},
            "updated_at": "2026-06-22T00:01:00",
        },
        ensure_ascii=False,
    )


def _pad_lines(count: int) -> list[str]:
    """Generate ``count`` filler lines (no parentUuid) for size-gate tests."""
    return [json.dumps({"filler": "x" * 200}, ensure_ascii=False) for _ in range(count)]


class TestInjectParentUuids(unittest.TestCase):
    """P103-E — write-side parentUuid stamping."""

    def test_root_message_has_null_parent(self):
        out = _inject_parent_uuids([_msg("u1", None, "user", "hi")])
        self.assertIsNone(out[0]["parentUuid"])

    def test_chain_topology(self):
        msgs = [
            _msg("u1", None, "user", "hi"),
            _msg("u2", "u1", "assistant", "hello"),
            _msg("u3", "u2", "user", "again"),
            _msg("u4", "u3", "assistant", "ok"),
        ]
        out = _inject_parent_uuids(msgs)
        self.assertIsNone(out[0]["parentUuid"])
        self.assertEqual(out[1]["parentUuid"], "u1")
        self.assertEqual(out[2]["parentUuid"], "u2")
        self.assertEqual(out[3]["parentUuid"], "u3")

    def test_rewind_creates_fork_topology(self):
        """After /rewind the new messages form a branch pointing at the rewind target."""
        # conversation.messages post-rewind (truncated to rewind target + new msg).
        # The NEW message u5 has no explicit parentUuid because the
        # API creates it without one — _inject_parent_uuids is
        # responsible for stamping it.
        post_rewind = [
            _msg("u1", None, "user", "hi"),
            _msg("u2", "u1", "assistant", "hello"),
            {"uuid": "u5", "role": "user", "content": "after rewind"},
        ]
        out = _inject_parent_uuids(post_rewind)
        # Chain must be u1 -> u2 -> u5 (NOT u4), proving the fork
        self.assertEqual([m["uuid"] for m in out], ["u1", "u2", "u5"])
        self.assertEqual(out[2]["parentUuid"], "u2")

    def test_always_recomputes_existing_parentUuid(self):
        """Design mandates recomputation on every write (写入时计算)."""
        msgs = [
            {"uuid": "u1", "role": "user", "content": "x"},  # no parentUuid field
        ]
        out = _inject_parent_uuids(msgs)
        self.assertIsNone(out[0]["parentUuid"])

    def test_missing_uuid_does_not_break_chain(self):
        """Defensive: messages without uuid get parentUuid=None but don't advance cursor."""
        msgs = [
            {"role": "system", "content": "hi"},  # no uuid
            _msg("u1", None, "user", "hi"),
        ]
        out = _inject_parent_uuids(msgs)
        # First entry has no uuid → parentUuid=None, cursor not advanced
        self.assertIsNone(out[0]["parentUuid"])
        # Second entry has uuid u1, no previous → parentUuid=None
        self.assertIsNone(out[1]["parentUuid"])

    def test_does_not_mutate_caller_list(self):
        """Pure function: input list and its dicts are not mutated."""
        original_entry = {"uuid": "u1", "role": "user", "content": "hi"}
        original = [original_entry]
        _inject_parent_uuids(original)
        # Original list and its dict are untouched
        self.assertNotIn("parentUuid", original[0])
        # The original dict reference is the same object (not replaced)
        self.assertIs(original[0], original_entry)


class TestWalkChainBeforeParse(unittest.TestCase):
    """P103-B — byte-level chain pruning."""

    def test_empty_input_returns_empty(self):
        result = walk_chain_before_parse(b"")
        self.assertTrue(result.skipped)
        self.assertEqual(result.raw_bytes, b"")

    def test_small_transcript_skips_scan(self):
        """Tiny transcripts short-circuit on the size gate."""
        lines = [_make_session_init(), _msg_json("u1", None, "user", "hi")]
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        self.assertLess(len(raw), ABS_SIZE_THRESHOLD)
        result = walk_chain_before_parse(raw)
        self.assertTrue(result.skipped)
        self.assertEqual(result.raw_bytes, raw)

    def test_legacy_transcript_no_parentUuid_skips_filter(self):
        """Transcripts without ``parentUuid`` tokens cannot be chain-pruned."""
        lines = [
            _make_session_init(),
            json.dumps({"uuid": "u1", "role": "user", "content": "hi"}),
            json.dumps({"uuid": "u2", "role": "assistant", "content": "hello"}),
        ]
        lines = lines + _pad_lines(50)
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        self.assertGreater(len(raw), ABS_SIZE_THRESHOLD)
        result = walk_chain_before_parse(raw)
        self.assertTrue(result.skipped)
        self.assertEqual(result.raw_bytes, raw)

    def test_high_dead_branch_ratio_filters(self):
        """When dead branches exceed the threshold, they are pruned."""
        # Active chain: u1 -> u2 -> u5 (after rewind to u2)
        # Dead branches: u3, u4 (2 dead out of 5 messages = 40%)
        chat_lines = [
            _msg_json("u1", None, "user", "hi"),
            _msg_json("u2", "u1", "assistant", "hello"),
            _msg_json("u3", "u2", "user", "q1"),
            _msg_json("u4", "u3", "assistant", "a1"),
            _msg_json("u5", "u2", "user", "after rewind"),
        ]
        padding = _pad_lines(60)
        lines = [_make_session_init()] + chat_lines + padding + [_snapshot_line()]
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        self.assertGreater(len(raw), ABS_SIZE_THRESHOLD)

        result = walk_chain_before_parse(raw)
        self.assertFalse(result.skipped)
        decoded = result.raw_bytes.decode("utf-8")
        # Active chain + metadata retained
        self.assertIn("u1", decoded)
        self.assertIn("u2", decoded)
        self.assertIn("u5", decoded)
        self.assertIn("session_init", decoded)
        self.assertIn("session_snapshot", decoded)
        # Dead branches pruned
        self.assertNotIn('"uuid": "u3"', decoded)
        self.assertNotIn('"uuid": "u4"', decoded)

    def test_low_dead_branch_ratio_skips_filter(self):
        """Below the ratio threshold, the filter is skipped (full parse).

        Configure the threshold low so the gate definitely fires
        on this input — the point is to verify the short-circuit
        path, not to compute the exact ratio for arbitrary data.
        """
        chat_lines = []
        prev = None
        for i in range(10):
            chat_lines.append(_msg_json(f"u{i}", prev, "user", f"m{i}"))
            prev = f"u{i}"
        chat_lines.append(_msg_json("dead1", "u0", "assistant", "dead"))

        padding = _pad_lines(60)
        lines = [_make_session_init()] + chat_lines + padding + [_snapshot_line()]
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        self.assertGreater(len(raw), ABS_SIZE_THRESHOLD)

        # Force the gate to fire by setting a high threshold.
        cfg = ChainFilterConfig(dead_branch_ratio=0.99)
        result = walk_chain_before_parse(raw, config=cfg)
        # Above the artificial threshold → skipped, bytes unchanged
        self.assertTrue(result.skipped)
        self.assertEqual(result.raw_bytes, raw)

    def test_multiple_leaves_picks_latest_leaf(self):
        """When several leaf candidates exist (forked chains), the latest in
        on-disk line order wins. For rewind scenarios this means the
        new branch (most recently appended) is selected over the
        older dead branch even if the dead branch is longer.
        """
        chat_lines = [
            _msg_json("u1", None, "user", "root"),
            _msg_json("u2", "u1", "assistant", "fork1-step"),
            _msg_json("u3", "u1", "assistant", "fork2-step"),
            _msg_json("u4", "u2", "user", "fork1-deep"),
        ]
        padding = _pad_lines(50)
        lines = [_make_session_init()] + chat_lines + padding
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        # Force filter to run regardless of dead-branch ratio
        cfg = ChainFilterConfig(dead_branch_ratio=0.1)
        result = walk_chain_before_parse(raw, config=cfg)
        decoded = result.raw_bytes.decode("utf-8")
        # u3 is at index 2, u4 is at index 3 — u4 is the latest leaf.
        # Walk from u4: u4 -> u2 -> u1. u3 is excluded.
        self.assertIn("u1", decoded)
        self.assertIn("u2", decoded)
        self.assertIn("u4", decoded)
        self.assertNotIn('"uuid": "u3"', decoded)


class TestBuildConversationChain(unittest.TestCase):
    """P103-C — chain reconstruction from leaf to root."""

    def test_empty_input_returns_empty(self):
        self.assertEqual(build_conversation_chain([]), [])

    def test_walks_leaf_to_root(self):
        msgs = [
            _msg("u1", None, "user", "hi"),
            _msg("u2", "u1", "assistant", "hello"),
            _msg("u3", "u2", "user", "q"),
            _msg("u4", "u3", "assistant", "a"),
        ]
        chain = build_conversation_chain(msgs)
        self.assertEqual([m["uuid"] for m in chain], ["u1", "u2", "u3", "u4"])

    def test_excludes_dead_branches(self):
        msgs = [
            _msg("u1", None, "user", "root"),
            _msg("u2", "u1", "assistant", "a"),
            _msg("u3", "u2", "user", "b"),
            _msg("u4", "u3", "assistant", "c"),
            _msg("u5", "u2", "user", "after rewind"),  # fork from u2
        ]
        chain = build_conversation_chain(msgs)
        # u5 is the latest leaf (last in input order); walk back: u5, u2, u1
        self.assertEqual([m["uuid"] for m in chain], ["u1", "u2", "u5"])
        self.assertNotIn("u3", [m["uuid"] for m in chain])
        self.assertNotIn("u4", [m["uuid"] for m in chain])

    def test_explicit_leaf_uuid(self):
        msgs = [
            _msg("u1", None, "user", "root"),
            _msg("u2", "u1", "assistant", "a"),
            _msg("u3", "u2", "user", "b"),
        ]
        chain = build_conversation_chain(msgs, leaf_uuid="u2")
        # Forced leaf u2 → walk: u2, u1
        self.assertEqual([m["uuid"] for m in chain], ["u1", "u2"])

    def test_no_chain_topology_returns_input(self):
        """When no parentUuid links exist, fall back to input order."""
        msgs = [
            {"uuid": "u1", "role": "user", "content": "a"},
            {"uuid": "u2", "role": "assistant", "content": "b"},
        ]
        chain = build_conversation_chain(msgs)
        # No parent links, no dead branches → returns in input order
        self.assertEqual([m["uuid"] for m in chain], ["u1", "u2"])


class TestFilterActiveChainMessages(unittest.TestCase):
    """P103-D — high-level helper for parsed-dict consumers."""

    def test_empty_returns_empty(self):
        self.assertEqual(filter_active_chain_messages([]), [])

    def test_skips_dead_branches_via_bytes_round_trip(self):
        """End-to-end: serialise dicts → walk chain → re-parse → chain."""
        msgs = [
            _msg("u1", None, "user", "hi"),
            _msg("u2", "u1", "assistant", "hello"),
            _msg("u3", "u2", "user", "q1"),
            _msg("u4", "u3", "assistant", "a1"),
            _msg("u5", "u2", "user", "after rewind"),
        ]
        # Pad entries so the size gate fires
        for _ in range(60):
            msgs.append({"filler": "x" * 200})
        result = filter_active_chain_messages(
            msgs,
            config=ChainFilterConfig(dead_branch_ratio=0.5),
        )
        uuids = [m["uuid"] for m in result if m.get("uuid")]
        self.assertIn("u1", uuids)
        self.assertIn("u2", uuids)
        self.assertIn("u5", uuids)
        self.assertNotIn("u3", uuids)
        self.assertNotIn("u4", uuids)


class TestSessionLoadIntegration(unittest.TestCase):
    """P103-G — Session.load / _load_from_enhanced_transcript integration.

    Uses an isolated temporary directory so the test does not touch
    real ``~/.clawcodex/sessions``. The Session class reads from
    ``_get_sessions_dir()`` which is hard-wired to ``Path.home()``,
    so we patch it for the duration of the test.
    """

    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp(prefix="f103-session-")
        self._patched = False

    def tearDown(self):
        if self._patched:
            from clawcodex_ext.agent import session as session_mod

            session_mod._get_sessions_dir = self._original_get_sessions_dir
            self._patched = False

    def _patch_sessions_dir(self):
        from clawcodex_ext.agent import session as session_mod
        from pathlib import Path as _Path

        self._original_get_sessions_dir = session_mod._get_sessions_dir
        session_mod._get_sessions_dir = lambda: _Path(self.tmpdir)
        self._patched = True

    def _write_transcript(self, session_id: str, raw_lines: list[str]) -> Path:
        session_dir = Path(self.tmpdir) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        transcript = session_dir / "transcript.jsonl"
        transcript.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        return transcript

    def test_load_returns_active_chain_after_rewind(self):
        """End-to-end: write a rewind transcript, Session.load returns only active chain."""
        self._patch_sessions_dir()
        session_id = "rewind-test-1"
        init = _make_session_init(session_id)
        chat_lines = [
            _msg_json("u1", None, "user", "hi"),
            _msg_json("u2", "u1", "assistant", "hello"),
            _msg_json("u3", "u2", "user", "q1"),
            _msg_json("u4", "u3", "assistant", "a1"),
            _msg_json("u5", "u2", "user", "after rewind"),
        ]
        chat_lines = chat_lines + _pad_lines(60)
        self._write_transcript(session_id, [init] + chat_lines + [_snapshot_line()])

        from clawcodex_ext.agent.session import Session

        loaded = Session.load(session_id)
        self.assertIsNotNone(loaded)
        uuids = [getattr(m, "uuid", None) for m in loaded.conversation.messages]
        # The active chain is u1 -> u2 -> u5 (latest leaf wins)
        self.assertIn("u1", uuids)
        self.assertIn("u2", uuids)
        self.assertIn("u5", uuids)
        # Dead branches excluded
        self.assertNotIn("u3", uuids)
        self.assertNotIn("u4", uuids)

    def test_load_returns_all_when_legacy_no_parentUuid(self):
        """Legacy transcripts (no parentUuid) return all messages unchanged."""
        self._patch_sessions_dir()
        session_id = "legacy-test-1"
        init = _make_session_init(session_id)
        legacy_lines = [
            json.dumps({"uuid": "u1", "role": "user", "content": "hi"}),
            json.dumps({"uuid": "u2", "role": "assistant", "content": "hello"}),
            json.dumps({"uuid": "u3", "role": "user", "content": "again"}),
        ]
        legacy_lines = legacy_lines + _pad_lines(60)
        self._write_transcript(session_id, [init] + legacy_lines + [_snapshot_line()])

        from clawcodex_ext.agent.session import Session

        loaded = Session.load(session_id)
        self.assertIsNotNone(loaded)
        uuids = [getattr(m, "uuid", None) for m in loaded.conversation.messages]
        self.assertIn("u1", uuids)
        self.assertIn("u2", uuids)
        self.assertIn("u3", uuids)

    def test_load_preserves_legacy_prefix_before_new_chained_messages(self):
        """A resumed legacy transcript keeps history written before parentUuid existed."""
        self._patch_sessions_dir()
        session_id = "mixed-legacy-chain-1"
        init = _make_session_init(session_id)
        lines = [
            init,
            json.dumps(
                {
                    "uuid": "u1",
                    "role": "user",
                    "content": "x" * ABS_SIZE_THRESHOLD,
                }
            ),
            json.dumps({"uuid": "u2", "role": "assistant", "content": "hello"}),
            _msg_json("u3", "u2", "user", "resumed"),
            _msg_json("u4", "u3", "assistant", "continued"),
            _snapshot_line(),
        ]
        transcript = self._write_transcript(session_id, lines)
        self.assertGreater(transcript.stat().st_size, ABS_SIZE_THRESHOLD)

        from clawcodex_ext.agent.session import Session

        loaded = Session.load(session_id)
        self.assertIsNotNone(loaded)
        uuids = [getattr(m, "uuid", None) for m in loaded.conversation.messages]
        self.assertEqual(uuids, ["u1", "u2", "u3", "u4"])

    def test_chain_filter_disabled_returns_full_transcript(self):
        """Opt-out: chain_filter=False returns the full transcript including dead branches."""
        self._patch_sessions_dir()
        session_id = "optout-test-1"
        init = _make_session_init(session_id)
        chat_lines = [
            _msg_json("u1", None, "user", "hi"),
            _msg_json("u2", "u1", "assistant", "hello"),
            _msg_json("u3", "u2", "user", "q1"),
            _msg_json("u4", "u3", "assistant", "a1"),
            _msg_json("u5", "u2", "user", "after rewind"),
        ]
        chat_lines = chat_lines + _pad_lines(60)
        self._write_transcript(session_id, [init] + chat_lines + [_snapshot_line()])

        from clawcodex_ext.agent.session import _load_from_enhanced_transcript

        path = Path(self.tmpdir) / session_id / "transcript.jsonl"
        loaded = _load_from_enhanced_transcript(session_id, path, chain_filter=False)
        self.assertIsNotNone(loaded)
        uuids = [getattr(m, "uuid", None) for m in loaded.conversation.messages]
        # All chat messages (including dead branches) are returned
        self.assertIn("u1", uuids)
        self.assertIn("u3", uuids)
        self.assertIn("u4", uuids)
        self.assertIn("u5", uuids)


# --- Local additions (not upstream) — keep separated for re-sync ---
from unittest import mock  # noqa: E402
from extensions.agent.session_persist import _write_session_init_line  # noqa: E402


class TestSessionInitAtomicWrite(unittest.TestCase):
    """Atomic-rewrite hardening for ``_write_session_init_line`` (review fix).

    Directly exercises ``extensions/agent/session_persist`` so the
    rewrite path is covered: a failed ``os.replace`` must leave the
    legacy transcript intact and no ``*.tmp`` residue behind.
    """

    def setUp(self):
        import tempfile

        from clawcodex_ext.services.session_storage import SessionStorage

        self.tmpdir = tempfile.mkdtemp(prefix="f103-atomic-")
        self.storage = SessionStorage(
            session_id="atomic-test-1",
            sessions_dir=Path(self.tmpdir),
        )
        self.session = mock.Mock(
            session_id="atomic-test-1",
            provider="anthropic",
            model="test-model",
        )
        self.transcript = self.storage.session_dir / "transcript.jsonl"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_legacy_lines(self) -> str:
        """Write a legacy transcript (no session_init) and return its content."""
        self.storage.session_dir.mkdir(parents=True, exist_ok=True)
        legacy = [
            _msg_json("u1", None, "user", "hi"),
            _msg_json("u2", "u1", "assistant", "hello"),
        ]
        content = "\n".join(legacy) + "\n"
        self.transcript.write_text(content, encoding="utf-8")
        return content

    def test_rewrite_prepends_session_init_and_preserves_legacy_lines(self):
        legacy = self._write_legacy_lines()

        _write_session_init_line(self.storage, self.session)

        lines = self.transcript.read_text(encoding="utf-8").splitlines()
        init = json.loads(lines[0])
        self.assertEqual(init["type"], "session_init")
        self.assertEqual(init["session_id"], "atomic-test-1")
        self.assertEqual(init["provider"], "anthropic")
        self.assertEqual(init["model"], "test-model")
        self.assertEqual("\n".join(lines[1:]) + "\n", legacy)
        self.assertEqual(list(self.storage.session_dir.glob("*.tmp")), [])

    def test_failed_replace_keeps_legacy_transcript_intact(self):
        legacy = self._write_legacy_lines()

        with mock.patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                _write_session_init_line(self.storage, self.session)

        self.assertEqual(self.transcript.read_text(encoding="utf-8"), legacy)
        self.assertEqual(list(self.storage.session_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
