"""Schema v1 → v2 migration helpers.

The on-disk event format grew from v1 to v2. The breaking
change is the shape of ``fields["fingerprint"]``:

* v1: ``"abc1234567890def"`` (16-char hex string)
* v2: ``{"hash": "abc1234567890def", "version": 2, "method": "sha1-truncate"}``

v2 is the *target* for newly-written events; v1 events already on disk
are read unchanged and auto-upgraded via :func:`normalize_event` before
aggregation. The upgrade is idempotent — feeding a v2 event through
:func:`migrate_v1_to_v2` is a no-op.

Why a dict instead of a string? The string form cannot tell us which
hash algorithm produced the digest, so cross-version dedupe cannot
guarantee that two fingerprints were generated the same way. The
explicit ``version``/``method`` fields make that auditable, and the
``hash`` field stays a stable 16-char join key for the daily aggregator.
"""

from __future__ import annotations

import logging
from typing import Any

from .events import SCHEMA_VERSION, SCHEMA_VERSION_V2

# Re-export so callers can keep importing the v2 constant from
# ``migration`` (avoids breaking the aggregator's existing import
# surface).
__all__ = [
    "SCHEMA_VERSION_V2",
    "migrate_v1_to_v2",
    "normalize_event",
    "_fingerprint_dict_to_hash",
]

logger = logging.getLogger(__name__)

#: Bumped in :func:`migrate_v1_to_v2` to record the upgrade timestamp.
_FINGERPRINT_V1_METHOD: str = "legacy"
_FINGERPRINT_V1_VERSION: int = 1


def _wrap_fingerprint_v2(value: str) -> dict[str, Any]:
    """Wrap a v1 fingerprint string into the v2 dict form."""
    if not isinstance(value, str) or not value:
        # Defensive: redactor should already drop empty strings, but if
        # an event slipped through, normalize to an empty v2 dict so
        # aggregator code that expects a dict never crashes.
        return {
            "hash": "",
            "version": _FINGERPRINT_V1_VERSION,
            "method": _FINGERPRINT_V1_METHOD,
        }
    return {
        "hash": value,
        "version": _FINGERPRINT_V1_VERSION,
        "method": _FINGERPRINT_V1_METHOD,
    }


def migrate_v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a v1 on-disk event dict in-place to v2.

    Only ``fields["fingerprint"]`` is restructured; every other field
    is preserved. The function returns the same dict object it was
    given (mutated) so callers can do ``raw = migrate_v1_to_v2(raw)``
    without changing the rest of the read path.

    Events already at v2 or higher are returned untouched.
    """
    if not isinstance(raw, dict):
        return raw
    current = raw.get("schema_version", SCHEMA_VERSION)
    if isinstance(current, int) and current >= SCHEMA_VERSION_V2:
        return raw
    fields = raw.get("fields")
    if isinstance(fields, dict):
        fp = fields.get("fingerprint")
        if isinstance(fp, str):
            fields["fingerprint"] = _wrap_fingerprint_v2(fp)
        elif fp is None:
            # ``fingerprint`` may be missing on non-error events; the
            # v2 form requires the dict shape only when present.
            pass
    raw["schema_version"] = SCHEMA_VERSION_V2
    return raw


def normalize_event(raw: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a raw event dict to the right shape for the current
    schema version.

    v1 events are upgraded in-place via :func:`migrate_v1_to_v2`. v2+
    events pass through. The function never raises — a malformed dict
    is returned as-is and the aggregator's per-key guards keep the
    pipeline safe.
    """
    if not isinstance(raw, dict):
        return raw
    try:
        current = raw.get("schema_version", SCHEMA_VERSION)
    except Exception:  # noqa: BLE001
        return raw
    if isinstance(current, int) and current >= SCHEMA_VERSION_V2:
        return raw
    return migrate_v1_to_v2(raw)


def _fingerprint_dict_to_hash(fp: Any) -> str:
    """Return the 16-char join key from a v1 string or v2 dict.

    Unknown shapes collapse to ``""`` so the aggregator can group them
    under the same bucket without raising. Callers must check for the
    empty string when they need to distinguish "no fingerprint" from
    "real fingerprint".
    """
    if isinstance(fp, str):
        return fp
    if isinstance(fp, dict):
        value = fp.get("hash")
        if isinstance(value, str):
            return value
    return ""
