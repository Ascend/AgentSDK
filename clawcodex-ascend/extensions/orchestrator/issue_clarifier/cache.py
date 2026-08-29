# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
"""Small persistent fingerprint cache for issue clarification results."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .models import ClarifyResult

if TYPE_CHECKING:
    from ..issue import Issue

logger = logging.getLogger(__name__)


def build_fingerprint(
    issue: "Issue",
    *,
    prior_replies: Iterable[str] = (),
    workspace_focuses: list[dict] | None = None,
    version: str = "issue-clarifier-v1",
) -> str:
    payload = {
        "version": version,
        "title": str(getattr(issue, "title", "") or ""),
        "description": str(getattr(issue, "description", "") or ""),
        "labels": sorted(str(label) for label in (getattr(issue, "labels", None) or [])),
        "replies": [str(reply) for reply in prior_replies if str(reply).strip()],
        "workspace_focuses": workspace_focuses or [],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ClarifierCache:
    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._records: dict[str, dict] = {}
        self._load()

    def get(self, fingerprint: str) -> ClarifyResult | None:
        if not self.enabled:
            return None
        raw = self._records.get(fingerprint)
        if not isinstance(raw, dict):
            return None
        return ClarifyResult.from_dict(raw).with_runtime_fields(
            fingerprint=fingerprint,
            cached=True,
        )

    def put(self, result: ClarifyResult) -> None:
        if not self.enabled or not result.fingerprint or result.degraded:
            return
        self._records[result.fingerprint] = result.to_dict()
        self._save()

    def _load(self) -> None:
        if not self.enabled or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._records = {str(key): value for key, value in raw.items() if isinstance(value, dict)}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Ignoring corrupted issue clarifier cache %s: %s", self.path, exc)
            self._records = {}
        except OSError as exc:
            logger.warning("Could not access issue clarifier cache %s due to a filesystem error: %s", self.path, exc)
            self._records = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._records, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning("Could not persist issue clarifier cache %s due to a filesystem error: %s", self.path, exc)
        except (TypeError, ValueError) as exc:
            logger.warning("Could not serialize issue clarifier cache %s: %s", self.path, exc)


__all__ = ["ClarifierCache", "build_fingerprint"]
