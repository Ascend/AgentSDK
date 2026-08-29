#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# pylint: disable=relative-beyond-top-level
"""Phase 4 notification integration for SR-5.1.

Pushes a digest summary to user-configured notification channels via the
existing Channels subsystem (``src.services.channels``).

Why ``Channels``:
* The Channels stack already abstracts Feishu / Slack / Discord / WeChat /
  MCP-push transports with a uniform ``ChannelManager.broadcast`` API.
* The notifications are best-effort: a single failing channel never
  blocks the others, and the radar never crashes when no channels are
  configured.

Configuration:
* ``~/.clawcodex/community-radar/notify.yaml`` — list of named channels.
  Schema:

  .. code-block:: yaml

      channels:
        - name: feishu-team
          type: feishu
          webhook_url: https://open.feishu.cn/...
        - name: slack-dev
          type: slack
          webhook_url: https://hooks.slack.com/...

  When no config file exists the notifier is a no-op so installs that
  never opted into notifications never try to reach the network.

* ``RadarConfig.notify`` (default ``True`` since Phase 3) controls
  whether the pipeline calls the notifier at all. The notifier also
  short-circuits when the config list is empty.

Payload:
* Title: a localized digest title (see ``build_digest_message``) built
  from the period label (weekly / monthly / full report).
* Body: digest summary + top-5 trending features (title + score +
  category). Keeps the message under the 30 000-character cap so
  the broadcast path never truncates.
* Level: ``SUCCESS`` when ``len(trending) > 0``, else ``INFO``.
* Metadata: digest stats + the markdown path so users can jump to the
  full report.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RadarConfig
from .models import CommunityDigest
from .reporter import DigestWriteResult

_log = logging.getLogger(__name__)


NOTIFY_CONFIG_RELATIVE_PATH = Path(".clawcodex") / "community-radar" / "notify.yaml"


# ---------------------------------------------------------------------------
# Lazy Channels import — same pattern as other clawcodex_ext consumers.
# ---------------------------------------------------------------------------


def _load_channels_module() -> Any:
    """Return the channels package or None on ImportError."""
    try:
        from src.services.channels import base as _base  # type: ignore
        from src.services.channels import models as _models  # type: ignore

        return {"base": _base, "models": _models}
    except Exception as exc:  # noqa: BLE001
        _log.debug("Channels unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


@dataclass
class NotifyConfig:
    """A list of channel descriptors loaded from disk."""

    channels: list[dict[str, Any]] = field(default_factory=list)


def _load_notify_config(path: Path | None = None) -> NotifyConfig:
    """Read ``channels`` from a YAML file; falls back to empty list on errors."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(path)
    explicit = os.environ.get("CLAWCODEX_NOTIFY_CONFIG")
    if explicit:
        candidates.append(Path(explicit))
    base = os.environ.get("CLAWCODEX_HOME")
    if base:
        candidates.append(Path(base) / "community-radar" / "notify.yaml")
    else:
        candidates.append(Path.home() / ".clawcodex" / "community-radar" / "notify.yaml")

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError as exc:
            _log.debug("notify config unreadable %s: %s", candidate, exc)
            continue
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text) or {}
        except Exception:  # noqa: BLE001
            try:
                import json

                data = json.loads(text) or {}
            except Exception:  # noqa: BLE001, S112
                continue
        if not isinstance(data, dict):
            continue
        channels = data.get("channels") or []
        if isinstance(channels, list):
            return NotifyConfig(channels=[c for c in channels if isinstance(c, dict)])
    return NotifyConfig()


class DigestNotifier:
    """Broadcast a digest summary via Channels.

    ``manager_factory`` is a test seam; production code passes ``None``
    and the notifier builds a fresh :class:`ChannelManager`. Tests can
    pass a factory returning a fake manager (anything with an async
    ``broadcast`` method).
    """

    def __init__(
        self,
        config: RadarConfig | None = None,
        *,
        notify_config_path: Path | None = None,
        manager_factory: Any | None = None,
    ) -> None:
        self.config = config or RadarConfig()
        self._notify_config_path = notify_config_path
        self._manager_factory = manager_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def broadcast(
        self,
        digest: CommunityDigest,
        write_result: DigestWriteResult | None = None,
    ) -> dict[str, bool]:
        """Send a digest summary to every configured channel.

        Returns a mapping of channel name → success bool. An empty
        dict means "nothing to send" (no channels configured or the
        notifier is disabled). Never raises.
        """
        if not self.config.notify:
            return {}

        notify_config = _load_notify_config(self._notify_config_path)
        if not notify_config.channels:
            _log.debug("no notify channels configured; skipping broadcast")
            return {}

        channels_module = _load_channels_module()
        if channels_module is None:
            _log.warning("Channels unavailable; install or fix Channels to enable notifications.")
            return {"_error": False}  # type: ignore[dict-item]

        try:
            return self._broadcast_async(digest, write_result, notify_config, channels_module)
        except Exception as exc:  # noqa: BLE001 — never crash the scan
            _log.warning("notification broadcast raised: %s", exc)
            return {"_error": False}  # type: ignore[dict-item]

    # ------------------------------------------------------------------
    # Async dispatch
    # ------------------------------------------------------------------

    def _broadcast_async(
        self,
        digest: CommunityDigest,
        write_result: DigestWriteResult | None,
        notify_config: NotifyConfig,
        channels_module: Any,
    ) -> dict[str, bool]:
        message = build_digest_message(digest, write_result)
        manager = self._build_manager(notify_config, channels_module)
        if manager is None:
            return {}

        async def _runner() -> dict[str, bool]:
            return await manager.broadcast(message)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an event loop (e.g. the TUI). Schedule
                # the broadcast as a task and return immediately so
                # the pipeline is not blocked by channel latency.
                task = loop.create_task(_runner())

                def _log_failure(fut: Any) -> None:
                    if fut.cancelled():
                        return
                    exc = fut.exception()
                    if exc is not None:
                        _log.warning("scheduled notification broadcast failed: %s", exc)

                task.add_done_callback(_log_failure)
                return {"_scheduled": True}  # type: ignore[dict-item]
            return loop.run_until_complete(_runner())
        except RuntimeError:
            # No loop in this thread (CLI / Cron thread).
            return asyncio.run(_runner())

    def _build_manager(
        self,
        notify_config: NotifyConfig,
        channels_module: Any,
    ) -> Any | None:
        if self._manager_factory is not None:
            try:
                return self._manager_factory()
            except Exception as exc:  # noqa: BLE001
                _log.warning("manager_factory raised: %s", exc)
                return None
        try:
            manager = channels_module["base"].ChannelManager()
        except Exception as exc:  # noqa: BLE001
            _log.warning("ChannelManager() failed: %s", exc)
            return None
        for entry in notify_config.channels:
            channel = self._build_channel(entry, channels_module)
            if channel is not None:
                try:
                    manager.register(channel)
                except Exception as exc:  # noqa: BLE001
                    _log.debug("register channel failed: %s", exc)
        return manager

    def _build_channel(self, entry: dict[str, Any], channels_module: Any) -> Any | None:
        try:
            models = channels_module["models"]
            config = models.ChannelConfig.from_dict(entry)
            channel_type = config.type.value
            submodule_name = {
                "feishu": "feishu",
                "slack": "slack",
                "discord": "discord",
                "wechat": "wechat",
                "mcp_push": "mcp_push",
            }.get(channel_type)
            if submodule_name is None:
                _log.warning("unknown channel type %s; skipping", channel_type)
                return None
            import importlib

            submodule = importlib.import_module(f"src.services.channels.{submodule_name}")
            for attr in (
                "FeishuChannel",
                "SlackChannel",
                "DiscordChannel",
                "WeChatChannel",
                "McpPushChannel",
                "BaseChannel",
            ):
                channel_cls = getattr(submodule, attr, None)
                if channel_cls is not None:
                    try:
                        return channel_cls(config)
                    except TypeError:
                        # Some channel classes take only ``config``.
                        continue
            return None
        except Exception as exc:  # noqa: BLE001
            _log.debug("build channel failed for %s: %s", entry.get("name"), exc)
            return None


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def build_digest_message(
    digest: CommunityDigest,
    write_result: DigestWriteResult | None = None,
    *,
    top_n: int = 5,
    issue_sync_created: list[dict[str, Any]] | None = None,
    repo_url: str = "",
) -> Any:
    """Build the ``ChannelMessage`` broadcast by :class:`DigestNotifier`.

    Imported lazily so importing this module never requires Channels.
    """
    channels_module = _load_channels_module()
    if channels_module is None:
        raise RuntimeError("Channels module unavailable; cannot build message.")
    models = channels_module["models"]

    period_label = {"weekly": "周报", "monthly": "月报", "full": "总报"}.get(digest.period, digest.period)
    title = f"ClawCodex 社区动态 {period_label}"
    lines: list[str] = []
    lines.append(f"**{digest.summary.strip()}**")
    lines.append("")
    lines.append(
        f"覆盖 {len(digest.sources_used)} 个项目 · "
        f"{digest.stats.total_versions} 个版本 · "
        f"{digest.stats.total_features} 条候选特性"
    )
    if digest.trending:
        lines.append("")
        lines.append(f"**Top-{min(top_n, len(digest.trending))} 高分候选：**")
        for item in digest.trending[:top_n]:
            related = " + ".join([item.record.source, *item.record.related_projects])
            lines.append(f"- {item.record.title} · {related} · {item.score.overall:.1f} · {item.record.category.value}")
    if digest.breaking_changes:
        lines.append("")
        lines.append(f"**破坏性变更 ({len(digest.breaking_changes)} 条)**：")
        for record in digest.breaking_changes[:3]:
            lines.append(f"- {record.source}: {record.title}")

    if issue_sync_created:
        lines.append("")
        created_n = len(issue_sync_created)
        lines.append(f"📮 本周已自动创建 {created_n} 个社区雷达 issue")
        if repo_url:
            lines.append(f"  目标仓库: {repo_url}")
        for item in issue_sync_created:
            iss_num = item.get("issue_number", "?")
            iss_title = item.get("feature_title", "")[:50]
            iss_url = item.get("issue_url", "")
            lines.append(f"  - #{iss_num} {iss_title} → {iss_url}")

    text = "\n".join(lines).strip()
    if len(text) > 8_000:  # leave headroom under the 30 000-char hard cap
        text = text[:7997] + "…"

    level = models.MessageLevel.SUCCESS if digest.trending else models.MessageLevel.INFO

    metadata: dict[str, Any] = {
        "period": digest.period,
        "generated_at": digest.generated_at,
        "stats": digest.stats.to_dict(),
        "sources": list(digest.sources_used),
    }
    if write_result is not None:
        metadata["digest_markdown"] = str(write_result.markdown_path)
        metadata["digest_json"] = str(write_result.json_path)
        if write_result.proposals_path is not None:
            metadata["proposals_json"] = str(write_result.proposals_path)

    return models.ChannelMessage(
        text=text,
        level=level,
        title=title,
        markdown=True,
        metadata=metadata,
    )


__all__ = [
    "NOTIFY_CONFIG_RELATIVE_PATH",
    "DigestNotifier",
    "NotifyConfig",
    "build_digest_message",
]
