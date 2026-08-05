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

"""Gateway configuration: load/save ``channels.yaml`` atomically.

The YAML file is both the hand-editable file-state base and the CLI
wizard's persistence target — the wizard reads/writes the same file.
Saves are atomic (tmp file + ``os.replace``) under a single-writer
``fcntl`` lock so concurrent processes cannot interleave writes.

Channel entries are stored as :class:`ChannelConfig` objects; WeChat
platform-specific fields (``base_url``, ``account_id``, ``allowed_users``,
…) live in ``ChannelConfig.extra`` so the existing ``to_dict``/``from_dict``
round-trip is reused unchanged.
"""

from __future__ import annotations

# pylint: disable=no-name-in-module  # Split migration branches provide these modules.

import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from clawcodex_ext.utils.file_lock import exclusive_file_lock

from clawcodex_ext.services.channels.models import ChannelConfig, ChannelType

from .command_allowlist import (
    CommandAllowlistConfig,
    DEFAULT_ORCHESTRATOR_COMMAND_ALLOWLIST,
    DEFAULT_REPL_COMMAND_ALLOWLIST,
)

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = "~/.clawcodex/gateway"
DEFAULT_CHANNELS_YAML = "~/.clawcodex/gateway/channels.yaml"
DEFAULT_PUSH_TIMEOUT_SECONDS = 5.0
# Pre-rename state dir (<= 2026-06). Kept so :func:`migrate_legacy_state_dir`
# can move an existing install forward the first time the new path is used.
LEGACY_STATE_DIR = "~/.clawcodex/im-gateway"


def migrate_legacy_state_dir(target: str | Path | None = None) -> Path:
    """One-way migration ``~/.clawcodex/im-gateway`` → ``~/.clawcodex/gateway``.

    Idempotent and safe to call from any entry point that resolves the state
    directory. If ``target`` already exists, it is returned unchanged. If the
    legacy directory exists, it is moved to ``target`` (``Path.rename``, with a
    ``shutil.copytree`` + ``rmtree`` fallback for cross-filesystem renames or
    platforms where ``rename`` refuses). If neither exists, ``target`` is
    returned without being created — the caller owns ``mkdir``.

    Only the *default* location is migrated; an explicit ``target`` override is
    honored as-is. Returns the resolved target :class:`~pathlib.Path`.
    """
    new = Path(target or DEFAULT_STATE_DIR).expanduser()
    legacy = Path(LEGACY_STATE_DIR).expanduser()
    new.parent.mkdir(parents=True, exist_ok=True)
    migration_lock = new.parent / ".gateway-state-migration.lock"
    with exclusive_file_lock(migration_lock):
        if new.exists() or not legacy.exists():
            return new
        try:
            legacy.rename(new)
        except OSError:
            # Copy to a sibling staging directory and publish it atomically.
            # A failed copy is never mistaken for a completed migration.
            stage = Path(tempfile.mkdtemp(prefix=f".{new.name}.migrate-", dir=new.parent))
            stage.rmdir()
            try:
                shutil.copytree(legacy, stage)
                os.replace(stage, new)
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                if new.exists() and not legacy.exists():
                    return new
                raise
            shutil.rmtree(legacy, ignore_errors=True)
        logger.info("migrated gateway state dir %s -> %s", legacy, new)
    return new


@dataclass
class ReliabilityConfig:
    outbox_max_attempts: int = 5
    retry_base_seconds: float = 2.0
    retry_max_seconds: float = 60.0
    inbound_dedupe_ttl_seconds: int = 600
    storm_window_seconds: int = 60
    secret_encryption_env: str = "CLAWCODEX_IM_SECRET"
    markdown_fallback: bool = True
    long_message_threshold_chunks: int = 4
    deferred_outbox_limit: int = 500
    # Persistence retention for bounded append-style files.
    retention_enabled: bool = True
    retention_cron_interval_seconds: int = 24 * 3600
    retention_processed_inbound_ttl_seconds: int = 7 * 86400
    retention_processed_inbound_max_entries: int = 10000
    retention_outbox_ttl_seconds: int = 30 * 86400
    retention_outbox_max_entries: int = 50000
    retention_unsupported_inbound_ttl_seconds: int = 7 * 86400
    retention_unsupported_inbound_max_entries: int = 10000
    # Append-time rotation for audit/dead-letter logs.
    dead_letter_max_bytes: int = 10 * 1024 * 1024  # 10 MiB
    dead_letter_backup_count: int = 3
    audit_max_bytes: int = 10 * 1024 * 1024  # 10 MiB
    audit_backup_count: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "outbox_max_attempts": self.outbox_max_attempts,
            "retry_base_seconds": self.retry_base_seconds,
            "retry_max_seconds": self.retry_max_seconds,
            "inbound_dedupe_ttl_seconds": self.inbound_dedupe_ttl_seconds,
            "storm_window_seconds": self.storm_window_seconds,
            "secret_encryption_env": self.secret_encryption_env,
            "markdown_fallback": self.markdown_fallback,
            "long_message_threshold_chunks": self.long_message_threshold_chunks,
            "deferred_outbox_limit": self.deferred_outbox_limit,
            "retention_enabled": self.retention_enabled,
            "retention_cron_interval_seconds": self.retention_cron_interval_seconds,
            "retention_processed_inbound_ttl_seconds": self.retention_processed_inbound_ttl_seconds,
            "retention_processed_inbound_max_entries": self.retention_processed_inbound_max_entries,
            "retention_outbox_ttl_seconds": self.retention_outbox_ttl_seconds,
            "retention_outbox_max_entries": self.retention_outbox_max_entries,
            "retention_unsupported_inbound_ttl_seconds": self.retention_unsupported_inbound_ttl_seconds,
            "retention_unsupported_inbound_max_entries": self.retention_unsupported_inbound_max_entries,
            "dead_letter_max_bytes": self.dead_letter_max_bytes,
            "dead_letter_backup_count": self.dead_letter_backup_count,
            "audit_max_bytes": self.audit_max_bytes,
            "audit_backup_count": self.audit_backup_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ReliabilityConfig:
        if not data:
            return cls()
        known = {
            k: v
            for k, v in data.items()
            if k
            in {
                "outbox_max_attempts",
                "retry_base_seconds",
                "retry_max_seconds",
                "inbound_dedupe_ttl_seconds",
                "storm_window_seconds",
                "secret_encryption_env",
                "markdown_fallback",
                "long_message_threshold_chunks",
                "deferred_outbox_limit",
                "retention_enabled",
                "retention_cron_interval_seconds",
                "retention_processed_inbound_ttl_seconds",
                "retention_processed_inbound_max_entries",
                "retention_outbox_ttl_seconds",
                "retention_outbox_max_entries",
                "retention_unsupported_inbound_ttl_seconds",
                "retention_unsupported_inbound_max_entries",
                "dead_letter_max_bytes",
                "dead_letter_backup_count",
                "audit_max_bytes",
                "audit_backup_count",
            }
        }
        return cls(**known)


@dataclass
class GatewayConfig:
    enabled: bool = True
    default_targets: list[str] = field(default_factory=list)
    state_dir: str = DEFAULT_STATE_DIR
    storage_backend: str = "files"
    push_timeout_seconds: float = DEFAULT_PUSH_TIMEOUT_SECONDS
    command_allowlists: CommandAllowlistConfig = field(default_factory=CommandAllowlistConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    channels: list[ChannelConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.push_timeout_seconds = float(self.push_timeout_seconds)
        if not math.isfinite(self.push_timeout_seconds) or self.push_timeout_seconds <= 0:
            raise ValueError("push_timeout_seconds must be a finite value greater than zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "default_targets": _normalize_default_targets(self.default_targets),
            "state_dir": self.state_dir,
            "storage_backend": self.storage_backend,
            "push_timeout_seconds": self.push_timeout_seconds,
            "command_allowlists": self.command_allowlists.to_dict(),
            "reliability": self.reliability.to_dict(),
            "channels": [c.to_dict() for c in _unique_channels_by_type(self.channels)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GatewayConfig:
        if not data:
            return cls()
        cfg = cls(
            enabled=bool(data.get("enabled", True)),
            default_targets=list(data.get("default_targets") or []),
            state_dir=str(data.get("state_dir", DEFAULT_STATE_DIR)),
            storage_backend=str(data.get("storage_backend", "files")),
            push_timeout_seconds=float(data.get("push_timeout_seconds", DEFAULT_PUSH_TIMEOUT_SECONDS)),
            command_allowlists=CommandAllowlistConfig.from_dict(data.get("command_allowlists")),
            reliability=ReliabilityConfig.from_dict(data.get("reliability")),
        )
        channels_raw = data.get("channels") or []
        for entry in channels_raw:
            if not isinstance(entry, dict):
                continue
            cfg.replace_channel(ChannelConfig.from_dict(entry))
        cfg.default_targets = _normalize_default_targets(cfg.default_targets)
        return cfg

    def get_channel(self, name: str) -> ChannelConfig | None:
        for c in self.channels:
            if c.name == name:
                return c
        return None

    def get_channel_by_type(self, channel_type: ChannelType | str) -> ChannelConfig | None:
        key = channel_type.value if isinstance(channel_type, ChannelType) else str(channel_type).lower()
        for c in self.channels:
            if c.type.value == key:
                return c
        return None

    def replace_channel(self, channel: ChannelConfig) -> None:
        """Replace an existing channel entry by type, or append.

        IM Message Gateway v1 intentionally keeps one configured channel per
        channel type. Names remain useful for status/restart display, but they
        are not a second axis of multiplicity.
        """
        channel = _normalize_channel(channel)
        self.default_targets = _normalize_default_targets(self.default_targets)
        replaced = False
        next_channels: list[ChannelConfig] = []
        for c in self.channels:
            c = _normalize_channel(c)
            if c.type == channel.type:
                if not replaced:
                    if c.name != channel.name:
                        self.default_targets = [
                            channel.name if target == c.name else target for target in self.default_targets
                        ]
                    next_channels.append(channel)
                    replaced = True
                continue
            next_channels.append(c)
        if not replaced:
            next_channels.append(channel)
        self.channels = next_channels

    def remove_channel(self, name: str) -> bool:
        before = len(self.channels)
        self.channels = [c for c in self.channels if c.name != name]
        return len(self.channels) < before


def _unique_channels_by_type(channels: list[ChannelConfig]) -> list[ChannelConfig]:
    cfg = GatewayConfig()
    for channel in channels:
        cfg.replace_channel(channel)
    return cfg.channels


def _normalize_channel(channel: ChannelConfig) -> ChannelConfig:
    if channel.type is ChannelType.WECHAT and channel.name != "wechat":
        return replace(channel, name="wechat")
    return channel


def _normalize_default_targets(default_targets: list[str]) -> list[str]:
    return ["wechat" if target == "wechat-main" else target for target in default_targets]


def _file_lock(lock_path: Path):
    """Return a cross-platform single-writer lock context manager."""
    return exclusive_file_lock(lock_path)


def load_config(path: str | Path | None = None) -> GatewayConfig:
    """Load :class:`GatewayConfig` from ``path`` (default ``channels.yaml``)."""
    if path is None:
        # Move a pre-rename ~/.clawcodex/im-gateway install forward the first
        # time the default path is resolved, so existing channels.yaml survives.
        migrate_legacy_state_dir()
    p = Path(path or DEFAULT_CHANNELS_YAML).expanduser()
    if not p.exists():
        logger.warning("gateway config not found at %s; using defaults", p)
        return GatewayConfig()
    with _file_lock(_lock_path(p)):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        logger.error("%s: top-level YAML is not a mapping", p)
        raise ValueError(f"{p}: expected a YAML mapping at the top level")
    cfg = GatewayConfig.from_dict(data)
    logger.info("gateway config loaded: %s (%d channel(s))", p, len(cfg.channels))
    return cfg


def save_config(config: GatewayConfig, path: str | Path | None = None) -> Path:
    """Atomically save ``config`` to ``path`` under a single-writer lock."""
    if path is None:
        migrate_legacy_state_dir()
    p = Path(path or DEFAULT_CHANNELS_YAML).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False)
    with _file_lock(_lock_path(p)):
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, p)
    logger.info("gateway config saved: %s", p)
    return p


def _lock_path(yaml_path: Path) -> Path:
    return yaml_path.with_suffix(yaml_path.suffix + ".lock")


__all__ = [
    "CommandAllowlistConfig",
    "DEFAULT_CHANNELS_YAML",
    "DEFAULT_PUSH_TIMEOUT_SECONDS",
    "DEFAULT_ORCHESTRATOR_COMMAND_ALLOWLIST",
    "DEFAULT_REPL_COMMAND_ALLOWLIST",
    "DEFAULT_STATE_DIR",
    "LEGACY_STATE_DIR",
    "GatewayConfig",
    "ReliabilityConfig",
    "load_config",
    "migrate_legacy_state_dir",
    "save_config",
]
