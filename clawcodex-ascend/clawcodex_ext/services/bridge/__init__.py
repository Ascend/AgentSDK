"""Bridge SDK stub (NOT the CCR bridge; see ``src/bridge/`` for that work).

The package name collides with ``src/bridge/`` (the CCR bridge
implementation tracked in ``my-docs/ch16-remote-refactoring-plan.md``),
so this stub provides its own minimal ``BridgeSession`` / ``BridgeAuth``
/ ``BridgeTransport`` classes for ``tests/bridge/test_bridge.py`` and
similar unit tests. New code targeting the CCR bridge should import
from ``src.bridge`` directly.
"""

from __future__ import annotations

from clawcodex_ext.services.bridge.auth import BridgeAuth, BridgeToken
from clawcodex_ext.services.bridge.session import (
    BridgeSession,
    BridgeSessionConfig,
    BridgeSessionState,
)
from clawcodex_ext.services.bridge.transport import BridgeTransport, WebSocketTransport

__all__ = [
    "BridgeAuth",
    "BridgeSession",
    "BridgeSessionConfig",
    "BridgeSessionState",
    "BridgeToken",
    "BridgeTransport",
    "WebSocketTransport",
]
