"""Compatibility facade — see :mod:`extensions.ports.transports.websocket_v1`.

P3-out-2: this facade previously routed through
``src.transports.websocket_transport``, which transitively loaded the
``src.transports`` package ``__init__`` and risked a circular import
via ``extensions.ports.transports.hybrid_v1``. Routing directly at
the ``extensions/`` module avoids the package ``__init__`` side
effects; the legacy ``src.transports.*`` path is still a thin
forwarding seam and keeps working for upstream callers.
"""

from extensions.ports.transports.websocket_v1 import *  # noqa: F401,F403  # pylint: disable=wildcard-import
