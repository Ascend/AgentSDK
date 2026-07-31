"""Compatibility facade — see :mod:`extensions.ports.transports.serial_uploader`.

P3-out-2: this facade previously routed through
``src.transports.serial_batch_event_uploader``, which transitively
loaded the ``src.transports`` package ``__init__`` and triggered a
circular import via ``extensions.ports.transports.hybrid_v1``.
Routing directly at the ``extensions/`` module avoids the package
``__init__`` side effects; the legacy ``src.transports.*`` path is
still a thin forwarding seam and keeps working for upstream callers.
"""

from extensions.ports.transports.serial_uploader import *  # noqa: F401,F403  # pylint: disable=wildcard-import
