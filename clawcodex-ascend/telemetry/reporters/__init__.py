"""Reporter package.

Public surface:

* :class:`Reporter`            — Protocol
* :class:`CompositeReporter`   — fan-out wrapper
* :class:`LocalFileReporter`   — writes ``reports/YYYY-MM-DD.md``
* :class:`DryRunReporter`      — in-memory, used by tests + CLI preview
* :class:`IssueReporter`       — opt-in GitHub/Gitee/GitCode issue upload
"""

from __future__ import annotations

from .base import CompositeReporter, Reporter
from .dry_run import DryRunReporter
from .issue import IssueReporter
from .local_file import LocalFileReporter

__all__ = [
    "CompositeReporter",
    "DryRunReporter",
    "IssueReporter",
    "LocalFileReporter",
    "Reporter",
]
