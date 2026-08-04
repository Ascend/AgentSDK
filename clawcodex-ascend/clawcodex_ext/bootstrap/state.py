"""Module-identity bridge to the canonical bootstrap state singleton."""

import sys

from src.bootstrap import state as _implementation

sys.modules[__name__] = _implementation
