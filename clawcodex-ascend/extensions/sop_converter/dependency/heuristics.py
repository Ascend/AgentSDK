"""Compatibility facade preserving private dependency heuristics."""

# Public names are intentionally re-exported from the relocated module.
# pylint: disable=wildcard-import,unused-wildcard-import
from ..core.dependency.heuristics import *  # noqa: F401, F403
from ..core.dependency import heuristics as _impl

for _compat_name, _compat_value in vars(_impl).items():
    if _compat_name.startswith("_") and not _compat_name.startswith("__"):
        globals()[_compat_name] = _compat_value
globals().pop("_compat_name", None)
globals().pop("_compat_value", None)
del _impl
