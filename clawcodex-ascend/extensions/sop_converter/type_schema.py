"""Compatibility facade for core type-schema helpers."""

# Public names are intentionally re-exported from the relocated module.
# pylint: disable=wildcard-import,unused-wildcard-import
from .core.type_schema import *  # noqa: F401, F403
from .core.type_schema import _import_resolved_type  # noqa: F401
