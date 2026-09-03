"""Compatibility facade for core import-alias resolution."""

# Public names are intentionally re-exported from the relocated module.
# pylint: disable=wildcard-import,unused-wildcard-import
from .core.import_alias_resolver import *  # noqa: F401, F403
