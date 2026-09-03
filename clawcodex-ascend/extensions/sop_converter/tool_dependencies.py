"""Compatibility facade for core tool dependency analysis."""

# Public names are intentionally re-exported from the relocated module.
# pylint: disable=wildcard-import,unused-wildcard-import
from .core.tool_dependencies import *  # noqa: F401, F403
from .core.tool_dependencies import _PRIMITIVE_TYPES, _is_chain_builder_producer  # noqa: F401
