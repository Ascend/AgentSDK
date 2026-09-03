"""Compatibility facade for the skill-grouping runtime."""

# Public names are intentionally re-exported from the relocated module.
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import
from .runtime.skill_grouper import *  # noqa: F401, F403
from .runtime.skill_grouper import _common_ancestor_segment  # noqa: F401
