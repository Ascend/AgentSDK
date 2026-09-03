"""Backward-compatible import path for SDK dependency resolution."""

# Public names are intentionally re-exported from the relocated module.
# pylint: disable=wildcard-import,unused-wildcard-import
from extensions.sop_converter.core.sdk_dependency_resolver import *  # noqa: F401, F403
