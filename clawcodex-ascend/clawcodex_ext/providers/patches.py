"""Provider monkey-patches for downstream extension integration.

Patches are applied once at import time via :func:`install` (called from
``clawcodex_ext/__init__``).  Each patch is idempotent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_patches_installed = False


def install() -> None:
    """Install all provider patches.

    Safe to call multiple times — patches are applied only on the first
    invocation.
    """
    global _patches_installed
    if _patches_installed:
        return
    _patches_installed = True

    _patch_discovery_chain()
    logger.info("provider patches installed")


def _patch_discovery_chain() -> None:
    """Ensure the downstream discovery hooks are wired into the core
    :class:`ModelRegistry`.

    The :mod:`clawcodex_ext.providers` module already registers
    ``openai-codex`` discovery via :func:`register_discovery_hook` at
    import time.  This patch verifies the registration is present and
    proactively warms the registry so the first ``/model`` query does
    not pay a cold-start penalty.
    """
    try:
        from clawcodex_ext.cli.model_cmd.registry import ModelRegistry  # pylint: disable=no-name-in-module

        registry = ModelRegistry()
        # Warm the discovery cache for openai-codex so the first
        # ``available_models()`` call is responsive.
        if "openai-codex" in registry._discovery_hooks:
            registry.available_models("openai-codex")
    except Exception:
        logger.debug("provider discovery warm-up skipped (non-fatal)", exc_info=True)


__all__ = ["install"]
