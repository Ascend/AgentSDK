"""Auto Mode Runner Integration.

This module provides the integration layer for the LLM-based auto mode
classifier. It replaces the default rule-based auto_mode_classify with
the extended version that includes LLM judgment for uncertain scenarios.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_AUTO_MODE_CLASSIFIER_INSTALLED = False
_ORIGINAL_CLASSIFY = None


def install_llm_auto_mode_classifier(
    *,
    provider: Any | None = None,
    use_llm_for_uncertain: bool = True,
    cache_ttl_seconds: float = 600.0,
) -> bool:
    """Install the LLM-enhanced auto mode classifier.

    This function replaces the default auto_mode_classify in
    src.permissions.check with the extended version that uses
    LLM classification for uncertain scenarios.

    Args:
        provider: Optional provider instance for LLM calls
        use_llm_for_uncertain: Whether to use LLM for uncertain cases
        cache_ttl_seconds: Cache TTL for classification results

    Returns:
        True if installation succeeded, False otherwise
    """
    global _AUTO_MODE_CLASSIFIER_INSTALLED

    if _AUTO_MODE_CLASSIFIER_INSTALLED:
        log.debug("LLM auto mode classifier already installed")
        return True

    try:
        import src.permissions.check as check_module

        _original_classify = check_module.auto_mode_classify

        global _ORIGINAL_CLASSIFY
        _ORIGINAL_CLASSIFY = _original_classify
        log.info("Successfully saved original auto_mode_classify reference")

        from clawcodex_ext.permissions.classifier import (
            auto_mode_classify_with_llm,
            get_cache,
        )

        log.info("Successfully imported LLM classifier components")

        cache = get_cache()
        cache.ttl_seconds = cache_ttl_seconds
        log.info("Cache configured with TTL=%s seconds", cache_ttl_seconds)

        def _llm_enhanced_auto_mode_classify(
            tool_name: str,
            tool_input: dict[str, Any],
            context: Any,
        ) -> Any:
            return auto_mode_classify_with_llm(
                tool_name,
                tool_input,
                context,
                provider=provider,
                use_llm_for_uncertain=use_llm_for_uncertain,
                _original_classify=_original_classify,
            )

        check_module.auto_mode_classify = _llm_enhanced_auto_mode_classify

        # Verify the patch worked
        if check_module.auto_mode_classify is _llm_enhanced_auto_mode_classify:
            _AUTO_MODE_CLASSIFIER_INSTALLED = True
            log.info(
                "LLM auto mode classifier installed successfully (use_llm=%s, cache_ttl=%s)",
                use_llm_for_uncertain,
                cache_ttl_seconds,
            )
            return True
        else:
            log.error("Monkey patch failed: function reference not updated")
            return False

    except Exception as e:
        import traceback

        log.warning("Failed to install LLM auto mode classifier: %s", e)
        log.warning("Traceback: %s", traceback.format_exc())
        return False


def uninstall_llm_auto_mode_classifier() -> bool:
    """Restore the original rule-based auto mode classifier.

    Returns:
        True if uninstallation succeeded, False otherwise
    """
    global _AUTO_MODE_CLASSIFIER_INSTALLED

    if not _AUTO_MODE_CLASSIFIER_INSTALLED:
        return True

    try:
        import src.permissions.check as check_module

        if _ORIGINAL_CLASSIFY is not None:
            check_module.auto_mode_classify = _ORIGINAL_CLASSIFY
        _AUTO_MODE_CLASSIFIER_INSTALLED = False

        log.info("LLM auto mode classifier uninstalled, restored original")
        return True

    except Exception as e:
        log.warning("Failed to uninstall LLM auto mode classifier: %s", e)
        return False


def is_llm_auto_mode_classifier_installed() -> bool:
    """Check if the LLM-enhanced classifier is installed."""
    return _AUTO_MODE_CLASSIFIER_INSTALLED


def verify_auto_mode_classifier() -> dict[str, Any]:
    """Return diagnostic info about the auto mode classifier status.

    Returns:
        Dict with classifier status, original function, and current function
    """
    import src.permissions.check as check_module

    current_func = getattr(check_module, "auto_mode_classify", None)
    func_name = getattr(current_func, "__name__", str(current_func))
    func_module = getattr(current_func, "__module__", "unknown")

    return {
        "installed": _AUTO_MODE_CLASSIFIER_INSTALLED,
        "current_function": func_name,
        "function_module": func_module,
        "is_llm_enhanced": "llm_enhanced" in func_name or "auto_mode_classify_with_llm" in str(current_func),
    }


__all__ = [
    "install_llm_auto_mode_classifier",
    "uninstall_llm_auto_mode_classifier",
    "is_llm_auto_mode_classifier_installed",
]
