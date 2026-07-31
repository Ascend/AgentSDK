"""Media generation providers — image and video generation.

Architecture::

    media/
        __init__.py          ← exports
        base.py              ← MediaProvider / ImageProvider / VideoProvider ABCs
        registry.py          ← MediaProviderRegistry + media_registry singleton
        image/
            __init__.py
            agnes.py         ← AgnesImageProvider
        video/
            __init__.py
            agnes.py         ← AgnesVideoProvider

New providers are registered by importing the registry and calling
``media_registry.register_image(...)`` or
``media_registry.register_video(...)`` at module level (same pattern as
``register_provider(...)`` in ``clawcodex_ext/providers/__init__.py``).
"""

from clawcodex_ext.providers.media.base import (
    ImageProvider,
    ImageResult,
    MediaProvider,
    VideoProvider,
    VideoResult,
    VideoStatus,
    VideoTask,
)
from clawcodex_ext.providers.media.registry import MediaProviderRegistry, media_registry

__all__ = [
    # Base classes
    "ImageProvider",
    "ImageResult",
    "MediaProvider",
    "VideoProvider",
    "VideoResult",
    "VideoStatus",
    "VideoTask",
    # Registry
    "MediaProviderRegistry",
    "media_registry",
]
