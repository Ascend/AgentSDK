"""Media provider registry — decoupled from the chat-provider registry.

Media providers (image generation, video generation, etc.) are
fundamentally different from chat providers and use a separate
registration system.

Usage::

    from clawcodex_ext.providers.media.registry import media_registry

    # Register an image provider
    media_registry.register_image("agnes", AgnesImageProvider)

    # Look up and instantiate
    provider_cls = media_registry.get_image_provider("agnes")
    provider = provider_cls(api_key="...", base_url="...")

    # List all registered image providers
    for name in media_registry.list_image_providers():
        print(name)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clawcodex_ext.providers.media.base import ImageProvider, VideoProvider


class MediaProviderRegistry:
    """Registry for image/video generation provider classes.

    This is a **singleton-style** global registry (same pattern as
    ``src.providers._EXTRA_PROVIDER_CLASSES``).  Use the module-level
    ``media_registry`` instance.

    Image and video providers are stored separately so that a single
    backend (e.g. Agnes) can register both an image provider and a
    video provider under the same name.
    """

    def __init__(self) -> None:
        #: Maps provider name → ImageProvider subclass or zero-arg callable.
        self._image: dict[str, type[ImageProvider] | callable] = {}
        #: Maps provider name → VideoProvider subclass or zero-arg callable.
        self._video: dict[str, type[VideoProvider] | callable] = {}

    # ------------------------------------------------------------------
    # Image provider registration
    # ------------------------------------------------------------------

    def register_image(
        self,
        name: str,
        provider_cls: type[ImageProvider] | callable,
    ) -> None:
        """Register an image provider class.

        Args:
            name: Stable lowercase identifier (e.g. ``"agnes"``).
            provider_cls: A subclass of :class:`ImageProvider` **or**
                a zero-arg callable that returns one (lazy import).

        Idempotent: the first registration wins.
        """
        if name not in self._image:
            self._image[name] = provider_cls

    def get_image_provider(self, name: str) -> type[ImageProvider] | None:
        """Look up an image provider class by name.

        Returns ``None`` when the name is unknown.
        """
        entry = self._image.get(name)
        if entry is None:
            return None
        if callable(entry) and not isinstance(entry, type):
            entry = entry()
            self._image[name] = entry
        return entry  # type: ignore[return-value]

    def list_image_providers(self) -> list[str]:
        """Return a sorted list of registered image provider names."""
        return sorted(self._image.keys())

    # ------------------------------------------------------------------
    # Video provider registration
    # ------------------------------------------------------------------

    def register_video(
        self,
        name: str,
        provider_cls: type[VideoProvider] | callable,
    ) -> None:
        """Register a video provider class.

        Args:
            name: Stable lowercase identifier (e.g. ``"agnes"``).
            provider_cls: A subclass of :class:`VideoProvider` **or**
                a zero-arg callable that returns one (lazy import).

        Idempotent: the first registration wins.
        """
        if name not in self._video:
            self._video[name] = provider_cls

    def get_video_provider(self, name: str) -> type[VideoProvider] | None:
        """Look up a video provider class by name.

        Returns ``None`` when the name is unknown.
        """
        entry = self._video.get(name)
        if entry is None:
            return None
        if callable(entry) and not isinstance(entry, type):
            entry = entry()
            self._video[name] = entry
        return entry  # type: ignore[return-value]

    def list_video_providers(self) -> list[str]:
        """Return a sorted list of registered video provider names."""
        return sorted(self._video.keys())

    # ------------------------------------------------------------------
    # Convenience: build an instance from config
    # ------------------------------------------------------------------

    def build_image_provider(
        self,
        name: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> ImageProvider | None:
        """Look up and instantiate an image provider.

        Returns ``None`` when the name is unknown.
        """
        cls = self.get_image_provider(name)
        if cls is None:
            return None
        return cls(api_key=api_key, base_url=base_url, **kwargs)

    def build_video_provider(
        self,
        name: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> VideoProvider | None:
        """Look up and instantiate a video provider.

        Returns ``None`` when the name is unknown.
        """
        cls = self.get_video_provider(name)
        if cls is None:
            return None
        return cls(api_key=api_key, base_url=base_url, **kwargs)


#: Module-level singleton (same pattern as ``_EXTRA_PROVIDER_CLASSES``).
media_registry = MediaProviderRegistry()


__all__ = [
    "MediaProviderRegistry",
    "media_registry",
]
