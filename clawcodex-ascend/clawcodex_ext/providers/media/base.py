#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Abstract base classes for media generation providers.

Media providers are independent of the chat provider hierarchy and expose
task creation, polling, and result retrieval APIs. They register through
``MediaProviderRegistry`` rather than the chat-provider factory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result and status dataclasses shared by all media providers
# ---------------------------------------------------------------------------


@dataclass
class ImageResult:
    """Result of an image generation request."""

    #: Public URL of the generated image.
    url: str
    #: Optional revised prompt (some providers rewrite the prompt for
    #: safety/content-policy reasons).
    revised_prompt: str | None = None
    #: Optional base64-encoded image data (alternative to ``url``).
    b64_json: str | None = None
    #: Opaque extra fields from the provider response (e.g. seed, model).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoTask:
    """Descriptor returned when a video generation task is created."""

    #: Provider-assigned task ID, used for polling and result retrieval.
    task_id: str
    #: Initial status (typically ``"queued"`` or ``"processing"``).
    status: str
    #: Optional video ID (some providers return both task_id and video_id).
    video_id: str | None = None
    #: Opaque extra fields from the provider response.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoStatus:
    """Status of an in-progress video generation task."""

    #: Current status string (``"queued"``, ``"processing"``,
    #: ``"completed"``, ``"failed"``).
    status: str
    #: Optional progress indicator from 0.0 to 1.0.
    progress: float | None = None
    #: Error message when status is ``"failed"`.
    error: str | None = None
    #: Opaque extra fields from the provider response.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoResult:
    """Final result of a completed video generation task."""

    #: Public URL of the generated video file (typically MP4).
    video_url: str
    #: Approximate duration in seconds, if reported by the provider.
    duration_seconds: float | None = None
    #: Opaque extra fields from the provider response.
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class MediaProvider(ABC):
    """Abstract base for **all** media generation providers.

    Media providers are completely independent of the chat-oriented
    :class:`~clawcodex_ext.providers.base.BaseProvider` hierarchy.
    They are registered via :class:`MediaProviderRegistry` rather than
    ``register_provider()``.

    Every media provider must return a stable lowercase identifier
    from :meth:`get_provider_name` (e.g. ``"agnes"``, ``"openai"``).
    """

    def __init__(self, api_key: str, base_url: str | None = None, **kwargs: Any):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a stable lowercase provider identifier.

        Used as the key in :class:`MediaProviderRegistry` lookups.
        Examples: ``"agnes"``, ``"openai"``, ``"runway"``.
        """


class ImageProvider(MediaProvider):
    """Abstract base for **image generation** providers.

    Subclasses implement :meth:`generate_image` which accepts a text
    prompt and returns an :class:`ImageResult`.
    """

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        n: int = 1,
        **kwargs: Any,
    ) -> ImageResult:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            model: Model identifier (overrides the default).
            size: Output dimensions, e.g. ``"1024x1024"``.
            n: Number of images to generate (provider-dependent
               maximum applies).
            **kwargs: Provider-specific parameters (e.g. ``image``
                      for image-to-image, ``style``, ``quality``).

        Returns:
            An :class:`ImageResult` with the generated image URL/data.
        """


class VideoProvider(MediaProvider):
    """Abstract base for **video generation** providers.

    Video generation is inherently asynchronous.  The typical flow is:

    1. :meth:`generate_video` creates and returns a :class:`VideoTask`.
    2. :meth:`get_video_status` polls for progress.
    3. :meth:`get_video_result` retrieves the completed video URL.

    The :meth:`generate_video` *may* return a completed task
    (``status == "completed"``) if the provider is fast enough, but
    callers should always be prepared to poll.

    Subclasses may also implement :meth:`poll_until_done` as a
    convenience.
    """

    @abstractmethod
    def generate_video(
        self,
        prompt: str,
        *,
        model: str | None = None,
        width: int = 1152,
        height: int = 768,
        **kwargs: Any,
    ) -> VideoTask:
        """Create a video generation task.

        Args:
            prompt: Text description of the desired video.
            model: Model identifier (overrides the default).
            width: Output width in pixels.
            height: Output height in pixels.
            **kwargs: Provider-specific parameters (e.g. ``image``
                      for image-to-video, ``num_frames``, ``frame_rate``,
                      ``negative_prompt``).

        Returns:
            A :class:`VideoTask` with a ``task_id`` for polling.
        """

    @abstractmethod
    def get_video_status(self, task_id: str) -> VideoStatus:
        """Poll the current status of a video generation task.

        Args:
            task_id: The task ID returned by :meth:`generate_video`.

        Returns:
            A :class:`VideoStatus` describing the current state.
        """

    @abstractmethod
    def get_video_result(self, task_id: str) -> VideoResult:
        """Retrieve the result of a completed video task.

        Args:
            task_id: The task ID returned by :meth:`generate_video`.

        Returns:
            A :class:`VideoResult` with the video URL.

        Raises:
            RuntimeError: If the task has not completed yet.
        """

    def poll_until_done(
        self,
        task_id: str,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
    ) -> VideoResult:
        """Convenience: poll :meth:`get_video_status` until completion.

        Args:
            task_id: The task ID from :meth:`generate_video`.
            poll_interval: Seconds between polls.
            max_wait: Maximum total wait time in seconds.

        Returns:
            A :class:`VideoResult` with the video URL.

        Raises:
            TimeoutError: If the task does not complete within ``max_wait``.
            RuntimeError: If the task fails.
        """
        import time

        deadline = time.monotonic() + max_wait
        while True:
            status = self.get_video_status(task_id)
            if status.status == "completed":
                return self.get_video_result(task_id)
            if status.status == "failed":
                raise RuntimeError(f"Video generation failed for task {task_id}: {status.error}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Video generation timed out after {max_wait}s for task {task_id}")
            time.sleep(poll_interval)


__all__ = [
    "ImageProvider",
    "ImageResult",
    "MediaProvider",
    "VideoProvider",
    "VideoResult",
    "VideoStatus",
    "VideoTask",
]
