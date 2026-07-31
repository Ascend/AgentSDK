"""Agnes AI video generation provider.

Uses the **async task pattern**::

    POST /v1/videos      → { "id": "task_…", "status": "queued" }
    GET  /v1/videos/{id}  → { "status": "completed", "video_url": "…" }

This pattern is shared by many video generation backends (Runway, Pika,
Replicate, etc.) and is abstracted by :class:`VideoProvider` so that
future providers can reuse the same interface.

API reference: https://agnes-ai.com/doc/agnes-video-v20

Supported models:
    - ``agnes-video-v2.0`` — text-to-video / image-to-video / keyframes

Supported video parameters:
    - ``num_frames``: 81, 121, 161, 241, or 441 (rule: 8n+1, ≤ 441)
    - ``frame_rate``: 1–60 (default 24)
    - ``width``: 6–2048 (default 1152)
    - ``height``: 6–2048 (default 768)
"""

from __future__ import annotations

from typing import Any

import httpx

from clawcodex_ext.providers.media.base import (
    VideoProvider,
    VideoResult,
    VideoStatus,
    VideoTask,
)

_DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"


class AgnesVideoProvider(VideoProvider):
    """Video generation via Agnes AI (async task pattern).

    Args:
        api_key: Agnes AI API key.
        base_url: Base URL (defaults to
            ``https://apihub.agnes-ai.com/v1``).
        default_model: Override the default model
            (``"agnes-video-v2.0"``).
        poll_interval: Default seconds between status polls when using
            :meth:`poll_until_done` (default 10).
        max_wait: Default max wait in seconds (default 1800 = 30 min).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        default_model: str = "agnes-video-v2.0",
        poll_interval: float = 10.0,
        max_wait: float = 1800.0,
        **kwargs: Any,
    ):
        super().__init__(api_key, base_url or _DEFAULT_BASE_URL, **kwargs)
        self.default_model = default_model
        self.default_poll_interval = poll_interval
        self.default_max_wait = max_wait
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=kwargs.get("timeout", 30.0),
        )

    def get_provider_name(self) -> str:
        return "agnes"

    def generate_video(
        self,
        prompt: str,
        *,
        model: str | None = None,
        width: int = 1152,
        height: int = 768,
        **kwargs: Any,
    ) -> VideoTask:
        """Create a video generation task via ``POST /v1/videos``.

        Args:
            prompt: Text description of the desired video.
            model: Model name (default ``"agnes-video-v2.0"``).
            width: Output width in pixels (6–2048).
            height: Output height in pixels (6–2048).
            **kwargs: Additional parameters:

                - ``image`` (str): Public URL for image-to-video.
                - ``num_frames`` (int): Number of frames (81, 121,
                  161, 241, or 441; must satisfy 8n+1).
                - ``frame_rate`` (int): 1–60 (default 24).
                - ``image_a`` / ``image_b`` (str): Keyframe URLs for
                  multi-video / morph scenes.

        Returns:
            A :class:`VideoTask` with the ``task_id`` for polling.

        Raises:
            httpx.HTTPStatusError: On API error.
        """
        resolved_model = model or self.default_model

        body: dict[str, Any] = {
            "model": resolved_model,
            "prompt": prompt,
            "width": width,
            "height": height,
        }

        # Optional video parameters
        for key in ("num_frames", "frame_rate", "image", "image_a", "image_b"):
            val = kwargs.get(key)
            if val is not None:
                body[key] = val

        resp = self._client.post("/videos", json=body)
        resp.raise_for_status()
        data = resp.json()

        # Parse response — Agnes returns { "id": "task_…", "status": "queued", … }
        task_id = data.get("id") or data.get("task_id", "")
        if not task_id:
            raise RuntimeError(f"Video task creation returned no id: {data}")

        return VideoTask(
            task_id=task_id,
            status=data.get("status", "queued"),
            video_id=data.get("video_id"),
            extra={"model": resolved_model, "raw_response": data},
        )

    def get_video_status(self, task_id: str) -> VideoStatus:
        """Poll the current status via ``GET /v1/videos/{task_id}``.

        Returns:
            A :class:`VideoStatus`.

        Raises:
            httpx.HTTPStatusError: On API error.
        """
        resp = self._client.get(f"/videos/{task_id}")
        resp.raise_for_status()
        data = resp.json()

        status_str = data.get("status", "unknown")
        # Attempt to parse a progress indicator (some providers return
        # a ``progress`` field 0.0–1.0).
        progress_raw = data.get("progress")
        progress: float | None = None
        if progress_raw is not None:
            try:
                progress = float(progress_raw)
            except (ValueError, TypeError):
                pass

        error = data.get("error") or data.get("failure_reason")

        return VideoStatus(
            status=status_str,
            progress=progress,
            error=error,
            extra={"raw_response": data},
        )

    def get_video_result(self, task_id: str) -> VideoResult:
        """Retrieve the completed video result.

        Fetches the current task status and extracts the video URL.

        Raises:
            RuntimeError: If the task hasn't completed yet.
            httpx.HTTPStatusError: On API error.
        """
        resp = self._client.get(f"/videos/{task_id}")
        resp.raise_for_status()
        data = resp.json()

        status_str = data.get("status", "unknown")
        if status_str != "completed":
            raise RuntimeError(f"Task {task_id} is {status_str!r}, not 'completed'. Use get_video_status() to poll.")

        video_url = data.get("video_url") or data.get("url", "")
        if not video_url:
            raise RuntimeError(f"Task {task_id} completed but no video_url in response: {data}")

        # Attempt to derive duration from num_frames / frame_rate if present.
        duration: float | None = None
        nf = data.get("num_frames")
        fr = data.get("frame_rate")
        if nf and fr:
            try:
                duration = int(nf) / int(fr)
            except (ValueError, ZeroDivisionError):
                pass

        return VideoResult(
            video_url=video_url,
            duration_seconds=duration,
            extra={"raw_response": data},
        )


__all__ = ["AgnesVideoProvider"]
