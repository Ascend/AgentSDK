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

"""Image comparison and processing support."""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Optional

from clawcodex_ext.native import NativeModuleRegistry

__all__ = ["ImageProcessorModule", "ImageFallback"]

_logger = logging.getLogger("clawcodex_ext.native.image")


def _pil_numpy_available() -> bool:
    try:
        import PIL  # noqa: F401
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False


@NativeModuleRegistry.register("image_processor")
class ImageProcessorModule:
    """Compare, crop, resize, and encode images."""

    name = "image_processor"

    def __init__(self) -> None:
        self._available = _pil_numpy_available()

    # -- NativeModule protocol --------------------------------------------

    def is_available(self) -> bool:
        return self._available

    def get_version(self) -> str:
        if not self._available:
            return "unavailable"
        try:
            import PIL
            import numpy

            return f"PIL={PIL.__version__},numpy={numpy.__version__}"
        except ImportError:
            return "unavailable"

    # -- Image comparison -------------------------------------------------

    def compute_diff(self, img1_path: str, img2_path: str) -> float:
        """Return the normalized pixel difference between two images."""
        if not self._available:
            from clawcodex_ext.native import NativeModuleError

            raise NativeModuleError("image backend unavailable (install Pillow and numpy)")
        from PIL import Image
        import numpy as np

        im1 = Image.open(img1_path).convert("RGB")
        im2 = Image.open(img2_path).convert("RGB")
        # Align mismatched images to their smallest common dimensions.
        if im1.size != im2.size:
            w = min(im1.width, im2.width)
            h = min(im1.height, im2.height)
            im1 = im1.crop((0, 0, w, h))
            im2 = im2.crop((0, 0, w, h))
        arr1 = np.asarray(im1, dtype=np.float32)
        arr2 = np.asarray(im2, dtype=np.float32)
        return float(np.mean((arr1 - arr2) ** 2) / (255.0**2))

    def images_equal(self, img1_path: str, img2_path: str, threshold: float = 0.01) -> bool:
        """Return whether two images differ less than the threshold."""
        return self.compute_diff(img1_path, img2_path) < threshold

    # -- Cropping and resizing --------------------------------------------

    def crop_and_resize(
        self,
        image_path: str,
        box: tuple[int, int, int, int],
        size: Optional[tuple[int, int]] = None,
        output_path: Optional[str] = None,
        quality: int = 85,
    ) -> bytes:
        """Return cropped and optionally resized image bytes."""
        if not self._available:
            from clawcodex_ext.native import NativeModuleError

            raise NativeModuleError("image backend unavailable (install Pillow and numpy)")
        from PIL import Image

        im = Image.open(image_path)
        cropped = im.crop(box)
        if size is not None:
            cropped = cropped.resize(size, Image.LANCZOS)  # pylint: disable=no-member
        if output_path:
            cropped.save(output_path, "JPEG", quality=quality)
        buf = io.BytesIO()
        cropped.save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    def encode(
        self,
        image_path: str,
        fmt: str = "JPEG",
        quality: int = 85,
        output_path: Optional[str] = None,
    ) -> bytes:
        """Encode an image in the requested format."""
        if not self._available:
            from clawcodex_ext.native import NativeModuleError

            raise NativeModuleError("image backend unavailable (install Pillow and numpy)")
        from PIL import Image

        im = Image.open(image_path).convert("RGB")
        if output_path:
            im.save(output_path, fmt, quality=quality)
        buf = io.BytesIO()
        im.save(buf, fmt, quality=quality)
        return buf.getvalue()

    # -- fallback --------------------------------------------------

    @classmethod
    def fallback(cls) -> "ImageFallback":
        return ImageFallback()


class ImageFallback:
    """Compare images without optional imaging dependencies."""

    name = "image_processor"

    def is_available(self) -> bool:
        return False

    def get_version(self) -> str:
        return "fallback-bytesha"

    def compute_diff(self, img1_path: str, img2_path: str) -> float:
        with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
            h1 = hashlib.sha256(f1.read()).digest()
            h2 = hashlib.sha256(f2.read()).digest()
        return 0.0 if h1 == h2 else 1.0

    def images_equal(self, img1_path: str, img2_path: str, threshold: float = 0.01) -> bool:
        return self.compute_diff(img1_path, img2_path) < threshold

    def crop_and_resize(
        self,
        image_path: str,
        box: tuple[int, int, int, int],
        size: Optional[tuple[int, int]] = None,
        output_path: Optional[str] = None,
        quality: int = 85,
    ) -> bytes:
        """Return cropped and optionally resized image bytes."""
        with open(image_path, "rb") as f:
            data = f.read()
        if output_path:
            with open(output_path, "wb") as f:
                f.write(data)
        return data

    def encode(
        self,
        image_path: str,
        fmt: str = "JPEG",
        quality: int = 85,
        output_path: Optional[str] = None,
    ) -> bytes:
        with open(image_path, "rb") as f:
            data = f.read()
        if output_path:
            with open(output_path, "wb") as f:
                f.write(data)
        return data
