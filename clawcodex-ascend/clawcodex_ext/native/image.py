#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""F-81.3: 图像差异对比与处理模块.

对标 CCB ``color-diff-napi`` + ``image-processor-napi``，用 ``Pillow`` +
``numpy`` 实现:

* :meth:`ImageProcessorModule.compute_diff` —— 像素 MSE 差异比率 (0.0 ~ 1.0)
* :meth:`ImageProcessorModule.crop_and_resize` —— 裁剪 + 缩放 + JPEG 编码

缺失 ``Pillow`` 或 ``numpy`` 时，:func:`clawcodex_ext.native.load_or_fallback`
返回 :class:`ImageFallback`，后者用纯 Python 字节比较给出 *近似* 差异
（字节级 ``hashlib`` 比较，不区分像素）。

前置依赖: F-61 Computer Use.
"""

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
    """截图差异对比与图像处理（Pillow + NumPy 实现）."""

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

    # -- 差异对比 ---------------------------------------------------------

    def compute_diff(self, img1_path: str, img2_path: str) -> float:
        """计算两张截图的像素差异比率 (0.0 完全相同 ~ 1.0 完全不同).

        Returns:
            ``MSE / 255**2``，归一化到 [0, 1] 区间.

        Raises:
            NativeModuleError: 依赖缺失或图像无法打开.
        """
        if not self._available:
            from clawcodex_ext.native import NativeModuleError

            raise NativeModuleError("image backend unavailable (install Pillow and numpy)")
        from PIL import Image
        import numpy as np

        im1 = Image.open(img1_path).convert("RGB")
        im2 = Image.open(img2_path).convert("RGB")
        # 尺寸不一致 → 取最小公共尺寸对齐（避免广播报错）
        if im1.size != im2.size:
            w = min(im1.width, im2.width)
            h = min(im1.height, im2.height)
            im1 = im1.crop((0, 0, w, h))
            im2 = im2.crop((0, 0, w, h))
        arr1 = np.asarray(im1, dtype=np.float32)
        arr2 = np.asarray(im2, dtype=np.float32)
        return float(np.mean((arr1 - arr2) ** 2) / (255.0**2))

    def images_equal(self, img1_path: str, img2_path: str, threshold: float = 0.01) -> bool:
        """便捷包装：差异 < ``threshold`` 视为相等."""
        return self.compute_diff(img1_path, img2_path) < threshold

    # -- 裁剪 + 缩放 ------------------------------------------------------

    def crop_and_resize(
        self,
        image_path: str,
        box: tuple[int, int, int, int],
        size: Optional[tuple[int, int]] = None,
        output_path: Optional[str] = None,
        quality: int = 85,
    ) -> bytes:
        """裁剪 ``box`` 区域，可选缩放到 ``size``，返回 JPEG 字节.

        Args:
            image_path: 输入图像路径.
            box: 裁剪框 ``(left, upper, right, lower)``.
            size: 可选目标尺寸 ``(width, height)``；``None`` 保持原尺寸.
            output_path: 可选落盘路径；同时返回字节.
            quality: JPEG 质量 (1-100).

        Returns:
            JPEG 编码字节.
        """
        if not self._available:
            from clawcodex_ext.native import NativeModuleError

            raise NativeModuleError("image backend unavailable (install Pillow and numpy)")
        from PIL import Image

        im = Image.open(image_path)
        cropped = im.crop(box)
        if size is not None:
            cropped = cropped.resize(size, Image.LANCZOS)
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
        """重编码图像为指定格式."""
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

    # -- F-81.6 fallback --------------------------------------------------

    @classmethod
    def fallback(cls) -> "ImageFallback":
        return ImageFallback()


class ImageFallback:
    """F-81.6 fallback: Pillow/NumPy 缺失时的兜底实现.

    差异对比降级为字节级 ``sha256`` 比较（完全相同 → 0.0，否则 → 1.0），
    失去像素级精度但能区分"完全相同 / 完全不同"两种极端.
    """

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
        """fallback 不支持裁剪/缩放，返回原始字节."""
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
