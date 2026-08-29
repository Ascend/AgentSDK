#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""image comparison and processing unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from clawcodex_ext.native import load_or_fallback
from clawcodex_ext.native.image import ImageFallback, ImageProcessorModule


def _pil_available() -> bool:
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def two_png_files(tmp_path: Path):
    """Create identical and different pairs of solid-color PNG files."""
    same1 = tmp_path / "a.png"
    same2 = tmp_path / "b.png"
    diff1 = tmp_path / "c.png"
    diff2 = tmp_path / "d.png"
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed; image module main path untestable here")
    Image.new("RGB", (8, 8), (255, 0, 0)).save(same1)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(same2)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(diff1)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(diff2)
    return same1, same2, diff1, diff2


def test_image_module_registered():
    from clawcodex_ext.native import NativeModuleRegistry

    assert NativeModuleRegistry.is_registered("image_processor")


def test_image_fallback_compute_diff_identical(tmp_path: Path):
    fb = ImageFallback()
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"hello")
    assert fb.compute_diff(str(a), str(b)) == 0.0


def test_image_fallback_compute_diff_different(tmp_path: Path):
    fb = ImageFallback()
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    assert fb.compute_diff(str(a), str(b)) == 1.0


def test_image_fallback_crop_returns_raw_bytes(tmp_path: Path):
    fb = ImageFallback()
    src = tmp_path / "src.bin"
    src.write_bytes(b"\x89PNG-fake-data")
    out = tmp_path / "out.bin"
    data = fb.crop_and_resize(str(src), (0, 0, 4, 4), output_path=str(out))
    assert data == b"\x89PNG-fake-data"
    assert out.read_bytes() == b"\x89PNG-fake-data"


def test_image_load_or_fallback_returns_object():
    inst = load_or_fallback("image_processor")
    assert inst is not None
    assert isinstance(inst, (ImageProcessorModule, ImageFallback))


@pytest.mark.skipif(not _pil_available(), reason="Pillow+numpy not installed")
def test_image_main_compute_diff_identical(two_png_files):
    same1, same2, _, _ = two_png_files
    mod = ImageProcessorModule()
    if not mod.is_available():
        pytest.skip("Pillow/numpy unavailable")
    assert mod.compute_diff(str(same1), str(same2)) == 0.0


@pytest.mark.skipif(not _pil_available(), reason="Pillow+numpy not installed")
def test_image_main_compute_diff_different(two_png_files):
    _, _, diff1, diff2 = two_png_files
    mod = ImageProcessorModule()
    if not mod.is_available():
        pytest.skip("Pillow/numpy unavailable")
    val = mod.compute_diff(str(diff1), str(diff2))
    assert 0.0 < val <= 1.0


@pytest.mark.skipif(not _pil_available(), reason="Pillow+numpy not installed")
def test_image_main_crop_and_resize(two_png_files, tmp_path: Path):
    same1, _, _, _ = two_png_files
    mod = ImageProcessorModule()
    if not mod.is_available():
        pytest.skip("Pillow/numpy unavailable")
    out = tmp_path / "cropped.jpg"
    data = mod.crop_and_resize(str(same1), (0, 0, 4, 4), size=(2, 2), output_path=str(out))
    assert data[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert out.exists() and out.read_bytes()[:2] == b"\xff\xd8"
