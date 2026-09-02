"""Tests for the RHK ``.sm4`` reader."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import write_sm4
from spmtools.io import SpmFormatError
from spmtools.io.sm4 import PAGE_HEADER_SIZE, Sm4File, read_pages


def grid(y_size: int, x_size: int) -> np.ndarray:
    return (np.arange(y_size * x_size, dtype=np.int32) * 7 - 50).reshape(y_size, x_size)


def test_page_header_layout_is_180_bytes():
    assert PAGE_HEADER_SIZE == 180


def test_non_square_image_roundtrip(tmp_path):
    raw = grid(4, 8)
    path = write_sm4(tmp_path / "rect.sm4", [{"data": raw, "z_scale": 2.0e-12, "z_offset": 1.0e-9}])
    sm4 = Sm4File(path)
    assert len(sm4.pages) == 1
    page = sm4.pages[0]
    assert (page.x_size, page.y_size) == (8, 4)
    image = sm4.read_image(page)
    assert image.shape == (4, 8)
    assert image.dtype == np.float64
    assert np.allclose(image, raw * 2.0e-12 + 1.0e-9)


def test_flips_follow_scale_signs(tmp_path):
    raw = grid(3, 5)
    path = write_sm4(
        tmp_path / "flip.sm4",
        [
            {"data": raw, "x_scale": 1e-9, "y_scale": -1e-9, "z_scale": 1.0},
            {"data": raw, "x_scale": -1e-9, "y_scale": -1e-9, "z_scale": 1.0},
            {"data": raw, "x_scale": 1e-9, "y_scale": 1e-9, "z_scale": 1.0},
        ],
    )
    sm4 = Sm4File(path)
    images = [sm4.read_image(page) for page in sm4.pages]
    assert np.array_equal(images[0], raw)
    assert np.array_equal(images[1], raw[:, ::-1])
    assert np.array_equal(images[2], raw[::-1, :])


def test_float_line_types_are_decoded_as_float32(tmp_path):
    raw = np.linspace(-1.5, 1.5, 12, dtype=np.float32).reshape(3, 4)
    path = write_sm4(tmp_path / "float.sm4", [{"data": raw, "line_type": 1, "z_scale": 1.0}])
    page = Sm4File(path).pages[0]
    assert page.sample_dtype == np.dtype("<f4")
    assert np.allclose(Sm4File(path).read_image(page), raw)


def test_first_topography_prefers_forward_scan(tmp_path):
    pages = [
        {"data": grid(2, 2), "page_type": 1, "scan_type": 1, "label": "Topography bwd"},
        {"data": grid(2, 2), "page_type": 2, "scan_type": 0, "label": "Current"},
        {"data": grid(2, 2), "page_type": 1, "scan_type": 0, "label": "Topography fwd"},
    ]
    sm4 = Sm4File(write_sm4(tmp_path / "multi.sm4", pages))
    assert [p.label for p in sm4.pages] == ["Topography bwd", "Current", "Topography fwd"]
    chosen = sm4.first_topography()
    assert chosen is not None and chosen.index == 2
    assert chosen.is_forward and chosen.is_topography


def test_first_topography_falls_back_to_backward(tmp_path):
    pages = [
        {"data": grid(2, 2), "page_type": 2, "scan_type": 0},
        {"data": grid(2, 2), "page_type": 1, "scan_type": 1},
    ]
    sm4 = Sm4File(write_sm4(tmp_path / "bwd.sm4", pages))
    chosen = sm4.first_topography()
    assert chosen is not None and chosen.index == 1 and not chosen.is_forward
    only_current = Sm4File(write_sm4(tmp_path / "cur.sm4", pages[:1]))
    assert only_current.first_topography() is None


def test_page_scalars_and_strings(tmp_path):
    path = write_sm4(
        tmp_path / "meta.sm4",
        [{"data": grid(2, 3), "bias": 0.5, "current": 1e-10, "angle": 12.5, "z_unit": "nm"}],
    )
    page = Sm4File(path).pages[0]
    assert page.bias == pytest.approx(0.5)
    assert page.current == pytest.approx(1e-10)
    assert page.angle == pytest.approx(12.5)
    assert page.label == "Topography"
    assert page.z_unit == "nm"
    assert page.data_size == 2 * 3 * 4


def test_non_image_pages_are_listed_but_not_decoded(tmp_path):
    pages = [{"data": grid(1, 4), "data_type": 1}, {"data": grid(2, 2)}]
    sm4 = Sm4File(write_sm4(tmp_path / "line.sm4", pages))
    assert len(sm4.pages) == 2
    assert [p.index for p in sm4.image_pages()] == [1]
    with pytest.raises(SpmFormatError, match="not an image page"):
        sm4.read_image(sm4.pages[0])


def test_bad_signature_is_rejected(tmp_path):
    path = tmp_path / "bad.sm4"
    path.write_bytes(b"\0" * 200)
    with pytest.raises(SpmFormatError, match="not an RHK SM4"):
        Sm4File(path)


def test_truncated_file_raises(tmp_path):
    path = write_sm4(tmp_path / "trunc.sm4", [{"data": grid(4, 4)}])
    data = path.read_bytes()
    path.write_bytes(data[:-10])
    sm4 = Sm4File(path)  # headers are intact
    with pytest.raises(SpmFormatError, match="unexpected end of file"):
        sm4.read_image(sm4.pages[0])
    path.write_bytes(data[:40])
    with pytest.raises(SpmFormatError):
        with open(path, "rb") as handle:
            read_pages(handle)
