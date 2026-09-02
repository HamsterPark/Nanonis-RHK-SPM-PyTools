"""Tests for solid-border detection and cropping."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spmtools import crop
from spmtools.crop import CropBox, CropConfig


def noisy(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3)).astype(np.float32)


def test_scan_edge_advances_in_whole_blocks():
    flags = np.array([True] * 25 + [False] * 10)
    assert crop.scan_edge(flags, block_size=10, block_thresh=0.6) == 20
    assert crop.scan_edge(flags, block_size=5, block_thresh=0.6) == 25
    assert crop.scan_edge(flags, block_size=10, block_thresh=0.6, from_end=True) == 0
    assert crop.scan_edge(flags[::-1], block_size=10, block_thresh=0.6, from_end=True) == 20
    assert crop.scan_edge(np.zeros(5, dtype=bool), 20, 0.6) == 0


def test_detect_edges_finds_top_and_left_borders():
    img = noisy(200, 300)
    img[:40, :, :] = 128.0
    img[:, :60, :] = 128.0
    assert crop.detect_edges(img) == CropBox(top=40, left=60)


def test_thin_border_is_kept():
    img = noisy(200, 300)
    img[:10, :, :] = 128.0  # 5 % < min_crop_frac, and less than one 20-line block
    assert not crop.detect_edges(img)
    assert crop.detect_edges(img, CropConfig(min_crop_frac=0.02, block_size=5)) == CropBox(top=10)


def test_iteration_reveals_second_border():
    img = noisy(200, 300)
    img[:40, :, :] = 128.0
    img[:, :60, :] = 200.0  # the corner belongs to the left border: top rows are 80 % solid
    assert crop.detect_edges(img) == CropBox(left=60)
    cropped, total = crop.crop_array(img)
    assert total == CropBox(top=40, left=60)
    assert cropped.shape == (160, 240, 3)


def test_uniform_image_is_not_destroyed():
    img = np.full((100, 100, 3), 77.0, dtype=np.float32)
    assert not crop.detect_edges(img)
    _, total = crop.crop_array(img)
    assert not total


def test_scalebar_inside_border_does_not_stop_the_scan():
    img = noisy(200, 300)
    img[:60, :, :] = 128.0
    img[45:48, 200:260, :] = 255.0  # a 3-px white scale bar drawn on the border
    assert crop.detect_edges(img) == CropBox(top=60)


def test_crop_box_arithmetic():
    assert CropBox(1, 2, 3, 4) + CropBox(10, 20, 30, 40) == CropBox(11, 22, 33, 44)
    assert not CropBox()


def _bordered_rgba(w: int = 100, h: int = 160, border: int = 40) -> Image.Image:
    """Random RGBA content with a solid grey, opaque border of two full blocks on top."""
    rng = np.random.default_rng(1)
    rgba = rng.integers(0, 256, size=(h, w, 4)).astype(np.uint8)
    rgba[:border, :, :3] = 128
    rgba[:border, :, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def test_crop_image_file_preserves_alpha(tmp_path):
    path = tmp_path / "rgba.png"
    original = _bordered_rgba()
    original.save(path)
    result = crop.crop_image_file(path)
    assert result.changed and result.box == CropBox(top=40)
    assert result.original_size == (100, 160) and result.cropped_size == (100, 120)
    assert result.removed_percent == 25.0 and result.written == path
    with Image.open(path) as out:
        assert out.mode == "RGBA" and out.size == (100, 120)
        assert np.array_equal(np.array(out), np.array(original)[40:])


def test_crop_image_file_preserves_palette(tmp_path):
    path = tmp_path / "pal.png"
    _bordered_rgba().convert("RGB").quantize(colors=64).save(path)
    result = crop.crop_image_file(path)
    assert result.box.top == 40
    with Image.open(path) as out:
        assert out.mode == "P" and out.size == (100, 120)


def test_animated_gif_is_skipped(tmp_path):
    path = tmp_path / "anim.gif"
    first = _bordered_rgba().convert("RGB")
    second = Image.eval(first, lambda v: 255 - v)
    first.save(path, save_all=True, append_images=[second], duration=100)
    before = path.read_bytes()
    with Image.open(path) as gif:
        assert gif.n_frames == 2
    result = crop.crop_image_file(path)
    assert result.skipped == "animated image" and not result.changed
    assert path.read_bytes() == before


def test_dry_run_and_out_dir(tmp_path):
    src_root = tmp_path / "src"
    path = src_root / "sub" / "img.png"
    path.parent.mkdir(parents=True)
    _bordered_rgba().save(path)
    before = path.read_bytes()

    dry = crop.crop_image_file(path, dry_run=True)
    assert dry.changed and dry.written == path and path.read_bytes() == before

    out_dir = tmp_path / "out"
    result = crop.crop_image_file(path, out_dir=out_dir, root=src_root)
    assert result.written == out_dir / "sub" / "img.png" and result.written.is_file()
    assert path.read_bytes() == before
    flat = crop.crop_image_file(path, out_dir=tmp_path / "flat")
    assert flat.written == tmp_path / "flat" / "img.png"


def test_jpeg_keeps_exif_and_uses_quality(tmp_path):
    path = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[0x010E] = "synthetic STM export"
    _bordered_rgba().convert("RGB").save(path, quality=90, exif=exif.tobytes())
    result = crop.crop_image_file(path, jpeg_quality=50)
    assert result.box.top == 40
    with Image.open(path) as out:
        assert out.format == "JPEG" and out.size == (100, 120)
        assert out.getexif()[0x010E] == "synthetic STM export"


def test_no_crop_needed(tmp_path):
    path = tmp_path / "plain.png"
    Image.fromarray(noisy(40, 50).astype(np.uint8)).save(path)
    result = crop.crop_image_file(path)
    assert not result.changed and result.written is None and result.removed_percent == 0.0


def test_find_images_is_case_insensitive(tmp_path):
    for name in ("a.PNG", "b.jpg", "c.txt", "d.tiff"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "e.webp").write_bytes(b"")
    assert [p.name for p in crop.find_images(tmp_path)] == ["a.PNG", "b.jpg", "d.tiff", "e.webp"]


@pytest.mark.parametrize(("block_size", "expected"), [(1, 48), (20, 40), (500, 0)])
def test_block_size_controls_granularity(block_size, expected):
    img = noisy(120, 100)
    img[:48, :, :] = 0.0
    box = crop.detect_edges(img, CropConfig(block_size=block_size))
    assert box.top == expected
