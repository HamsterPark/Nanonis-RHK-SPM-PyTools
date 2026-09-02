"""Removal of solid-colour (unscanned) borders from exported SPM images.

Exported STM/AFM images (PNG/JPEG) often carry flat single-colour regions where the
probe did not scan, sometimes with a scale bar or crosshair drawn on top of them.  The
detector works on rendered images, not on raw ``.sxm``/``.sm4`` data:

1. For every row and column the median colour is taken and the fraction of pixels
   within ``color_tol`` of it is measured; rows/columns above ``row_thresh`` are *solid*.
2. Starting from each edge, blocks of ``block_size`` rows/columns are removed while at
   least ``block_thresh`` of the block is solid (so a thin scale-bar line inside the
   border does not stop the scan).
3. Crops thinner than ``min_crop_frac`` of the dimension are ignored (thin frames are
   usually intentional) and a crop that would remove ``max_crop_frac`` or more is
   rejected as a false positive.
4. Steps 1-3 are repeated up to ``max_iter`` times, because removing a top/bottom
   border can expose a left/right border that was previously "diluted".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"})


@dataclass(frozen=True)
class CropConfig:
    """Detection thresholds (see the module docstring)."""

    color_tol: float = 15.0
    row_thresh: float = 0.85
    block_size: int = 20
    block_thresh: float = 0.6
    min_crop_frac: float = 0.08
    max_crop_frac: float = 0.9
    max_iter: int = 3


DEFAULT_CONFIG = CropConfig()


@dataclass(frozen=True)
class CropBox:
    """Pixels to remove from each edge."""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def __bool__(self) -> bool:
        return any((self.top, self.bottom, self.left, self.right))

    def __add__(self, other: CropBox) -> CropBox:
        return CropBox(
            self.top + other.top,
            self.bottom + other.bottom,
            self.left + other.left,
            self.right + other.right,
        )


def solid_lines(
    arr: np.ndarray, color_tol: float, row_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(rows, columns)`` flags: True where the line is (almost) one colour."""
    arr = np.asarray(arr, dtype=np.float32)
    row_medians = np.median(arr, axis=1, keepdims=True)
    row_close = np.all(np.abs(arr - row_medians) < color_tol, axis=2)
    col_medians = np.median(arr, axis=0, keepdims=True)
    col_close = np.all(np.abs(arr - col_medians) < color_tol, axis=2)
    return row_close.mean(axis=1) >= row_thresh, col_close.mean(axis=0) >= row_thresh


def scan_edge(
    solid: np.ndarray, block_size: int, block_thresh: float, from_end: bool = False
) -> int:
    """Number of lines to remove from one edge, advancing in blocks."""
    flags = solid[::-1] if from_end else solid
    crop = 0
    start = 0
    while start < len(flags):
        end = min(start + block_size, len(flags))
        if np.mean(flags[start:end]) < block_thresh:
            break
        crop = end
        start = end
    return crop


def detect_edges(arr: np.ndarray, config: CropConfig = DEFAULT_CONFIG) -> CropBox:
    """One detection pass over an ``(H, W, 3)`` array; returns the crop for each edge."""
    h, w = arr.shape[:2]
    rows, cols = solid_lines(arr, config.color_tol, config.row_thresh)
    top = scan_edge(rows, config.block_size, config.block_thresh)
    bottom = scan_edge(rows, config.block_size, config.block_thresh, from_end=True)
    left = scan_edge(cols, config.block_size, config.block_thresh)
    right = scan_edge(cols, config.block_size, config.block_thresh, from_end=True)

    min_h, min_w = int(h * config.min_crop_frac), int(w * config.min_crop_frac)
    top, bottom = (top if top >= min_h else 0), (bottom if bottom >= min_h else 0)
    left, right = (left if left >= min_w else 0), (right if right >= min_w else 0)
    if top + bottom >= h * config.max_crop_frac:
        top = bottom = 0
    if left + right >= w * config.max_crop_frac:
        left = right = 0
    return CropBox(top, bottom, left, right)


def crop_array(arr: np.ndarray, config: CropConfig = DEFAULT_CONFIG) -> tuple[np.ndarray, CropBox]:
    """Iteratively crop an ``(H, W, 3)`` array; returns the result and the total crop."""
    arr = np.asarray(arr, dtype=np.float32)
    total = CropBox()
    for _ in range(config.max_iter):
        box = detect_edges(arr, config)
        if not box:
            break
        h, w = arr.shape[:2]
        arr = arr[box.top : h - box.bottom, box.left : w - box.right]
        total = total + box
    return arr, total


@dataclass
class CropResult:
    """What happened to one image file."""

    path: Path
    original_size: tuple[int, int]
    """``(width, height)`` before cropping."""
    cropped_size: tuple[int, int]
    box: CropBox
    written: Path | None = None
    """Destination that was (or, in dry-run mode, would be) written."""
    skipped: str | None = None
    """Reason the file was left alone, if any."""

    @property
    def changed(self) -> bool:
        return bool(self.box)

    @property
    def removed_percent(self) -> float:
        ow, oh = self.original_size
        cw, ch = self.cropped_size
        return round(100.0 * (1.0 - (cw * ch) / (ow * oh)), 1) if ow and oh else 0.0


def find_images(directory: Path, exts: frozenset[str] = IMAGE_EXTS) -> list[Path]:
    """All image files below ``directory`` (case-insensitive extensions), sorted."""
    return sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in exts)


def crop_image_file(
    path: Path,
    config: CropConfig = DEFAULT_CONFIG,
    *,
    dry_run: bool = False,
    out_dir: Path | None = None,
    root: Path | None = None,
    jpeg_quality: int = 95,
) -> CropResult:
    """Detect and remove solid borders of one image file.

    The crop is applied to the original image object, so colour mode, palette and alpha
    channel survive.  JPEGs are re-encoded with ``jpeg_quality`` and keep their EXIF
    block; ICC profiles are preserved for every format.  Animated images are skipped.
    Without ``out_dir`` the file is replaced in place; with it, the result is written to
    ``out_dir / path.relative_to(root)`` (or ``out_dir / path.name`` without ``root``).
    """
    path = Path(path)
    with Image.open(path) as img:
        fmt = img.format
        width, height = img.size
        if getattr(img, "n_frames", 1) > 1:
            return CropResult(
                path, (width, height), (width, height), CropBox(), skipped="animated image"
            )
        img.load()
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        _, box = crop_array(rgb, config)
        if not box:
            return CropResult(path, (width, height), (width, height), CropBox())
        cropped = img.crop((box.left, box.top, width - box.right, height - box.bottom))
        save_kwargs: dict[str, object] = {}
        if "icc_profile" in img.info:
            save_kwargs["icc_profile"] = img.info["icc_profile"]
        if fmt == "JPEG":
            save_kwargs["quality"] = jpeg_quality
            if "exif" in img.info:
                save_kwargs["exif"] = img.info["exif"]

    if out_dir is None:
        target = path
    else:
        rel = path.relative_to(root) if root is not None else Path(path.name)
        target = Path(out_dir) / rel
    result = CropResult(path, (width, height), cropped.size, box, written=target)
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(target, format=fmt, **save_kwargs)
    return result


__all__ = [
    "IMAGE_EXTS",
    "CropBox",
    "CropConfig",
    "CropResult",
    "crop_array",
    "crop_image_file",
    "detect_edges",
    "find_images",
    "scan_edge",
    "solid_lines",
]
