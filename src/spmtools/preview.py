"""Per-channel mosaic previews of whole SXM/SM4 datasets.

Every data folder gets a ``PreviewPy`` sub-folder with one PNG mosaic per channel
(``Z_01.png``, ``Z_line_01.png``, ``Current_01.png``, ...).  Each tile is one file,
percentile-clipped and colour-mapped, labelled with the trailing digits of the file name.
Large folders are split into several mosaics.

Channel policy for ``.sxm`` files (see :func:`iter_sxm_tiles`):

* backward-only channels and the channels in :data:`SKIP_CHANNELS` are ignored;
* ``Z`` is rendered twice: raw and after row-median subtraction (``Z_line``);
* ``Current`` is kept only when a frequency-shift channel carries signal (nc-AFM),
  or when there is no usable Z channel;
* constant or all-zero channels are dropped unless ``skip_constant`` is off.

For ``.sm4`` files only the first forward topography page is rendered (``Z`` and
``Z_line``).
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import shutil
import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spmtools.dataset import (
    SUPPORTED_EXTS,
    collect_folder_files,
    find_data_folders,
    relative_prefix,
    safe_filename,
)
from spmtools.io.sm4 import Sm4File
from spmtools.io.sxm import SxmFile

logger = logging.getLogger(__name__)

SKIP_CHANNELS = frozenset(
    {
        "bias",
        "x",
        "y",
        "oc_d1_phase",
        "oc_d1_amplitude",
        "oc_d1_excitation",
        "oc_m1_excitation",
        "li_demod_1_x",
        "li_demod_1_y",
        "li_demod_2_x",
        "li_demod_2_y",
    }
)
FREQ_SHIFT_CHANNELS = frozenset({"oc_d1_freq_shift", "oc_m1_freq_shift"})
Z_CHANNELS = frozenset({"z"})
CURRENT_CHANNELS = frozenset({"current"})

CMAP_FIXED = {
    "z": "copper",
    "z_line": "cividis",
    "current": "inferno",
    "oc_d1_freq_shift": "gray",
    "oc_m1_freq_shift": "gray",
}
CMAP_CYCLE = ("viridis", "magma", "plasma", "inferno", "cividis", "turbo", "cubehelix", "coolwarm")

GRID_COLOR = (48, 48, 48)
GRID_WIDTH = 1
SM4_PREVIEW_PATTERNS = ("Z_*.png", "Z_line_*.png")
SM4_COLLECT_DEFAULT = "SM4_PreviewPy"


@dataclass(frozen=True)
class PreviewConfig:
    """Rendering and channel-selection options (the former command-line namespace)."""

    out_name: str = "PreviewPy"
    """Name of the output folder created inside every data folder."""
    tile_size: int = 256
    max_tiles: int = 25
    """Tiles per mosaic before a new part is started (0 = unlimited)."""
    cols: int = 5
    """Fixed number of columns (0 = square-ish automatic layout)."""
    percentiles: tuple[float, float] = (1.0, 99.0)
    limit: int = 0
    """Maximum number of files per folder (0 = all)."""
    label_digits: int = 5
    add_label: bool = True
    skip_constant: bool = True
    const_tol: float = 0.0
    zero_tol: float = 0.0


# --------------------------------------------------------------------------- helpers


def canonical_name(name: str) -> str:
    """``"OC D1 Freq. Shift"`` -> ``"oc_d1_freq_shift"``."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def extract_tail_id(stem: str, digits: int) -> str:
    """Last ``digits`` trailing digits of a file stem, zero-padded; the stem if none."""
    match = re.search(r"(\d+)$", stem)
    if not match:
        return stem
    return match.group(1)[-digits:].zfill(digits)


def has_signal(data: np.ndarray | None, const_tol: float = 0.0, zero_tol: float = 0.0) -> bool:
    """False for empty, all-NaN, all-zero (within ``zero_tol``) or constant images."""
    if data is None:
        return False
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return False
    if zero_tol > 0 and np.all(np.abs(finite) <= zero_tol):
        return False
    return bool(float(np.max(finite)) - float(np.min(finite)) > const_tol)


def line_normalize(data: np.ndarray) -> np.ndarray:
    """Subtract the median of every row (removes line-to-line offsets of Z images)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row_median = np.nanmedian(data, axis=1, keepdims=True)
    return data - np.where(np.isnan(row_median), 0.0, row_median)


def normalize_for_display(data: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    """Scale to ``[0, 1]`` between the given percentiles of the finite values."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.float32)
    vmin, vmax = np.percentile(finite, percentiles)
    if vmin == vmax:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmin == vmax:
        return np.zeros(data.shape, dtype=np.float32)
    normed = (data - vmin) / (vmax - vmin)
    return np.clip(normed, 0.0, 1.0).astype(np.float32)


def pick_colormap(canonical: str) -> str:
    """Fixed colormap for known channels, otherwise a stable hash-based choice."""
    if canonical in CMAP_FIXED:
        return CMAP_FIXED[canonical]
    digest = hashlib.md5(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()
    return CMAP_CYCLE[int(digest, 16) % len(CMAP_CYCLE)]


def render_tile(
    data: np.ndarray,
    cmap_name: str,
    tile_size: int,
    label: str = "",
    percentiles: tuple[float, float] = (1.0, 99.0),
    add_label: bool = True,
) -> Image.Image:
    """Colour-map one image into a square RGB tile with an optional text label."""
    normed = normalize_for_display(data, percentiles)
    rgb = (matplotlib.colormaps[cmap_name](normed)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    if tile_size and img.size != (tile_size, tile_size):
        img = img.resize((tile_size, tile_size), resample=Image.Resampling.BILINEAR)
    if add_label and label:
        draw = ImageDraw.Draw(img)
        draw.text(
            (2, 2),
            label,
            fill=(255, 255, 255),
            font=ImageFont.load_default(),
            stroke_fill=(0, 0, 0),
            stroke_width=1,
        )
    return img


def is_backward_channel(name: str, direction: str) -> bool:
    """True for channels that only exist in (or are named after) the backward scan."""
    if "bwd" in canonical_name(name):
        return True
    d = direction.strip().lower()
    return d.startswith("bwd") or d == "backward"


class MosaicBuilder:
    """Collects tiles for one channel and writes them as ``<channel>_<part>.png``."""

    def __init__(self, channel_name: str, out_dir: Path, config: PreviewConfig):
        self.channel_name = channel_name
        self.out_dir = Path(out_dir)
        self.config = config
        self.tiles: list[Image.Image] = []
        self.part_index = 1
        self.outputs: list[Path] = []

    def add(self, tile: Image.Image) -> None:
        self.tiles.append(tile)
        if self.config.max_tiles and len(self.tiles) >= self.config.max_tiles:
            self.flush()

    def flush(self) -> Path | None:
        if not self.tiles:
            return None
        size = self.config.tile_size
        count = len(self.tiles)
        cols = self.config.cols or int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))
        mosaic = Image.new("RGB", (cols * size, rows * size), color=(0, 0, 0))
        for idx, tile in enumerate(self.tiles):
            mosaic.paste(tile, ((idx % cols) * size, (idx // cols) * size))
        if cols > 1 or rows > 1:
            draw = ImageDraw.Draw(mosaic)
            for col in range(1, cols):
                draw.line([(col * size, 0), (col * size, rows * size)], GRID_COLOR, GRID_WIDTH)
            for row in range(1, rows):
                draw.line([(0, row * size), (cols * size, row * size)], GRID_COLOR, GRID_WIDTH)
        out_path = self.out_dir / f"{safe_filename(self.channel_name)}_{self.part_index:02d}.png"
        mosaic.save(out_path, optimize=True)
        self.outputs.append(out_path)
        self.tiles = []
        self.part_index += 1
        return out_path

    def finalize(self) -> list[Path]:
        self.flush()
        return list(self.outputs)


# --------------------------------------------------------------------------- per-file policy

Tile = tuple[str, Image.Image]


def iter_sxm_tiles(path: Path, config: PreviewConfig) -> Iterator[Tile]:
    """Yield ``(mosaic name, tile)`` for every channel of an ``.sxm`` file worth showing."""
    sxm = SxmFile(path)
    label = extract_tail_id(path.stem, config.label_digits)
    by_canonical = {canonical_name(ch.name): ch for ch in reversed(sxm.channels)}

    def first(names: frozenset[str]):
        for name in names:
            ch = by_canonical.get(name)
            if ch is not None and ch.has_forward:
                return ch
        return None

    z_ch, freq_ch = first(Z_CHANNELS), first(FREQ_SHIFT_CHANNELS)
    z_data = sxm.read(z_ch) if z_ch is not None else None
    freq_data = sxm.read(freq_ch) if freq_ch is not None else None
    z_has_signal = has_signal(z_data, config.const_tol, config.zero_tol)
    freq_has_signal = has_signal(freq_data, config.const_tol, config.zero_tol)

    def tile(data: np.ndarray, cmap_key: str) -> Image.Image:
        return render_tile(
            data,
            pick_colormap(cmap_key),
            config.tile_size,
            label,
            config.percentiles,
            config.add_label,
        )

    for ch in sxm.channels:
        cname = canonical_name(ch.name)
        if is_backward_channel(ch.name, ch.direction) or not ch.has_forward:
            continue
        if cname in SKIP_CHANNELS:
            continue
        if cname in Z_CHANNELS:
            if z_has_signal and z_data is not None:
                yield "Z", tile(z_data, "z")
                yield "Z_line", tile(line_normalize(z_data), "z_line")
            continue
        if cname in FREQ_SHIFT_CHANNELS:
            if not freq_has_signal or freq_data is None:
                continue
            data = freq_data
        elif cname in CURRENT_CHANNELS:
            if z_has_signal and not freq_has_signal:
                continue
            data = sxm.read(ch)
        else:
            data = sxm.read(ch)
        if config.skip_constant and not has_signal(data, config.const_tol, config.zero_tol):
            continue
        yield ch.name, tile(data, cname)


def iter_sm4_tiles(path: Path, config: PreviewConfig) -> Iterator[Tile]:
    """Yield ``Z`` and ``Z_line`` tiles for the first forward topography page."""
    sm4 = Sm4File(path)
    page = sm4.first_topography()
    if page is None:
        logger.warning("%s: no topography image page", path)
        return
    data = sm4.read_image(page)
    if config.skip_constant and not has_signal(data, config.const_tol, config.zero_tol):
        return
    label = extract_tail_id(path.stem, config.label_digits)
    for key, img in (("Z", data), ("Z_line", line_normalize(data))):
        yield (
            key,
            render_tile(
                img,
                pick_colormap(key.lower()),
                config.tile_size,
                label,
                config.percentiles,
                config.add_label,
            ),
        )


def iter_file_tiles(path: Path, config: PreviewConfig) -> Iterator[Tile]:
    suffix = path.suffix.lower()
    if suffix == ".sxm":
        yield from iter_sxm_tiles(path, config)
    elif suffix == ".sm4":
        yield from iter_sm4_tiles(path, config)
    else:
        logger.warning("%s: unsupported extension", path)


# --------------------------------------------------------------------------- folders


@dataclass
class FolderResult:
    """What :func:`process_folder` produced for one data folder."""

    folder: Path
    out_dir: Path
    n_files: int = 0
    n_failed: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    """Tiles per mosaic name."""
    outputs: list[Path] = field(default_factory=list)
    """Mosaic PNGs written into ``out_dir``."""
    collected: list[Path] = field(default_factory=list)
    """Copies made into the collect folders."""


def collect_previews(
    out_dir: Path, collect_dir: Path, prefix: str, patterns: Iterable[str] = ("*.png",)
) -> list[Path]:
    """Copy ``out_dir/<pattern>`` into ``collect_dir`` as ``<prefix>__<name>``."""
    collect_dir = Path(collect_dir)
    collect_dir.mkdir(parents=True, exist_ok=True)
    images: set[Path] = set()
    for pattern in patterns:
        images.update(Path(out_dir).glob(pattern))
    copied = []
    for image_path in sorted(images):
        target = collect_dir / f"{prefix}__{image_path.name}"
        shutil.copy2(image_path, target)
        copied.append(target)
    return copied


def process_folder(
    folder: Path,
    files: Iterable[Path],
    config: PreviewConfig,
    *,
    prefix: str | None = None,
    collect_dir: Path | None = None,
    sm4_collect_dir: Path | None = None,
    on_file: Callable[[Path], None] | None = None,
) -> FolderResult:
    """Render every file of one folder into mosaics under ``folder / config.out_name``.

    Unreadable files are logged as warnings and counted in ``n_failed``; the rest of the
    folder is still processed.
    """
    folder = Path(folder)
    out_dir = folder / config.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = FolderResult(folder=folder, out_dir=out_dir)
    builders: dict[str, MosaicBuilder] = {}
    sm4_seen = False
    for path in files:
        path = Path(path)
        result.n_files += 1
        sm4_seen = sm4_seen or path.suffix.lower() == ".sm4"
        try:
            for key, tile in iter_file_tiles(path, config):
                builders.setdefault(key, MosaicBuilder(key, out_dir, config)).add(tile)
                result.counts[key] = result.counts.get(key, 0) + 1
        except Exception as exc:  # one bad file must not stop the run
            result.n_failed += 1
            logger.warning("%s: %s: %s", path, type(exc).__name__, exc)
        if on_file is not None:
            on_file(path)
    for builder in builders.values():
        result.outputs.extend(builder.finalize())
    if prefix is not None:
        if collect_dir is not None:
            result.collected += collect_previews(out_dir, collect_dir, prefix)
        if sm4_collect_dir is not None and sm4_seen:
            result.collected += collect_previews(
                out_dir, sm4_collect_dir, prefix, SM4_PREVIEW_PATTERNS
            )
    logger.info(
        "%s: %s",
        folder,
        ", ".join(f"{k}={v}" for k, v in sorted(result.counts.items())) or "no tiles",
    )
    return result


def _folder_task(
    folder: str,
    files: list[str],
    config: PreviewConfig,
    prefix: str,
    collect_dir: str | None,
    sm4_collect_dir: str | None,
) -> FolderResult:
    """Picklable worker entry point for :class:`ProcessPoolExecutor`."""
    return process_folder(
        Path(folder),
        [Path(f) for f in files],
        config,
        prefix=prefix,
        collect_dir=Path(collect_dir) if collect_dir else None,
        sm4_collect_dir=Path(sm4_collect_dir) if sm4_collect_dir else None,
    )


@dataclass
class PreviewSummary:
    """Aggregate of a :func:`run_preview` call."""

    root: Path
    folders: list[FolderResult] = field(default_factory=list)
    type_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_files(self) -> int:
        return sum(r.n_files for r in self.folders)

    @property
    def n_failed(self) -> int:
        return sum(r.n_failed for r in self.folders)

    @property
    def outputs(self) -> list[Path]:
        return [p for r in self.folders for p in r.outputs]


def run_preview(
    root: Path,
    config: PreviewConfig | None = None,
    *,
    recursive: bool = False,
    workers: int = 1,
    collect_dir: Path | None = None,
    sm4_collect_dir: Path | None = None,
    on_start: Callable[[int, dict[str, int]], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> PreviewSummary:
    """Render previews for ``root`` (or, with ``recursive``, every data folder below it).

    ``collect_dir`` receives a copy of every mosaic named ``<relative path>__<mosaic>.png``;
    ``sm4_collect_dir`` receives only the SM4 topography mosaics and is created only
    when the dataset contains ``.sm4`` files.  ``on_start`` is called once with the total
    number of files and the count per extension; ``on_progress`` with the number of files
    finished since the previous call (per file when ``workers <= 1``, per folder
    otherwise).
    """
    root = Path(root)
    config = config or PreviewConfig()
    folders = find_data_folders(root, recursive=recursive, skip_names={config.out_name})
    folder_files = collect_folder_files(folders, limit=config.limit)
    summary = PreviewSummary(root=root)
    for _, files in folder_files:
        for path in files:
            ext = path.suffix.lower()
            summary.type_counts[ext] = summary.type_counts.get(ext, 0) + 1
    if not folder_files:
        logger.warning("no %s files found under %s", "/".join(sorted(SUPPORTED_EXTS)), root)
        return summary
    if on_start is not None:
        on_start(sum(summary.type_counts.values()), dict(summary.type_counts))
    if collect_dir is not None:
        Path(collect_dir).mkdir(parents=True, exist_ok=True)
    if sm4_collect_dir is not None and summary.type_counts.get(".sm4", 0) == 0:
        sm4_collect_dir = None  # do not create an empty folder

    jobs = [(folder, files, relative_prefix(folder, root)) for folder, files in folder_files]
    if workers <= 1:
        for folder, files, prefix in jobs:
            result = process_folder(
                folder,
                files,
                config,
                prefix=prefix,
                collect_dir=collect_dir,
                sm4_collect_dir=sm4_collect_dir,
                on_file=(lambda _p: on_progress(1)) if on_progress else None,
            )
            summary.folders.append(result)
        return summary

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _folder_task,
                str(folder),
                [str(f) for f in files],
                config,
                prefix,
                str(collect_dir) if collect_dir else None,
                str(sm4_collect_dir) if sm4_collect_dir else None,
            ): (folder, len(files))
            for folder, files, prefix in jobs
        }
        for future in as_completed(futures):
            folder, n_files = futures[future]
            try:
                summary.folders.append(future.result())
            except Exception as exc:  # keep the other workers going
                logger.error("%s: worker failed: %s", folder, exc)
                summary.folders.append(
                    FolderResult(
                        folder=folder,
                        out_dir=folder / config.out_name,
                        n_files=n_files,
                        n_failed=n_files,
                    )
                )
            if on_progress is not None:
                on_progress(n_files)
    summary.folders.sort(key=lambda r: r.folder)
    return summary


__all__ = [
    "CMAP_CYCLE",
    "CMAP_FIXED",
    "CURRENT_CHANNELS",
    "FREQ_SHIFT_CHANNELS",
    "SKIP_CHANNELS",
    "SM4_COLLECT_DEFAULT",
    "SM4_PREVIEW_PATTERNS",
    "Z_CHANNELS",
    "FolderResult",
    "MosaicBuilder",
    "PreviewConfig",
    "PreviewSummary",
    "canonical_name",
    "collect_previews",
    "extract_tail_id",
    "has_signal",
    "is_backward_channel",
    "iter_file_tiles",
    "iter_sm4_tiles",
    "iter_sxm_tiles",
    "line_normalize",
    "normalize_for_display",
    "pick_colormap",
    "process_folder",
    "render_tile",
    "run_preview",
]
