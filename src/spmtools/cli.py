"""Console entry points: ``spm-preview``, ``spm-align-diff``, ``spm-crop-solid``, ``spm-sample``.

The parsers only translate arguments into the configuration objects of the library
modules; all processing lives in :mod:`spmtools.preview`, :mod:`spmtools.align`,
:mod:`spmtools.crop` and :mod:`spmtools.dataset`.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from spmtools import __version__
from spmtools.dataset import normalize_extensions, sample_subset
from spmtools.io import SpmFormatError

logger = logging.getLogger("spmtools")


class Progress:
    """Minimal text progress bar written to stderr."""

    def __init__(self, total: int, enabled: bool = True, width: int = 28):
        self.total = total
        self.enabled = enabled and total > 0
        self.width = width
        self.count = 0
        if self.enabled:
            self._print()

    def advance(self, step: int = 1) -> None:
        if not self.enabled:
            return
        self.count = min(self.total, self.count + step)
        self._print()
        if self.count >= self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _print(self) -> None:
        filled = int(self.width * self.count / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = 100.0 * self.count / self.total
        sys.stderr.write(f"\rProgress {self.count}/{self.total} [{bar}] {percent:5.1f}%")
        sys.stderr.flush()


def configure_logging(verbose: bool) -> None:
    """WARNING by default (bad files are always reported), INFO with ``--verbose``."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-folder details")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")


def parse_percentiles(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("percentiles must look like '1,99'")
    lo, hi = float(parts[0]), float(parts[1])
    if not 0 <= lo < hi <= 100:
        raise argparse.ArgumentTypeError("percentiles must satisfy 0 <= low < high <= 100")
    return lo, hi


def _resolve_dir(value: str, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


# --------------------------------------------------------------------------- spm-preview


def build_preview_parser() -> argparse.ArgumentParser:
    from spmtools.preview import SM4_COLLECT_DEFAULT

    parser = argparse.ArgumentParser(
        prog="spm-preview",
        description="Generate per-channel mosaic previews for Nanonis SXM and RHK SM4 files.",
    )
    parser.add_argument(
        "root", help="Folder with .sxm/.sm4 files (or a dataset root with --recursive)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process every folder below root that contains .sxm/.sm4 files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes; folders are processed in parallel (0 = up to 4, one per CPU)",
    )
    parser.add_argument(
        "--out-name", default="PreviewPy", help="Output folder name inside each data folder"
    )
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels (square)")
    parser.add_argument(
        "--max-tiles", type=int, default=25, help="Max tiles per mosaic before splitting"
    )
    parser.add_argument("--cols", type=int, default=5, help="Fixed mosaic columns (0 = auto)")
    parser.add_argument(
        "--percentiles",
        type=parse_percentiles,
        default=(1.0, 99.0),
        help="Clip percentiles for display scaling (default 1,99)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit files per folder (0 = no limit)"
    )
    parser.add_argument(
        "--label-digits", type=int, default=5, help="Digits of the file-number label"
    )
    parser.add_argument(
        "--keep-constant",
        dest="skip_constant",
        action="store_false",
        help="Keep constant/empty channels",
    )
    parser.add_argument(
        "--const-tol", type=float, default=0.0, help="Tolerance for constant detection"
    )
    parser.add_argument(
        "--zero-tol", type=float, default=0.0, help="Tolerance for all-zero detection"
    )
    parser.add_argument("--no-label", action="store_true", help="Disable file-number labels")
    parser.add_argument(
        "--collect-dir",
        default="",
        help="Also copy every mosaic into this folder with path-prefixed names",
    )
    parser.add_argument(
        "--sm4-collect-dir",
        default=SM4_COLLECT_DEFAULT,
        help='Folder (relative to root) for SM4 topography mosaics; "" disables',
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")
    _add_common(parser)
    return parser


def preview_main(argv: Sequence[str] | None = None) -> int:
    from spmtools.preview import PreviewConfig, run_preview

    parser = build_preview_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    root = Path(args.root)
    if not root.is_dir():
        parser.error(f"{root} is not a directory")
    config = PreviewConfig(
        out_name=args.out_name,
        tile_size=args.tile_size,
        max_tiles=args.max_tiles,
        cols=args.cols,
        percentiles=args.percentiles,
        limit=args.limit,
        label_digits=args.label_digits,
        add_label=not args.no_label,
        skip_constant=args.skip_constant,
        const_tol=args.const_tol,
        zero_tol=args.zero_tol,
    )
    workers = args.workers if args.workers > 0 else max(1, min(4, os.cpu_count() or 1))
    progress: Progress | None = None

    def on_start(total: int, type_counts: dict[str, int]) -> None:
        nonlocal progress
        summary = ", ".join(f"{ext[1:].upper()}={n}" for ext, n in sorted(type_counts.items()))
        print(f"Total files: {total} ({summary})")
        progress = Progress(total, enabled=not args.no_progress)

    summary = run_preview(
        root,
        config,
        recursive=args.recursive,
        workers=workers,
        collect_dir=_resolve_dir(args.collect_dir, root),
        sm4_collect_dir=_resolve_dir(args.sm4_collect_dir, root),
        on_start=on_start,
        on_progress=lambda n: progress.advance(n) if progress else None,
    )
    if summary.n_files == 0:
        print("No .sxm or .sm4 files found.")
        return 1
    print(
        f"Wrote {len(summary.outputs)} mosaics for {len(summary.folders)} folders"
        f" ({summary.n_failed} files failed)"
    )
    return 0


# --------------------------------------------------------------------------- spm-align-diff


def build_align_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spm-align-diff",
        description="Sub-pixel align one channel of two .sxm scans and write their difference.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Two .sxm files (A B); default: the two .sxm files in the current directory",
    )
    parser.add_argument(
        "--channel",
        default="Z",
        help='Channel name or keyword; exact match first, then substring (e.g. "Freq Shift")',
    )
    parser.add_argument("--direction", default="forward", choices=["forward", "backward"])
    parser.add_argument("--upsample", type=int, default=100, help="Sub-pixel up-sampling factor")
    parser.add_argument(
        "--cmap",
        default="RdBu",
        help="Difference colormap (RdBu: red negative / white zero / blue positive; RdBu_r flips)",
    )
    parser.add_argument("--outdir", default=None, help="Output directory (default: next to A)")
    parser.add_argument(
        "--clip", type=float, default=99.0, help="Percentile for the symmetric colour scale"
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing of the difference maps in pixels (0 = off)",
    )
    _add_common(parser)
    return parser


def align_main(argv: Sequence[str] | None = None) -> int:
    parser = build_align_parser()
    args = parser.parse_args(argv)
    configure_logging(True)
    files = list(args.files) or sorted(glob.glob("*.sxm"))
    if len(files) != 2:
        parser.error(f"exactly two .sxm files are required, got {len(files)}: {files}")
    try:
        from spmtools.align import align_sxm_files

        result, outputs = align_sxm_files(
            Path(files[0]),
            Path(files[1]),
            channel=args.channel,
            direction=args.direction,
            upsample=args.upsample,
            smooth_sigma=args.smooth_sigma,
            clip=args.clip,
            cmap=args.cmap,
            outdir=Path(args.outdir) if args.outdir else None,
        )
    except (SpmFormatError, KeyError, ValueError, ImportError, OSError) as exc:
        message = exc.args[0] if exc.args else str(exc)
        parser.exit(1, f"error: {message}\n")
    dy, dx = result.shift
    print(
        f"shift: dx={dx:+.4f} px, dy={dy:+.4f} px; residual RMS {result.rms_before:.4g} -> "
        f"{result.rms_after:.4g} ({result.improvement_percent:.1f}% lower)"
    )
    for path in outputs:
        print(f"  {path}")
    return 0


# --------------------------------------------------------------------------- spm-crop-solid


def build_crop_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spm-crop-solid",
        description="Crop solid-colour (unscanned) borders from exported STM/AFM images.",
    )
    parser.add_argument("image_dir", help="Directory with images (searched recursively)")
    parser.add_argument(
        "--tolerance", type=float, default=15.0, help="Colour tolerance per channel"
    )
    parser.add_argument(
        "--min-crop", type=float, default=0.08, help="Minimum crop as a fraction of the dimension"
    )
    parser.add_argument(
        "--block-size", type=int, default=20, help="Rows/columns per scanning block"
    )
    parser.add_argument(
        "--row-thresh",
        type=float,
        default=0.85,
        help="Fraction of matching pixels for a solid line",
    )
    parser.add_argument(
        "--block-thresh", type=float, default=0.6, help="Fraction of solid lines for a solid block"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Write cropped copies here (mirroring the tree) instead of replacing files in place",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Quality for re-encoded JPEGs")
    parser.add_argument("--dry-run", action="store_true", help="Report crops without writing files")
    _add_common(parser)
    return parser


def crop_main(argv: Sequence[str] | None = None) -> int:
    from PIL import Image

    from spmtools.crop import CropConfig, crop_image_file, find_images

    parser = build_crop_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        parser.error(f"{image_dir} is not a directory")
    Image.MAX_IMAGE_PIXELS = None  # exported overview images can be very large
    config = CropConfig(
        color_tol=args.tolerance,
        row_thresh=args.row_thresh,
        block_size=args.block_size,
        block_thresh=args.block_thresh,
        min_crop_frac=args.min_crop,
    )
    out_dir = Path(args.out_dir) if args.out_dir else None
    images = find_images(image_dir)
    if not images:
        print(f"No images found in '{image_dir}'")
        return 1
    print(f"Found {len(images)} images in '{image_dir}'")
    if args.dry_run:
        print("DRY RUN - no files will be modified")

    cropped = skipped = errors = 0
    started = time.time()
    for index, path in enumerate(images, start=1):
        if index % 500 == 0:
            rate = index / max(time.time() - started, 1e-9)
            print(f"  [{index}/{len(images)}] {rate:.1f}/s, {cropped} cropped")
        try:
            result = crop_image_file(
                path,
                config,
                dry_run=args.dry_run,
                out_dir=out_dir,
                root=image_dir,
                jpeg_quality=args.jpeg_quality,
            )
        except Exception as exc:  # keep going; the summary reports the count
            errors += 1
            logger.warning("%s: %s: %s", path, type(exc).__name__, exc)
            continue
        if result.skipped:
            skipped += 1
            logger.warning("%s: skipped (%s)", path, result.skipped)
        elif result.changed:
            cropped += 1
            if args.verbose or args.dry_run:
                box = result.box
                ow, oh = result.original_size
                cw, ch = result.cropped_size
                print(
                    f"  {'[DRY] ' if args.dry_run else ''}Crop {path.name}: {ow}x{oh} -> {cw}x{ch} "
                    f"(T={box.top} B={box.bottom} L={box.left} R={box.right}, "
                    f"-{result.removed_percent}%)"
                )
    print("=" * 40)
    print(f"Done in {time.time() - started:.0f}s")
    print(f"Total images scanned: {len(images)}")
    print(f"Images cropped: {cropped}")
    if skipped:
        print(f"Images skipped: {skipped}")
    if errors:
        print(f"Errors: {errors}")
    if args.dry_run:
        print("This was a dry run. Re-run without --dry-run to apply crops.")
    return 0


# --------------------------------------------------------------------------- spm-sample


def build_sample_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spm-sample",
        description="Copy a reproducible random subset of a dataset (for test runs).",
    )
    parser.add_argument("--root", required=True, help="Dataset root")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument(
        "--folders",
        "--days",
        dest="folders",
        type=int,
        default=3,
        help="Number of data folders to sample",
    )
    parser.add_argument(
        "--files-per-folder",
        "--files-per-day",
        dest="files_per_folder",
        type=int,
        default=12,
        help="Files per selected folder (0 = all)",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument(
        "--manifest", default="manifest.txt", help="Manifest file name (inside --out)"
    )
    parser.add_argument(
        "--extensions", default="sxm,sm4", help="Comma-separated extensions to include"
    )
    _add_common(parser)
    return parser


def sample_main(argv: Sequence[str] | None = None) -> int:
    parser = build_sample_parser()
    args = parser.parse_args(argv)
    configure_logging(True)
    root = Path(args.root)
    if not root.is_dir():
        parser.error(f"{root} is not a directory")
    copied = sample_subset(
        root,
        Path(args.out),
        folders=args.folders,
        files_per_folder=args.files_per_folder,
        seed=args.seed,
        exts=normalize_extensions(args.extensions),
        manifest_name=args.manifest,
    )
    print(f"Copied {len(copied)} files into {args.out}")
    return 0 if copied else 1


__all__ = [
    "Progress",
    "align_main",
    "build_align_parser",
    "build_crop_parser",
    "build_preview_parser",
    "build_sample_parser",
    "configure_logging",
    "crop_main",
    "parse_percentiles",
    "preview_main",
    "sample_main",
]
