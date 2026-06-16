#!/usr/bin/env python3
"""
crop_solid.py - Detect and crop solid-color (unscanned) edges from STM images.

STM images often contain solid-color regions where the probe hasn't scanned.
These regions may also contain scalebar overlays. This tool detects and removes
them using an iterative block-based algorithm that is robust to scalebar and
crosshair overlays within the solid regions.

Algorithm:
  1. For each row/column, compute the median color and check what fraction
     of pixels are within a tolerance of that median.
  2. Rows/columns where >85% of pixels match are considered "solid".
  3. Scan from each edge inward in blocks, allowing small gaps (for scalebar
     lines that break the solid pattern).
  4. Iterate up to 3 times to handle cases where removing one edge reveals
     another (e.g., removing top/bottom gray borders exposes left/right
     solid regions).

Usage:
  python crop_solid.py <image_dir> [--tolerance 15] [--min-crop 0.08] [--dry-run]

Examples:
  python crop_solid.py ./stm_images/
  python crop_solid.py ./stm_images/ --tolerance 20 --min-crop 0.05
  python crop_solid.py ./stm_images/ --dry-run   # preview without modifying
"""

import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check

# Supported image extensions
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}


def detect_edges(arr, color_tol, row_thresh, block_size, block_thresh, min_crop_pct):
    """
    Detect solid-color edges in an image array.

    Args:
        arr: numpy array of shape (H, W, 3), float32
        color_tol: max per-channel deviation from median to count as "same color"
        row_thresh: fraction of pixels that must match for a row/col to be "solid"
        block_size: number of rows/cols to group into a block
        block_thresh: fraction of rows/cols in a block that must be solid
        min_crop_pct: minimum crop as fraction of dimension

    Returns:
        (top, bottom, left, right) crop amounts in pixels
    """
    h, w = arr.shape[:2]

    # Per-row: check what fraction of pixels match the row's median color
    row_medians = np.median(arr, axis=1, keepdims=True)
    row_close = np.all(np.abs(arr - row_medians) < color_tol, axis=2)
    row_solid = np.mean(row_close, axis=1) >= row_thresh

    # Per-column: same analysis
    col_medians = np.median(arr, axis=0, keepdims=True)
    col_close = np.all(np.abs(arr - col_medians) < color_tol, axis=2)
    col_solid = np.mean(col_close, axis=0) >= row_thresh

    def scan(solid_arr, from_end=False):
        """Scan from an edge in blocks, returning how many rows/cols to crop."""
        a = solid_arr[::-1] if from_end else solid_arr
        crop = 0
        i = 0
        while i < len(a):
            end = min(i + block_size, len(a))
            if np.mean(a[i:end]) >= block_thresh:
                crop = end
                i = end
            else:
                break
        return crop

    t = scan(row_solid)
    b = scan(row_solid, True)
    l = scan(col_solid)
    r = scan(col_solid, True)

    # Apply minimum crop threshold
    min_h = int(h * min_crop_pct)
    min_w = int(w * min_crop_pct)
    if t < min_h: t = 0
    if b < min_h: b = 0
    if l < min_w: l = 0
    if r < min_w: r = 0

    # Safety: don't crop away the entire image
    if t + b >= h * 0.9: t = b = 0
    if l + r >= w * 0.9: l = r = 0

    return t, b, l, r


def process_image(img_path, color_tol=15, row_thresh=0.85, block_size=20,
                  block_thresh=0.6, min_crop_pct=0.08, max_iter=3, dry_run=False):
    """
    Process a single image: detect and crop solid edges.

    Args:
        img_path: Path to the image file
        color_tol: color tolerance (per channel)
        row_thresh: row/col solid threshold
        block_size: scanning block size
        block_thresh: block solid threshold
        min_crop_pct: minimum crop percentage
        max_iter: maximum crop iterations
        dry_run: if True, don't modify the file

    Returns:
        dict with crop info, or None if no crop needed
    """
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    orig_h, orig_w = arr.shape[:2]

    total_t = total_b = total_l = total_r = 0

    for _ in range(max_iter):
        t, b, l, r = detect_edges(arr, color_tol, row_thresh, block_size,
                                   block_thresh, min_crop_pct)
        if not any([t, b, l, r]):
            break
        total_t += t
        total_b += b
        total_l += l
        total_r += r
        h, w = arr.shape[:2]
        arr = arr[t:h - b if b else h, l:w - r if r else w]

    if not any([total_t, total_b, total_l, total_r]):
        return None

    new_h, new_w = arr.shape[:2]

    if not dry_run:
        cropped = Image.fromarray(arr.astype(np.uint8))
        ext = Path(img_path).suffix.lower()
        if ext in ('.jpg', '.jpeg'):
            cropped.save(img_path, quality=95)
        else:
            cropped.save(img_path)

    return {
        'file': str(img_path),
        'original': f'{orig_w}x{orig_h}',
        'cropped': f'{new_w}x{new_h}',
        'top': total_t,
        'bottom': total_b,
        'left': total_l,
        'right': total_r,
        'removed_pct': round(100 * (1 - new_w * new_h / (orig_w * orig_h)), 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Crop solid-color (unscanned) edges from STM images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[0].strip()
    )
    parser.add_argument('image_dir', type=str,
                        help='Directory containing STM images (searched recursively)')
    parser.add_argument('--tolerance', type=int, default=15,
                        help='Color tolerance per channel (default: 15)')
    parser.add_argument('--min-crop', type=float, default=0.08,
                        help='Minimum crop as fraction of dimension (default: 0.08)')
    parser.add_argument('--block-size', type=int, default=20,
                        help='Block size for edge scanning (default: 20)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview crops without modifying files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print details for each cropped image')

    args = parser.parse_args()
    image_dir = Path(args.image_dir)

    if not image_dir.exists():
        print(f"Error: directory '{image_dir}' not found")
        sys.exit(1)

    # Find all images
    images = []
    for ext in IMAGE_EXTS:
        images.extend(image_dir.rglob(f'*{ext}'))
        images.extend(image_dir.rglob(f'*{ext.upper()}'))
    images = sorted(set(images))

    if not images:
        print(f"No images found in '{image_dir}'")
        sys.exit(1)

    print(f"Found {len(images)} images in '{image_dir}'")
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")

    cropped_count = 0
    errors = 0
    t0 = time.time()

    for idx, img_path in enumerate(images):
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx+1}/{len(images)}] {rate:.1f}/sec, {cropped_count} cropped")

        try:
            result = process_image(
                img_path,
                color_tol=args.tolerance,
                min_crop_pct=args.min_crop,
                block_size=args.block_size,
                dry_run=args.dry_run,
            )
            if result:
                cropped_count += 1
                if args.verbose or args.dry_run:
                    name = img_path.name
                    print(f"  {'[DRY] ' if args.dry_run else ''}Crop {name}: "
                          f"{result['original']} -> {result['cropped']} "
                          f"(T={result['top']} B={result['bottom']} "
                          f"L={result['left']} R={result['right']}, "
                          f"-{result['removed_pct']}%)")
        except Exception as e:
            errors += 1
            if args.verbose:
                print(f"  ERROR {img_path.name}: {e}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 40}")
    print(f"Done in {elapsed:.0f}s")
    print(f"Total images scanned: {len(images)}")
    print(f"Images cropped: {cropped_count}")
    if errors:
        print(f"Errors: {errors}")
    if args.dry_run:
        print("\nThis was a dry run. Re-run without --dry-run to apply crops.")


if __name__ == '__main__':
    main()
