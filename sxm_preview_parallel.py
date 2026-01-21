#!/usr/bin/env python3
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sxm_preview as base


def worker_task(folder, files, args, rel_path, collect_dir):
    folder_path = Path(folder)
    file_paths = [Path(p) for p in files]
    if not file_paths:
        return 0
    rel = Path(rel_path) if rel_path else None
    collect = Path(collect_dir) if collect_dir else None
    base.process_folder(folder_path, file_paths, args, progress=None, rel_path=rel, collect_dir=collect)
    return len(file_paths)


def default_workers():
    count = os.cpu_count() or 1
    return max(1, min(4, count))


def main():
    parser = argparse.ArgumentParser(description="Parallel mosaic previews for Nanonis SXM files.")
    parser.add_argument("root", help="Folder with .sxm files, or dataset root if --recursive")
    parser.add_argument("--recursive", action="store_true", help="Scan year/month/day folders under root")
    parser.add_argument("--workers", type=int, default=default_workers(), help="Number of worker processes")
    parser.add_argument("--out-name", default="PreviewPy", help="Output folder name inside each day folder")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels (square)")
    parser.add_argument("--max-tiles", type=int, default=25, help="Max tiles per mosaic before splitting")
    parser.add_argument("--cols", type=int, default=5, help="Fixed mosaic columns (0 = auto)")
    parser.add_argument("--percentiles", default="1,99", help="Clip percentiles for display scaling")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files per folder (0 = no limit)")
    parser.add_argument("--label-digits", type=int, default=5, help="Digits for file label (tail) in each tile")
    parser.add_argument("--keep-constant", dest="skip_constant", action="store_false", help="Keep constant/empty channels")
    parser.add_argument("--const-tol", type=float, default=0.0, help="Tolerance for constant detection")
    parser.add_argument("--zero-tol", type=float, default=0.0, help="Tolerance for all-zero detection")
    parser.add_argument("--no-label", action="store_true", help="Disable file number labels")
    parser.add_argument("--collect-dir", default="", help="Copy mosaics into this folder with prefixed names")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    args.percentiles = base.parse_percentiles(args.percentiles)
    args.skip_constant = getattr(args, "skip_constant", True)

    root = Path(args.root)
    if args.recursive:
        folders = sorted(base.find_day_folders(root))
    else:
        folders = [root]

    folder_files, total_files = base.collect_folder_files(folders, args.limit)
    if total_files == 0:
        print("No .sxm files found.")
        return

    if args.verbose:
        print(f"Folders: {len(folder_files)}")
    print(f"Total SXM files: {total_files}")

    collect_dir = None
    if args.collect_dir:
        collect_dir = Path(args.collect_dir)
        if not collect_dir.is_absolute():
            collect_dir = root / collect_dir
        collect_dir.mkdir(parents=True, exist_ok=True)

    progress = base.Progress(total_files, enabled=not args.no_progress)

    if args.workers <= 1:
        for folder, files in folder_files:
            try:
                rel_path = folder.relative_to(root)
            except ValueError:
                rel_path = Path(folder.name)
            base.process_folder(folder, files, args, progress=progress, rel_path=rel_path, collect_dir=collect_dir)
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for folder, files in folder_files:
            try:
                rel_path = folder.relative_to(root)
            except ValueError:
                rel_path = Path(folder.name)
            futures.append(
                executor.submit(
                    worker_task,
                    str(folder),
                    [str(p) for p in files],
                    args,
                    str(rel_path),
                    str(collect_dir) if collect_dir else "",
                )
            )
        for future in as_completed(futures):
            try:
                done = future.result()
            except Exception as exc:
                if args.verbose:
                    print(f"[WARN] Worker failed: {exc}")
                done = 0
            progress.advance(done)


if __name__ == "__main__":
    main()
