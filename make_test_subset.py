#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


def find_day_folders(root: Path):
    day_folders = []
    for year_dir in root.iterdir():
        if not year_dir.is_dir():
            continue
        if year_dir.name == "data23":
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
            for day_dir in month_dir.iterdir():
                if not day_dir.is_dir():
                    continue
                if any(day_dir.glob("*.sxm")):
                    day_folders.append(day_dir)
    return day_folders


def main():
    parser = argparse.ArgumentParser(description="Create a random SXM test subset.")
    parser.add_argument("--root", required=True, help="Dataset root (year/month/day structure)")
    parser.add_argument("--out", required=True, help="Output test folder")
    parser.add_argument("--days", type=int, default=3, help="Number of day folders to sample")
    parser.add_argument("--files-per-day", type=int, default=12, help="Files per selected day")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--manifest", default="manifest.txt", help="Manifest filename (inside output)")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    day_folders = find_day_folders(root)
    if not day_folders:
        print("No day folders with .sxm files found.")
        return

    random.seed(args.seed)
    sample_days = random.sample(day_folders, k=min(args.days, len(day_folders)))

    manifest_lines = []
    for day in sample_days:
        files = list(day.glob("*.sxm"))
        random.shuffle(files)
        if args.files_per_day > 0:
            files = files[: args.files_per_day]
        for file_path in files:
            rel = file_path.relative_to(root)
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            manifest_lines.append(str(rel))

    manifest_path = out_root / args.manifest
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="ascii")
    print(f"Copied {len(manifest_lines)} files into {out_root}")


if __name__ == "__main__":
    main()
