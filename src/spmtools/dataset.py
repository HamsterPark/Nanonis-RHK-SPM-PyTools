"""Discovery of measurement folders and creation of random test subsets."""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = frozenset({".sxm", ".sm4"})


def safe_filename(name: str) -> str:
    """Replace everything but ``[A-Za-z0-9._-]`` by underscores (empty -> ``"channel"``)."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return value.strip("_") or "channel"


def normalize_extensions(value: str | Iterable[str]) -> frozenset[str]:
    """Turn ``"sxm,sm4"`` or ``[".SXM", "sm4"]`` into ``frozenset({".sxm", ".sm4"})``."""
    parts = value.split(",") if isinstance(value, str) else list(value)
    exts = set()
    for part in parts:
        part = part.strip().lower()
        if not part:
            continue
        exts.add(part if part.startswith(".") else "." + part)
    return frozenset(exts)


def list_data_files(folder: Path, exts: frozenset[str] = SUPPORTED_EXTS) -> list[Path]:
    """Sorted data files located directly inside ``folder`` (empty if it is no directory)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def find_data_folders(
    root: Path,
    exts: frozenset[str] = SUPPORTED_EXTS,
    *,
    recursive: bool = True,
    skip_names: Iterable[str] = (),
) -> list[Path]:
    """Folders (``root`` included) that directly contain data files, sorted by path.

    With ``recursive=False`` only ``root`` itself is considered.  Directories whose name
    is in ``skip_names`` or starts with a dot are not descended into.
    """
    root = Path(root)
    if not recursive:
        return [root] if list_data_files(root, exts) else []
    skip = set(skip_names)
    folders: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip and not d.startswith("."))
        if any(Path(name).suffix.lower() in exts for name in filenames):
            folders.append(Path(dirpath))
    return sorted(folders)


def collect_folder_files(
    folders: Iterable[Path],
    exts: frozenset[str] = SUPPORTED_EXTS,
    limit: int = 0,
) -> list[tuple[Path, list[Path]]]:
    """Pair every folder with its (optionally truncated) list of data files."""
    result: list[tuple[Path, list[Path]]] = []
    for folder in folders:
        files = list_data_files(Path(folder), exts)
        if limit > 0:
            files = files[:limit]
        if files:
            result.append((Path(folder), files))
    return result


def relative_prefix(folder: Path, root: Path) -> str:
    """Name prefix derived from the path of ``folder`` below ``root``.

    ``root/2025/01/07`` becomes ``2025_01_07``; ``root`` itself becomes its own name.
    """
    folder, root = Path(folder), Path(root)
    try:
        parts = folder.resolve().relative_to(root.resolve()).parts
    except ValueError:
        parts = (folder.name,)
    parts = tuple(part for part in parts if part not in (".", ""))
    if not parts:
        parts = (folder.resolve().name or "root",)
    return "_".join(safe_filename(part) for part in parts)


def sample_subset(
    root: Path,
    out: Path,
    *,
    folders: int = 3,
    files_per_folder: int = 12,
    seed: int = 2025,
    exts: frozenset[str] = SUPPORTED_EXTS,
    manifest_name: str = "manifest.txt",
) -> list[Path]:
    """Copy a reproducible random subset of a dataset, keeping the folder structure.

    ``folders`` data folders are drawn at random, then up to ``files_per_folder`` files
    from each (``0`` = all).  A UTF-8 manifest with the copied relative paths is written
    to ``out / manifest_name``.  Returns the relative paths that were copied.
    """
    root, out = Path(root), Path(out)
    candidates = find_data_folders(root, exts)
    if not candidates:
        logger.warning("no folders with %s files under %s", sorted(exts), root)
        return []
    rng = random.Random(seed)
    chosen = rng.sample(candidates, k=min(folders, len(candidates)))
    copied: list[Path] = []
    for folder in sorted(chosen):
        files = list_data_files(folder, exts)
        rng.shuffle(files)
        if files_per_folder > 0:
            files = files[:files_per_folder]
        for src in sorted(files):
            rel = src.relative_to(root)
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / manifest_name
    manifest.write_text("".join(f"{rel.as_posix()}\n" for rel in copied), encoding="utf-8")
    logger.info("copied %d files into %s", len(copied), out)
    return copied


__all__ = [
    "SUPPORTED_EXTS",
    "collect_folder_files",
    "find_data_folders",
    "list_data_files",
    "normalize_extensions",
    "relative_prefix",
    "safe_filename",
    "sample_subset",
]
