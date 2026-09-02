"""Regenerate the README figures from synthetic data.

Run from the repository root with the dev extras installed::

    python docs/make_figures.py

No measurement data is used: the ``.sxm`` files are generated on the fly (a stepped
surface with a hexagonal corrugation, adatoms and scan-line noise) and deleted again.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

from conftest import write_sxm  # noqa: E402
from spmtools.align import align_sxm_files  # noqa: E402
from spmtools.preview import PreviewConfig, run_preview  # noqa: E402

DOCS = REPO / "docs"
N = 128
PIXEL_M = 2.0e-10  # 0.2 nm per pixel -> 25.6 nm field of view


def surface(rng: np.random.Generator, *, shift=(0.0, 0.0), adatoms: int = 12) -> np.ndarray:
    """Height map in metres: two terraces, a lattice corrugation and a few adatoms."""
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    yy, xx = yy + shift[0], xx + shift[1]
    step = 0.23e-9 / (1.0 + np.exp(-(xx - 0.55 * N - 0.15 * yy) / 1.5))
    k = 2 * np.pi / 6.5
    lattice = 0.012e-9 * (
        np.cos(k * xx) + np.cos(k * (0.5 * xx + 0.866 * yy)) + np.cos(k * (0.5 * xx - 0.866 * yy))
    )
    height = step + lattice
    for _ in range(adatoms):
        cy, cx = rng.uniform(8, N - 8, size=2)
        height += 0.08e-9 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.2**2))
    return height


def noisy(height: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    line_noise = rng.normal(0, 0.004e-9, size=(N, 1))
    return height + line_noise + rng.normal(0, 0.003e-9, size=height.shape)


def write_scan(path: Path, height: np.ndarray, rng: np.random.Generator, *, ncafm: bool) -> None:
    current = 1e-10 * np.exp(-(height - height.min()) / 0.05e-9) + rng.normal(
        0, 2e-12, height.shape
    )
    freq = (-3.0 - 40e9 * (height - height.mean())) if ncafm else np.zeros_like(height)
    channels = [("Z", "m", "both"), ("Current", "A", "both"), ("OC_M1_Freq._Shift", "Hz", "fwd")]
    write_sxm(
        path,
        channels,
        {
            ("Z", "forward"): height,
            ("Z", "backward"): height[:, ::-1],
            ("Current", "forward"): current,
            ("Current", "backward"): current[:, ::-1],
            ("OC_M1_Freq._Shift", "forward"): freq,
        },
        scan_range=(N * PIXEL_M, N * PIXEL_M),
    )


def make_mosaic(workdir: Path) -> None:
    rng = np.random.default_rng(1)
    folder = workdir / "2026" / "03" / "14"
    folder.mkdir(parents=True)
    for i in range(10):
        base = surface(rng, shift=(rng.uniform(-20, 20), rng.uniform(-20, 20)))
        write_scan(folder / f"synthetic_{i + 1:04d}.sxm", noisy(base, rng), rng, ncafm=i % 2 == 1)
    config = PreviewConfig(tile_size=128, max_tiles=10, cols=5)
    summary = run_preview(workdir, config, recursive=True)
    for out in summary.outputs:
        if out.name in ("Z_01.png", "Z_line_01.png"):
            target = DOCS / f"mosaic_synthetic_{out.stem}.png"
            target.write_bytes(out.read_bytes())
            print("wrote", target)


def make_alignment(workdir: Path) -> None:
    rng = np.random.default_rng(7)
    base = surface(rng)
    shift = (3.3, -5.7)  # pixels of drift between the two frames
    frame_a = noisy(base, rng)
    frame_b = noisy(surface(np.random.default_rng(7), shift=shift), np.random.default_rng(8))
    write_scan(workdir / "synthetic_A.sxm", frame_a, rng, ncafm=False)
    write_scan(workdir / "synthetic_B.sxm", frame_b, rng, ncafm=False)
    _, outputs = align_sxm_files(
        workdir / "synthetic_A.sxm", workdir / "synthetic_B.sxm", outdir=workdir / "out"
    )
    target = DOCS / "alignment_overview_synthetic.png"
    target.write_bytes(outputs.overview.read_bytes())
    print("wrote", target)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        make_mosaic(Path(tmp) / "mosaic")
        (Path(tmp) / "align").mkdir()
        make_alignment(Path(tmp) / "align")


if __name__ == "__main__":
    main()
