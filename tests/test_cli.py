"""Tests for the console entry points."""

from __future__ import annotations

from importlib.metadata import entry_points

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from conftest import ramp, write_sxm
from spmtools import cli


def _scan(path, image, unit="m"):
    return write_sxm(
        path,
        [("Z", unit, "both")],
        {("Z", "forward"): image, ("Z", "backward"): image},
        scan_range=(1.6e-8, 1.6e-8),
    )


def _blobs(n=64):
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n))
    for cy, cx in [(20, 25), (40, 45), (45, 15)]:
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0**2))
    return img * 1e-9


@pytest.mark.parametrize(
    "build",
    [
        cli.build_preview_parser,
        cli.build_align_parser,
        cli.build_crop_parser,
        cli.build_sample_parser,
    ],
)
def test_help_and_version(build, capsys):
    parser = build()
    with pytest.raises(SystemExit) as info:
        parser.parse_args(["--help"])
    assert info.value.code == 0
    assert parser.prog in capsys.readouterr().out
    with pytest.raises(SystemExit) as info:
        parser.parse_args(["--version"])
    assert info.value.code == 0


def test_console_scripts_are_registered():
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts["spm-preview"] == "spmtools.cli:preview_main"
    assert scripts["spm-align-diff"] == "spmtools.cli:align_main"
    assert scripts["spm-crop-solid"] == "spmtools.cli:crop_main"
    assert scripts["spm-sample"] == "spmtools.cli:sample_main"


def test_parse_percentiles():
    assert cli.parse_percentiles("2,98") == (2.0, 98.0)
    for bad in ("5", "90,10", "-1,50", "1,2,3"):
        with pytest.raises(Exception, match="percentiles"):
            cli.parse_percentiles(bad)


def test_progress_bar_reaches_100(capsys):
    bar = cli.Progress(3)
    bar.advance()
    bar.advance(5)
    err = capsys.readouterr().err
    assert "3/3" in err and "100.0%" in err and err.endswith("\n")
    cli.Progress(0).advance()  # disabled, must not print
    assert capsys.readouterr().err == ""


def test_preview_cli_end_to_end(tmp_path, capsys):
    root = tmp_path / "data"
    day = root / "2025" / "02" / "03"
    day.mkdir(parents=True)
    _scan(day / "scan_0001.sxm", ramp(6, 8))
    _scan(day / "scan_0002.sxm", ramp(6, 8, 3))
    code = cli.preview_main(
        [str(root), "--recursive", "--tile-size", "16", "--no-progress", "--collect-dir", "all"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Total files: 2 (SXM=2)" in out and "Wrote 2 mosaics for 1 folders" in out
    assert (day / "PreviewPy" / "Z_01.png").is_file()
    assert (root / "all" / "2025_02_03__Z_line_01.png").is_file()
    assert not (root / "SM4_PreviewPy").exists()


def test_preview_cli_without_files_and_bad_root(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.preview_main([str(empty), "--no-progress"]) == 1
    assert "No .sxm or .sm4 files found." in capsys.readouterr().out
    with pytest.raises(SystemExit) as info:
        cli.preview_main([str(tmp_path / "missing")])
    assert info.value.code == 2


def test_align_cli(tmp_path, capsys, monkeypatch):
    ref = _blobs()
    mov = ndimage.shift(ref, (1.0, -1.5), order=3, mode="nearest")
    a = _scan(tmp_path / "scan_0001.sxm", ref)
    b = _scan(tmp_path / "scan_0002.sxm", mov)
    out = tmp_path / "out"
    assert cli.align_main([str(a), str(b), "--outdir", str(out), "--upsample", "10"]) == 0
    text = capsys.readouterr().out
    assert "shift: dx=" in text and (out / "z_fwd_0001-0002.npy").is_file()

    monkeypatch.chdir(tmp_path)
    assert cli.align_main(["--upsample", "10", "--outdir", str(out / "auto")]) == 0
    assert (out / "auto" / "z_fwd_alignment_overview_0001_0002.png").is_file()

    with pytest.raises(SystemExit) as info:
        cli.align_main([str(a), str(b), "--channel", "Nope", "--outdir", str(out)])
    assert info.value.code == 1
    assert "no channel matches" in capsys.readouterr().err

    with pytest.raises(SystemExit) as info:
        cli.align_main([str(a)])
    assert info.value.code == 2


def test_crop_cli(tmp_path, capsys):
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 256, size=(160, 100, 3)).astype(np.uint8)
    arr[:40] = 128  # two full 20-line blocks of solid border
    Image.fromarray(arr).save(images / "export.png")
    before = (images / "export.png").read_bytes()

    assert cli.crop_main([str(images), "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "DRY RUN" in text and "[DRY] Crop export.png: 100x160 -> 100x120" in text
    assert (images / "export.png").read_bytes() == before

    out = tmp_path / "cropped"
    assert cli.crop_main([str(images), "--out-dir", str(out), "-v"]) == 0
    assert "Images cropped: 1" in capsys.readouterr().out
    with Image.open(out / "export.png") as img:
        assert img.size == (100, 120)
    assert (images / "export.png").read_bytes() == before

    empty = tmp_path / "nothing"
    empty.mkdir()
    assert cli.crop_main([str(empty)]) == 1


def test_sample_cli(tmp_path, capsys):
    root = tmp_path / "root"
    for day in ("01", "02"):
        for i in range(3):
            _scan(root / "2026" / "01" / day / f"scan_{i:04d}.sxm", ramp(2, 2))
    out = tmp_path / "subset"
    args = ["--root", str(root), "--out", str(out), "--folders", "1", "--files-per-folder", "2"]
    assert cli.sample_main(args) == 0
    assert "Copied 2 files" in capsys.readouterr().out
    assert len((out / "manifest.txt").read_text(encoding="utf-8").splitlines()) == 2
    assert (
        cli.sample_main(["--root", str(out), "--out", str(tmp_path / "x"), "--extensions", "sm4"])
        == 1
    )
