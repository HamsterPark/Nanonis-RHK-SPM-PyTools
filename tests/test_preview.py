"""Tests for dataset discovery and the mosaic preview pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import ramp, write_sm4, write_sxm
from spmtools import dataset, preview
from spmtools.preview import PreviewConfig

# --------------------------------------------------------------------------- dataset


def test_normalize_extensions():
    assert dataset.normalize_extensions("sxm, SM4,,") == {".sxm", ".sm4"}
    assert dataset.normalize_extensions([".SXM"]) == {".sxm"}


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_find_data_folders_recursive_at_any_depth(tmp_path):
    root = tmp_path / "data"
    _touch(root / "a.sxm")
    _touch(root / "2025" / "01" / "07" / "b.sxm")
    _touch(root / "deep" / "er" / "dir" / "c.SM4")
    _touch(root / "empty" / "notes.txt")
    _touch(root / "PreviewPy" / "stray.sxm")
    _touch(root / ".hidden" / "d.sxm")
    found = dataset.find_data_folders(root, skip_names={"PreviewPy"})
    assert found == sorted([root, root / "2025" / "01" / "07", root / "deep" / "er" / "dir"])
    assert dataset.find_data_folders(root, recursive=False) == [root]
    assert dataset.find_data_folders(root / "empty", recursive=False) == []


def test_collect_folder_files_applies_limit(tmp_path):
    for i in range(4):
        _touch(tmp_path / f"f{i}.sxm")
    _touch(tmp_path / "ignored.txt")
    pairs = dataset.collect_folder_files([tmp_path, tmp_path / "missing_files_dir"], limit=2)
    assert len(pairs) == 1
    assert [p.name for p in pairs[0][1]] == ["f0.sxm", "f1.sxm"]


def test_relative_prefix(tmp_path):
    root = tmp_path / "root"
    assert dataset.relative_prefix(root / "2025" / "01" / "07", root) == "2025_01_07"
    assert dataset.relative_prefix(root, root) == "root"
    assert dataset.relative_prefix(tmp_path / "elsewhere", root) == "elsewhere"
    assert dataset.safe_filename(" OC D1 Freq. Shift ") == "OC_D1_Freq._Shift"
    assert dataset.safe_filename("///") == "channel"


def test_sample_subset_is_reproducible_and_utf8(tmp_path):
    root = tmp_path / "root"
    for day in ("01", "02", "03"):
        for i in range(4):
            _touch(root / "2025" / "01" / day / f"scan{i}.sxm")
    _touch(root / "2025" / "01" / "01" / "样品_0009.sxm")
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    copied1 = dataset.sample_subset(root, out1, folders=2, files_per_folder=2, seed=7)
    copied2 = dataset.sample_subset(root, out2, folders=2, files_per_folder=2, seed=7)
    assert copied1 == copied2 and len(copied1) == 4
    assert all((out1 / rel).is_file() for rel in copied1)
    manifest = (out1 / "manifest.txt").read_text(encoding="utf-8").splitlines()
    assert manifest == [rel.as_posix() for rel in copied1]
    everything = dataset.sample_subset(root, tmp_path / "all", folders=9, files_per_folder=0)
    assert len(everything) == 13 and any("样品" in rel.name for rel in everything)
    assert dataset.sample_subset(tmp_path / "nothing", tmp_path / "x") == []


# --------------------------------------------------------------------------- pure helpers


def test_canonical_name_and_tail_id():
    assert preview.canonical_name(" OC D1 Freq. Shift ") == "oc_d1_freq_shift"
    assert preview.extract_tail_id("Ag111_0123", 5) == "00123"
    assert preview.extract_tail_id("scan1234567", 5) == "34567"
    assert preview.extract_tail_id("noDigits", 5) == "noDigits"


def test_has_signal():
    assert not preview.has_signal(None)
    assert not preview.has_signal(np.full((3, 3), np.nan))
    assert not preview.has_signal(np.full((3, 3), 2.0))
    assert not preview.has_signal(np.array([[0.0, 1e-9]]), zero_tol=1e-6)
    assert not preview.has_signal(np.array([[0.0, 0.5]]), const_tol=1.0)
    assert preview.has_signal(np.array([[0.0, 0.5]]))


def test_line_normalize_removes_row_offsets():
    data = ramp(4, 5) + np.array([[0.0], [100.0], [np.nan], [-7.0]])
    data[2, :] = np.nan
    out = preview.line_normalize(data)
    assert np.allclose(np.nanmedian(out[[0, 1, 3]], axis=1), 0.0)
    assert np.isnan(out[2]).all()


def test_normalize_for_display():
    data = np.linspace(0, 100, 101).reshape(1, -1)
    out = preview.normalize_for_display(data, (10, 90))
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0
    assert (out[0, :11] == 0.0).all() and (out[0, 90:] == 1.0).all()
    assert (preview.normalize_for_display(np.full((2, 2), 3.0), (1, 99)) == 0).all()
    assert (preview.normalize_for_display(np.full((2, 2), np.nan), (1, 99)) == 0).all()


def test_pick_colormap_is_stable():
    assert preview.pick_colormap("z") == "copper"
    assert preview.pick_colormap("z_line") == "cividis"
    first = preview.pick_colormap("li_demod_3_x")
    assert first in preview.CMAP_CYCLE and first == preview.pick_colormap("li_demod_3_x")


def test_render_tile_shape_and_label():
    tile = preview.render_tile(ramp(6, 8), "viridis", 64, label="00042")
    assert tile.size == (64, 64) and tile.mode == "RGB"
    plain = preview.render_tile(ramp(6, 8), "viridis", 64, label="00042", add_label=False)
    assert plain.size == (64, 64)
    assert np.array(tile).sum() != np.array(plain).sum()  # the label changed pixels


def test_mosaic_builder_splits_into_parts(tmp_path):
    config = PreviewConfig(tile_size=16, max_tiles=4, cols=2)
    builder = preview.MosaicBuilder("Z line", tmp_path, config)
    for _ in range(5):
        builder.add(Image.new("RGB", (16, 16), (255, 0, 0)))
    outputs = builder.finalize()
    assert [p.name for p in outputs] == ["Z_line_01.png", "Z_line_02.png"]
    assert Image.open(outputs[0]).size == (32, 32)
    assert Image.open(outputs[1]).size == (32, 16)
    assert builder.finalize() == outputs  # nothing left to flush


def test_is_backward_channel():
    assert preview.is_backward_channel("Z_bwd", "both")
    assert preview.is_backward_channel("Z", "bwd")
    assert not preview.is_backward_channel("Z", "both")


# --------------------------------------------------------------------------- sxm/sm4 policy

CHANNELS = [
    ("Z", "m", "both"),
    ("Current", "A", "both"),
    ("Bias", "V", "both"),
    ("OC_M1_Freq._Shift", "Hz", "fwd"),
]


def _sxm(path: Path, *, z_signal: bool = True, freq_signal: bool = False) -> Path:
    ny, nx = 6, 8
    z = ramp(ny, nx) if z_signal else np.full((ny, nx), 1.0)
    freq = ramp(ny, nx, -30) if freq_signal else np.zeros((ny, nx))
    images = {
        ("Z", "forward"): z,
        ("Z", "backward"): z,
        ("Current", "forward"): ramp(ny, nx, 5),
        ("Current", "backward"): ramp(ny, nx, 5),
        ("Bias", "forward"): ramp(ny, nx),
        ("Bias", "backward"): ramp(ny, nx),
        ("OC_M1_Freq._Shift", "forward"): freq,
    }
    return write_sxm(path, CHANNELS, images)


def test_sxm_policy_stm_scan_drops_current_and_dead_freq_shift(tmp_path):
    keys = [k for k, _ in preview.iter_sxm_tiles(_sxm(tmp_path / "s_0001.sxm"), PreviewConfig())]
    assert keys == ["Z", "Z_line"]


def test_sxm_policy_ncafm_scan_keeps_current_and_freq_shift(tmp_path):
    path = _sxm(tmp_path / "s_0002.sxm", freq_signal=True)
    keys = [k for k, _ in preview.iter_sxm_tiles(path, PreviewConfig())]
    assert keys == ["Z", "Z_line", "Current", "OC_M1_Freq._Shift"]


def test_sxm_policy_without_z_signal_keeps_current(tmp_path):
    path = _sxm(tmp_path / "s_0003.sxm", z_signal=False)
    keys = [k for k, _ in preview.iter_sxm_tiles(path, PreviewConfig())]
    assert keys == ["Current"]
    # Z and frequency shift are gated by their own signal test even with skip_constant=False
    keys = [k for k, _ in preview.iter_sxm_tiles(path, PreviewConfig(skip_constant=False))]
    assert keys == ["Current"]


def test_sm4_tiles(tmp_path, caplog):
    path = write_sm4(tmp_path / "t_0007.sm4", [{"data": ramp(4, 6).astype(np.int32)}])
    keys = [k for k, _ in preview.iter_sm4_tiles(path, PreviewConfig(tile_size=32))]
    assert keys == ["Z", "Z_line"]
    no_topo = write_sm4(tmp_path / "c_0008.sm4", [{"data": ramp(4, 6), "page_type": 2}])
    with caplog.at_level(logging.WARNING, logger="spmtools.preview"):
        assert list(preview.iter_sm4_tiles(no_topo, PreviewConfig())) == []
    assert "no topography image page" in caplog.text


# --------------------------------------------------------------------------- folders


def _dataset(root: Path) -> None:
    day = root / "2025" / "01" / "07"
    day.mkdir(parents=True)
    _sxm(day / "scan_0001.sxm")
    _sxm(day / "scan_0002.sxm", freq_signal=True)
    write_sm4(day / "scan_0003.sm4", [{"data": ramp(4, 6).astype(np.int32)}])
    (day / "broken_0004.sxm").write_bytes(b":NANONIS_VERSION:\n2\n")
    other = root / "misc"
    other.mkdir()
    _sxm(other / "x_0005.sxm")


def test_process_folder_end_to_end(tmp_path, caplog):
    _dataset(tmp_path)
    day = tmp_path / "2025" / "01" / "07"
    config = PreviewConfig(tile_size=32, max_tiles=2, cols=2)
    with caplog.at_level(logging.WARNING, logger="spmtools.preview"):
        result = preview.process_folder(day, dataset.list_data_files(day), config)
    assert result.n_files == 4 and result.n_failed == 1
    assert "broken_0004.sxm" in caplog.text
    assert result.counts == {"Z": 3, "Z_line": 3, "Current": 1, "OC_M1_Freq._Shift": 1}
    names = sorted(p.name for p in result.outputs)
    assert names == [
        "Current_01.png",
        "OC_M1_Freq._Shift_01.png",
        "Z_01.png",
        "Z_02.png",
        "Z_line_01.png",
        "Z_line_02.png",
    ]
    assert all(p.parent == day / "PreviewPy" for p in result.outputs)


def test_run_preview_recursive_with_collect_dirs(tmp_path):
    _dataset(tmp_path)
    collect, sm4_collect = tmp_path / "all_previews", tmp_path / "SM4_PreviewPy"
    seen: list[int] = []
    summary = preview.run_preview(
        tmp_path,
        PreviewConfig(tile_size=32),
        recursive=True,
        collect_dir=collect,
        sm4_collect_dir=sm4_collect,
        on_progress=seen.append,
    )
    assert summary.type_counts == {".sxm": 4, ".sm4": 1}
    assert summary.n_files == 5 and summary.n_failed == 1 and sum(seen) == 5
    assert [r.folder.name for r in summary.folders] == ["07", "misc"]
    assert sorted(p.name for p in collect.iterdir()) == [
        "2025_01_07__Current_01.png",
        "2025_01_07__OC_M1_Freq._Shift_01.png",
        "2025_01_07__Z_01.png",
        "2025_01_07__Z_line_01.png",
        "misc__Z_01.png",
        "misc__Z_line_01.png",
    ]
    assert sorted(p.name for p in sm4_collect.iterdir()) == [
        "2025_01_07__Z_01.png",
        "2025_01_07__Z_line_01.png",
    ]
    # a second run must not pick up the PreviewPy folders as data folders
    again = preview.run_preview(tmp_path, PreviewConfig(tile_size=32), recursive=True)
    assert [r.folder.name for r in again.folders] == ["07", "misc"]


def test_run_preview_skips_sm4_folder_without_sm4_files(tmp_path, caplog):
    _sxm(tmp_path / "only_0001.sxm")
    sm4_collect = tmp_path / "SM4_PreviewPy"
    summary = preview.run_preview(
        tmp_path, PreviewConfig(tile_size=16), sm4_collect_dir=sm4_collect
    )
    assert summary.n_files == 1 and not sm4_collect.exists()
    with caplog.at_level(logging.WARNING, logger="spmtools.preview"):
        empty = preview.run_preview(tmp_path / "nowhere", PreviewConfig())
    assert empty.n_files == 0 and "no .sm4/.sxm files found" in caplog.text


def test_run_preview_with_worker_processes_matches_sequential(tmp_path):
    _dataset(tmp_path)
    config = PreviewConfig(tile_size=16)
    parallel = preview.run_preview(tmp_path, config, recursive=True, workers=2)
    assert parallel.n_files == 5 and parallel.n_failed == 1
    assert sorted(p.name for p in parallel.outputs) == sorted(
        p.name for p in preview.run_preview(tmp_path, config, recursive=True).outputs
    )


@pytest.mark.parametrize("cols", [0, 3])
def test_auto_and_fixed_columns(tmp_path, cols):
    config = PreviewConfig(tile_size=8, max_tiles=0, cols=cols)
    builder = preview.MosaicBuilder("Z", tmp_path, config)
    for _ in range(7):
        builder.add(Image.new("RGB", (8, 8)))
    (out,) = builder.finalize()
    expected = (24, 24) if cols == 0 else (24, 24)  # 7 tiles -> 3 columns either way
    assert Image.open(out).size == expected
