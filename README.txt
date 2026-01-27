Nanonis-RHK-SPM-PyPreview

Python tools for fast preview mosaics of Nanonis SXM and RHK SM4 data.

Files:
- sxm_preview.py: Generate channel mosaics per folder.
- sxm_preview_parallel.py: Parallel wrapper for large datasets.
- make_test_subset.py: Create a random test subset of SXM/SM4 files.

Running instructions:
1) Create a test subset
   py -3 make_test_subset.py --root "C:\\data\\spm" --out "C:\\data\\spm_test_subset" --days 3 --files-per-day 12 --seed 2025
   (Optional) Restrict extensions: --extensions sm4

2) Generate mosaics for the test subset (recursive year/month/day)
   py -3 sxm_preview.py "C:\\data\\spm_test_subset" --recursive

3) Generate mosaics for a single day folder
   py -3 sxm_preview.py "C:\\data\\spm_test_subset\\2025\\01\\01"

4) Full dataset run with a central preview copy
   py -3 sxm_preview.py "C:\\data\\spm" --recursive --collect-dir "C:\\data\\spm_previews"

5) Parallel full dataset run (use 4 workers)
   py -3 sxm_preview_parallel.py "C:\\data\\spm" --recursive --collect-dir "C:\\data\\spm_previews" --workers 4

Output description:
- Output goes to a folder named "PreviewPy" inside each processed day folder.
- Each channel becomes a mosaic PNG; large folders are split into multiple images.
- If --collect-dir is set, all mosaics are copied into that folder with path-prefixed names.
  Example: "2022_02_24__Z_01.png"

Channel filtering rules (default):
- Skip all backward (Bwd) channels.
- Skip channels by name: Bias, X, Y, OC D1 Phase, OC D1 Amplitude, OC D1 Excitation, LI Demod 1 X/Y, LI Demod 2 X/Y.
- Skip channels that are constant or all-zero (configurable by tolerance).
- Z channel: generate two mosaics (raw Z and Z_line row-normalized).
- Current: keep only when a Freq Shift channel has signal; otherwise drop it.
- SM4: use only topography (Z) forward pages (Z and Z_line mosaics).

Common parameters:
- --max-tiles: Max tiles per mosaic (default 25, i.e., 5x5).
- --cols: Fixed mosaic columns (default 5).
- --tile-size: Tile size in pixels (default 256).
- --percentiles: Display clipping percentiles (default 1,99).
- --const-tol: Constant detection tolerance (default 0).
- --zero-tol: All-zero detection tolerance (default 0).
- --no-label: Disable file tail labels.
- --collect-dir: Copy all mosaics into one folder with prefixed names.
- --no-progress: Disable progress bar.

Performance and notes:
- Full dataset runs can take hours; run from a local terminal for long jobs.
- --collect-dir duplicates images and increases storage usage.
- Default mosaic size is capped at 5x5 tiles with grid lines between tiles; adjust with --max-tiles or --cols if needed.
- Parallel mode updates the progress bar per folder completion (coarser than per file).
