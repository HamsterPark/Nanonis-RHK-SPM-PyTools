#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 对齐两个 Nanonis .sxm 文件的指定通道（默认 Z 形貌 fwd）并作差
================================================================================

功能概述
--------
1. 自己解析 Nanonis 的 .sxm 二进制格式（无需第三方库），读取指定通道；
2. 用「相位互相关 + 上采样 DFT」做亚像素级图像配准
   （Guizar-Sicairos, Thurman & Fienup, Optics Letters 33, 156 (2008)，
     即 skimage.registration.phase_cross_correlation）；
3. 把 B 对齐到 A 之后，分别生成 A−B、B−A 两张差分图；
4. 额外生成一张「对齐效果示意图」，直观对比对齐前/后的残差。

差分图统一使用最常见的「红-白-蓝」发散型 colormap（默认 RdBu_r：
蓝=负、白=零、红=正；若想严格按红→白→蓝即红=负，把 --cmap 改成 RdBu）。

.sxm 格式要点（本脚本据此解析）
-------------------------------
* 文件 = ASCII 文件头 + 分隔符 b"\\x1a\\x04" + 大端二进制数据；
* 文件头是若干 ":TAG:" 键值对，以 ":SCANIT_END:" 结束；
* :SCANIT_TYPE: 给出数据类型与字节序（FLOAT + MSBFIRST = 大端 float32）；
* :SCAN_PIXELS: 给出 nx(每行像素) 与 ny(行数)；
* :DATA_INFO: 是一张表，按顺序列出每个通道的名字、单位、扫描方向(both/fwd/bwd)；
* 二进制区按 :DATA_INFO: 的通道顺序排列，每个 "both" 通道先存 forward 再存 backward，
  每幅图为 ny×nx 个大端 float32；backward 图按列左右翻转即得真实空间图像。

用法
----
    # 不带参数：自动在脚本所在目录找两个 .sxm 文件（按文件名排序，第 1 个作 A，第 2 个作 B）
    python align_sxm_freqshift.py

    # 显式指定文件 / 通道 / 输出目录 / 上采样倍数 / colormap
    python align_sxm_freqshift.py A.sxm B.sxm --channel "Freq" --upsample 100 --cmap RdBu_r

作者注：脚本对 A、B 的选择只影响差分的正负号；两个极性（A−B 与 B−A）都会输出，
        所以无所谓哪个文件当 A。
================================================================================
"""

import os
import re
import glob
import argparse

# ---------------------------------------------------------------------------
# 兼容性补丁：规避部分 Windows 机器上 WMI(Winmgmt) 查询挂死导致导入卡死的问题
# 现象：某些 Windows 上 platform._wmi_query() 会无限阻塞；而 numpy 在导入时会经由
#       platform.machine() → win32_ver() → _wmi_query() 间接调用它，
#       从而导致 `import numpy / scipy / skimage` 整个卡死（与库本身无关）。
# 处理：在导入 numpy 之前禁用 platform 的 WMI 路径，改走 sys.getwindowsversion()
#       的快速回退分支；WMI 正常的机器上此补丁亦无副作用。
#       必须位于所有 numpy/scipy/matplotlib 导入之前。
# ---------------------------------------------------------------------------
import platform as _platform
_platform._wmi = None                      # platform 自带的禁用开关
def _disable_wmi(*_a, **_k):               # 双保险：让查询函数立即抛异常以触发回退
    raise OSError("WMI disabled to avoid Winmgmt hang")
_platform._wmi_query = _disable_wmi

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 用非交互后端，直接把图写成文件，不弹窗
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.registration import phase_cross_correlation

# 让 matplotlib 能正确显示中文标题（Windows 自带「微软雅黑/黑体」）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


# ==============================================================================
# 第 1 部分：.sxm 文件读取
# ==============================================================================
def read_sxm(path):
    """
    解析一个 Nanonis .sxm 文件。

    返回一个 dict，包含：
        'header'      : {TAG: [原始文本行, ...]}，所有头部标签
        'nx', 'ny'    : 每行像素数、行数
        'pix_x_nm'    : x 方向像素尺寸（纳米）
        'pix_y_nm'    : y 方向像素尺寸（纳米）
        'data_info'   : [{'name':..., 'unit':..., 'direction':...}, ...] 通道表
        'channels'    : {通道名: {'forward': 2D数组, 'backward': 2D数组(若有)}}
    """
    with open(path, "rb") as f:
        raw = f.read()

    # ---- 1.1 找到「头部/数据」分隔符 b"\x1a\x04" ----
    marker = b"\x1a\x04"
    idx = raw.find(marker)
    if idx == -1:
        raise ValueError(f"在 {path} 中找不到 sxm 数据分隔符 \\x1a\\x04，文件可能损坏")
    header_text = raw[:idx].decode("latin-1", errors="replace")
    data_bytes = raw[idx + len(marker):]

    # ---- 1.2 把 ASCII 头部解析成 {TAG: [行, ...]} ----
    # 规则：以 ':' 开头且以 ':' 结尾的整行是一个标签名；其后直到下一个标签的行都是它的值。
    header = {}
    current = None
    for line in header_text.split("\n"):
        line = line.rstrip("\r")
        stripped = line.strip()
        if stripped.startswith(":") and stripped.endswith(":") and len(stripped) > 2:
            current = stripped[1:-1]          # 去掉首尾冒号即标签名
            header[current] = []
        elif current is not None:
            header[current].append(line)

    # ---- 1.3 数据类型与字节序：FLOAT + MSBFIRST → 大端 float32 ----
    type_tokens = " ".join(header.get("SCANIT_TYPE", [""])).split()
    # 形如 ["FLOAT", "MSBFIRST"]
    if len(type_tokens) < 2 or type_tokens[0].upper() != "FLOAT":
        raise ValueError(f"暂只支持 FLOAT 类型的 sxm，实际为 {type_tokens}")
    byteorder = ">" if type_tokens[1].upper() == "MSBFIRST" else "<"
    dtype = np.dtype(byteorder + "f4")  # 4 字节单精度浮点

    # ---- 1.4 图像尺寸 nx(每行像素)、ny(行数) ----
    nx, ny = (int(v) for v in " ".join(header["SCAN_PIXELS"]).split())

    # ---- 1.5 扫描范围 → 像素物理尺寸（纳米）----
    rng = [float(v) for v in " ".join(header["SCAN_RANGE"]).split()]  # [x_range, y_range] 单位 m
    pix_x_nm = rng[0] / nx * 1e9
    pix_y_nm = rng[1] / ny * 1e9

    # ---- 1.6 解析 :DATA_INFO: 通道表，确定通道顺序与各自的扫描方向 ----
    data_info = []
    for row in header.get("DATA_INFO", []):
        cells = [c for c in row.split("\t") if c != ""]
        if not cells:
            continue
        # 表头行以 "Channel" 开头，跳过
        if cells[0].strip().lower() == "channel":
            continue
        # 正常行：Channel  Name  Unit  Direction  Calibration  Offset
        if len(cells) >= 4:
            data_info.append({
                "name": cells[1].strip(),
                "unit": cells[2].strip(),
                "direction": cells[3].strip().lower(),  # both / fwd / bwd
            })

    # ---- 1.7 按通道顺序计算需要读取多少幅图，并切分二进制数据 ----
    # both → 2 幅(forward, backward)；fwd → 1 幅(forward)；bwd → 1 幅(backward)
    plane = ny * nx                        # 单幅图的像素数
    # 整体读成一维大端 float32，再按图切片
    flat = np.frombuffer(data_bytes, dtype=dtype)
    needed_images = sum(2 if ch["direction"] == "both" else 1 for ch in data_info)
    if flat.size < needed_images * plane:
        raise ValueError(
            f"{path} 数据长度不足：需要 {needed_images*plane} 个浮点，实际只有 {flat.size} 个")
    flat = flat[: needed_images * plane].astype(np.float64)  # 转 float64 便于后续计算

    channels = {}
    cursor = 0
    for ch in data_info:
        name = ch["name"]
        entry = {}
        if ch["direction"] in ("both", "fwd"):
            fwd = flat[cursor:cursor + plane].reshape(ny, nx)
            entry["forward"] = fwd
            cursor += plane
        if ch["direction"] in ("both", "bwd"):
            bwd = flat[cursor:cursor + plane].reshape(ny, nx)
            # backward 是从右往左扫的，按列左右翻转才与 forward 对齐到同一真实空间
            entry["backward"] = np.fliplr(bwd)
            cursor += plane
        channels[name] = entry

    return {
        "header": header,
        "nx": nx, "ny": ny,
        "pix_x_nm": pix_x_nm, "pix_y_nm": pix_y_nm,
        "data_info": data_info,
        "channels": channels,
        "path": path,
    }


def find_channel_image(sxm, keyword="Freq", direction="forward"):
    """
    在已解析的 sxm 里，按关键字模糊匹配通道名，取出指定方向(forward/backward)的图。

    例如 keyword="Freq" 会匹配 "OC_M1_Freq._Shift"。
    返回 (二维数组, 通道全名, 单位)。
    """
    kws = keyword.lower().split()  # 支持用空格写多个关键字，全部命中才算
    # 先按整名精确匹配（如 'Z' 精确选中 Z 通道，避免被其它含该字母的通道误匹配），
    # 没有精确命中时再退回到「所有关键字均为子串」的模糊匹配
    exact = [ch["name"] for ch in sxm["data_info"] if ch["name"].lower() == keyword.lower()]
    if exact:
        matches = exact
    else:
        matches = [ch["name"] for ch in sxm["data_info"]
                   if all(k in ch["name"].lower() for k in kws)]
    if not matches:
        available = ", ".join(ch["name"] for ch in sxm["data_info"])
        raise ValueError(f"在 {sxm['path']} 找不到含关键字 '{keyword}' 的通道。可用通道：{available}")
    name = matches[0]  # 取第一个命中的
    entry = sxm["channels"][name]
    if direction not in entry:
        raise ValueError(f"通道 '{name}' 没有 {direction} 方向数据（只有 {list(entry)}）")
    unit = next((ch["unit"] for ch in sxm["data_info"] if ch["name"] == name), "")
    return entry[direction], name, unit


# ==============================================================================
# 第 2 部分：亚像素对齐（相位互相关 + 上采样 DFT）
# ==============================================================================
def _prepare_for_registration(img):
    """配准前预处理：把 NaN/Inf 用均值填掉，去直流，再乘汉宁窗以抑制边缘泄漏。"""
    finite = np.isfinite(img)
    fill = img[finite].mean() if finite.any() else 0.0
    clean = np.where(finite, img, fill)
    clean = clean - clean.mean()                       # 去直流分量
    win = np.outer(np.hanning(clean.shape[0]),         # 2D 可分离汉宁窗
                   np.hanning(clean.shape[1]))
    return clean * win


def estimate_shift(ref, mov, upsample=100):
    """
    估计把 mov 对齐到 ref 所需的平移量（亚像素）。

    返回 (shift, error)：
        shift = (drow, dcol)，即 ndimage.shift(mov, shift) 后即可与 ref 对齐；
        error = 配准误差（越小越好）。
    upsample=100 表示亚像素精度可达 1/100 像素。

    注意 normalization=None（标准归一化互相关）而非默认的 'phase'：
        相位归一化会把频谱白化，对 SPM 常见的「行噪声/周期性条纹」极其敏感，
        容易锁到由条纹噪声造成的虚假大位移峰（实测会给出几十像素的错误平移）；
        对于「差不多」的两张图，标准互相关更稳健，会锁定真正的特征峰。
    """
    ref_p = _prepare_for_registration(ref)
    mov_p = _prepare_for_registration(mov)
    shift, error, _diffphase = phase_cross_correlation(
        ref_p, mov_p, upsample_factor=upsample, normalization=None)
    return np.asarray(shift, dtype=float), float(error)


def apply_shift(mov, shift):
    """
    用三次样条插值把 mov 平移 shift=(drow,dcol)（可含小数，实现亚像素平移）。

    返回 (mov_aligned, valid_mask)：
        mov_aligned : 平移后的图（边缘用最近邻外推填充，保证处处有值）；
        valid_mask  : 布尔数组，True 表示该像素来自真实数据、未被边缘外推污染；
                      主流程据此把图裁剪到完全重合的公共区域（切掉非重合边缘）。
    """
    mov_aligned = ndimage.shift(mov, shift=shift, order=3, mode="nearest")
    # 用一张全 1 图做同样平移：平移进来的边缘会 <1，据此判定有效区域
    ones = np.ones_like(mov, dtype=float)
    valid = ndimage.shift(ones, shift=shift, order=1, mode="constant", cval=0.0) > 0.999
    return mov_aligned, valid


# ==============================================================================
# 第 3 部分：作图
# ==============================================================================
def _robust_limits(img, p=2.0):
    """用百分位数取稳健的显示上下限，避免极端值压低对比度。"""
    lo = np.nanpercentile(img, p)
    hi = np.nanpercentile(img, 100 - p)
    return lo, hi


def _add_scalebar(ax, pix_nm, bar_nm=5.0, color="k"):
    """在图右下角画一个已知长度(bar_nm 纳米)的比例尺。"""
    n_pix = bar_nm / pix_nm                       # 比例尺对应多少像素
    x0 = ax.get_xlim()[1] * 0.95 - n_pix          # 右下角留 5% 边距
    y0 = ax.get_ylim()[0] * 0.93 if ax.get_ylim()[0] > ax.get_ylim()[1] \
        else ax.get_ylim()[1] * 0.93
    # imshow 默认 origin='upper'，y 轴是反的，这里直接按像素坐标放在底部
    ymax = ax.get_ylim()[0]
    ax.plot([x0, x0 + n_pix], [ymax * 0.92, ymax * 0.92], color=color, lw=3)
    ax.text(x0 + n_pix / 2, ymax * 0.88, f"{bar_nm:g} nm",
            color=color, ha="center", va="bottom", fontsize=9)


def save_difference(diff, vmax, unit, quantity, title, out_png, cmap, pix_nm):
    """保存单张差分图（发散型 colormap，0 居中，对称色标）。"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(diff, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper",
                   interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    _add_scalebar(ax, pix_nm, bar_nm=5.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(f"Δ {quantity} ({unit})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_overview(A, B, B_aligned, diff_before, diff_after,
                  vmax, shift, error, rms_before, rms_after,
                  pix_x_nm, pix_y_nm, unit, quantity, name_A, name_B, out_png, cmap):
    """
    保存「对齐效果示意图」：上排原始图，下排差分图（对齐前 vs 对齐后）+ 残差直方图。
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)

    # —— 上排：原始 A、原始 B、对齐后的 B（统一灰度，稳健色标）——
    glo, ghi = _robust_limits(np.concatenate([A.ravel(), B.ravel()]))
    for ax, img, ttl in [
        (axes[0, 0], A, f"A 原始\n{name_A}"),
        (axes[0, 1], B, f"B 原始\n{name_B}"),
        (axes[0, 2], B_aligned, "B 对齐后"),
    ]:
        im = ax.imshow(img, cmap="afmhot", vmin=glo, vmax=ghi,
                       origin="upper", interpolation="nearest")
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.012, pad=0.01,
                 label=f"{quantity} ({unit})")

    # —— 下排左、中：对齐前 / 对齐后差分（同一对称色标，便于对比）——
    for ax, img, ttl in [
        (axes[1, 0], diff_before, f"对齐前差分 A−B\nRMS = {rms_before:.4g} {unit}"),
        (axes[1, 1], diff_after, f"对齐后差分 A−B\nRMS = {rms_after:.4g} {unit}"),
    ]:
        im2 = ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax,
                        origin="upper", interpolation="nearest")
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im2, ax=axes[1, :2].tolist(), fraction=0.012, pad=0.01,
                 label=f"Δ {quantity} ({unit})")

    # —— 下排右：残差直方图（对齐前 vs 对齐后），直观看到分布是否变窄 ——
    ax = axes[1, 2]
    db = diff_before[np.isfinite(diff_before)].ravel()
    da = diff_after[np.isfinite(diff_after)].ravel()
    hist_range = (-vmax * 1.5, vmax * 1.5)
    ax.hist(db, bins=200, range=hist_range, alpha=0.55, color="gray", label="对齐前")
    ax.hist(da, bins=200, range=hist_range, alpha=0.55, color="#c0392b", label="对齐后")
    ax.set_title("残差分布对比", fontsize=10)
    ax.set_xlabel(f"Δ {quantity} ({unit})")
    ax.set_ylabel("像素数")
    ax.legend(fontsize=9)

    # —— 总标题：写明检测到的平移量与改善程度 ——
    dr_nm = shift[0] * pix_y_nm
    dc_nm = shift[1] * pix_x_nm
    improve = (1 - rms_after / rms_before) * 100 if rms_before > 0 else 0.0
    fig.suptitle(
        f"亚像素对齐（互相关配准）：检测到平移 Δx={shift[1]:+.3f} px ({dc_nm:+.3f} nm), "
        f"Δy={shift[0]:+.3f} px ({dr_nm:+.3f} nm)   |   "
        f"残差 RMS {rms_before:.4g} → {rms_after:.4g} {unit}（改善 {improve:.1f}%）",
        fontsize=12)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 第 4 部分：主流程
# ==============================================================================
def short_id(path):
    """从文件名里抽取末尾数字作为简短标识（如 'Ag(111)1098' → '1098'）；取不到就用文件名主干。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.findall(r"\d+", stem)
    return m[-1] if m else stem


def main():
    parser = argparse.ArgumentParser(
        description="对齐两个 .sxm 文件的指定通道(默认 Z 形貌 fwd)并作差")
    parser.add_argument("files", nargs="*", help="两个 .sxm 文件（A B）；不填则自动在脚本目录查找")
    parser.add_argument("--channel", default="Z",
                        help="通道名/关键字：先按整名精确匹配，再按子串模糊匹配；"
                             "默认 'Z'(形貌)；频移通道用 --channel 'Freq Shift'")
    parser.add_argument("--direction", default="forward",
                        choices=["forward", "backward"], help="扫描方向，默认 forward(fwd)")
    parser.add_argument("--upsample", type=int, default=100, help="亚像素上采样倍数，默认 100")
    parser.add_argument("--cmap", default="RdBu",
                        help="差分图 colormap，默认 RdBu(红-白-蓝：红=负/白=零/蓝=正)；反向(蓝-白-红)用 RdBu_r")
    parser.add_argument("--outdir", default=None, help="输出目录，默认与第一个文件同目录")
    parser.add_argument("--clip", type=float, default=99.0,
                        help="差分图色标用的百分位(抑制离群点)，默认 99")
    parser.add_argument("--smooth-sigma", type=float, default=0.0,
                        help="对差分图做高斯平滑的 sigma(像素)，默认 0=不平滑；"
                             "freq shift 行噪声较强时可设 1~1.5 压噪显结构")
    args = parser.parse_args()

    # ---- 4.1 确定输入文件 ----
    files = args.files
    if len(files) == 0:
        here = os.path.dirname(os.path.abspath(__file__))
        files = sorted(glob.glob(os.path.join(here, "*.sxm")))
    if len(files) != 2:
        raise SystemExit(f"需要正好 2 个 .sxm 文件，实际得到 {len(files)} 个：{files}")
    path_A, path_B = files[0], files[1]
    outdir = args.outdir or os.path.dirname(os.path.abspath(path_A))
    os.makedirs(outdir, exist_ok=True)

    print(f"[读取] A = {path_A}")
    print(f"[读取] B = {path_B}")
    sxm_A = read_sxm(path_A)
    sxm_B = read_sxm(path_B)

    # ---- 4.2 取出目标通道（默认 Z fwd 形貌）----
    A, ch_name, unit = find_channel_image(sxm_A, args.channel, args.direction)
    B, _, _ = find_channel_image(sxm_B, args.channel, args.direction)

    # 单位自动换算：米(m)数值太小(纳米尺度)，统一乘 1e9 换成纳米(nm)便于显示读数
    if unit == "m":
        A, B, unit = A * 1e9, B * 1e9, "nm"
    # 物理量中文名（仅用于图注/色标），未知单位则回退到通道全名
    quantity = {"Hz": "频移", "nm": "高度", "m": "高度", "A": "电流",
                "V": "电压", "deg": "相位"}.get(unit, ch_name)
    # 输出文件名用的通道短标识（Z→z，频移→freqshift，其余取通道名字母数字）
    _nl = ch_name.lower()
    if _nl == "z":
        chan_slug = "z"
    elif "freq" in _nl and "shift" in _nl:
        chan_slug = "freqshift"
    else:
        chan_slug = re.sub(r"[^0-9a-z]+", "", _nl) or "chan"
    prefix = f"{chan_slug}_{'fwd' if args.direction == 'forward' else 'bwd'}"

    print(f"[通道] {ch_name}（{args.direction}），单位 {unit}")
    print(f"[尺寸] A: {A.shape}, B: {B.shape}; "
          f"像素 {sxm_A['pix_x_nm']:.4f} nm/px (x), {sxm_A['pix_y_nm']:.4f} nm/px (y)")

    # 几何一致性检查
    if A.shape != B.shape:
        raise SystemExit(f"两图尺寸不一致：A {A.shape} vs B {B.shape}，无法直接作差")

    # ---- 4.3 亚像素对齐：把 B 对齐到 A ----
    shift, error = estimate_shift(A, B, upsample=args.upsample)
    B_aligned, valid = apply_shift(B, shift)
    dc_nm, dr_nm = shift[1] * sxm_A["pix_x_nm"], shift[0] * sxm_A["pix_y_nm"]
    print(f"[对齐] 检测平移 (行,列)=({shift[0]:+.4f}, {shift[1]:+.4f}) px  "
          f"= (Δy {dr_nm:+.4f} nm, Δx {dc_nm:+.4f} nm), 配准误差={error:.4g}")

    # ---- 4.4 裁剪到完全重合的公共区域 ----
    # 平移后的 B_aligned 仅在 valid=True 处来自真实数据（A 处处有效），二者公共重合区
    # = valid 的最大整矩形。把 A / B / 对齐后的 B 都裁到这个矩形，直接切掉因平移产生的
    # 非重合边缘条带（不再用 NaN 留白）。
    # 纯平移得到的有效区是一个轴对齐矩形；对两个轴做「任一有效」投影即得其边界
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    ny0, nx0 = valid.shape
    if rows.size and cols.size:
        sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        A, B, B_aligned = A[sl], B[sl], B_aligned[sl]
        print(f"[裁剪] 公共重合区 {A.shape[0]}×{A.shape[1]}（原 {ny0}×{nx0}，"
              f"切掉非重合边缘 {ny0 - A.shape[0]} 行 / {nx0 - A.shape[1]} 列）")

    # ---- 4.5 作差（对齐前/后，均在公共重合区上）----
    diff_before = A - B                       # 未对齐的差分（仅用于对比展示）
    diff_after = A - B_aligned                # 对齐后的差分（真正的结果）
    # 可选：高斯平滑差分图以压制行噪声、凸显真实结构差异（默认关闭）
    if args.smooth_sigma and args.smooth_sigma > 0:
        diff_before = ndimage.gaussian_filter(diff_before, args.smooth_sigma)
        diff_after = ndimage.gaussian_filter(diff_after, args.smooth_sigma)

    rms_before = float(np.sqrt(np.nanmean(diff_before ** 2)))
    rms_after = float(np.sqrt(np.nanmean(diff_after ** 2)))
    print(f"[残差] 对齐前 RMS={rms_before:.4g} {unit} → 对齐后 RMS={rms_after:.4g} {unit} "
          f"（改善 {(1-rms_after/rms_before)*100:.1f}%）")

    # ---- 4.6 输出文件名 ----
    id_A, id_B = short_id(path_A), short_id(path_B)
    tag = f"_sigma{args.smooth_sigma:g}" if (args.smooth_sigma and args.smooth_sigma > 0) else ""
    f_AmB = os.path.join(outdir, f"{prefix}_{id_A}-{id_B}{tag}.png")   # A − B
    f_BmA = os.path.join(outdir, f"{prefix}_{id_B}-{id_A}{tag}.png")   # B − A
    f_over = os.path.join(outdir, f"{prefix}_alignment_overview_{id_A}_{id_B}{tag}.png")
    f_npy = os.path.join(outdir, f"{prefix}_{id_A}-{id_B}{tag}.npy")   # 对齐后差分数据(便于二次分析)

    # ---- 4.7 统一的对称色标（基于对齐后差分的稳健百分位）----
    vmax = float(np.nanpercentile(np.abs(diff_after), args.clip))
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = float(np.nanmax(np.abs(diff_after)))

    # ---- 4.8 保存两张差分图 + 示意图 + 数据 ----
    save_difference(diff_after, vmax, unit, quantity,
                    f"A − B  ({id_A} − {id_B})，对齐后 {ch_name}", f_AmB,
                    args.cmap, sxm_A["pix_x_nm"])
    save_difference(-diff_after, vmax, unit, quantity,
                    f"B − A  ({id_B} − {id_A})，对齐后 {ch_name}", f_BmA,
                    args.cmap, sxm_A["pix_x_nm"])
    save_overview(A, B, B_aligned, diff_before, diff_after,
                  vmax, shift, error, rms_before, rms_after,
                  sxm_A["pix_x_nm"], sxm_A["pix_y_nm"], unit, quantity, id_A, id_B, f_over, args.cmap)
    np.save(f_npy, diff_after)

    print("[输出]")
    for p in (f_AmB, f_BmA, f_over, f_npy):
        print("   ", p)
    print("[完成]")


if __name__ == "__main__":
    main()
