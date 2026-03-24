#!/usr/bin/env python3
"""Sweep rolling ball radii and compute quality metrics. Outputs CSV + plot."""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.restoration import rolling_ball

TEST_DIR = Path.home() / "Downloads" / "20250802_andrea_test_R0" / "R0"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "metrics"
RADII = [10, 25, 50, 75, 100, 150, 200, 300, 500]
DOWNSAMPLE = 4  # Downsample factor for speed; radii are scaled proportionally

SAMPLE_FILES = [
    "R0_0_0_Fluorescence_488_nm_Ex_-_single_band_0000.tiff",
    "R0_0_0_Fluorescence_561_nm_Ex_-_single_band_0000.tiff",
]


def compute_metrics(original: np.ndarray, foreground: np.ndarray, background: np.ndarray) -> dict:
    bg_mean = background.mean()
    bg_uniformity = background.std() / bg_mean if bg_mean > 0 else float("inf")

    flat_orig = original.ravel()
    flat_fg = foreground.ravel()
    corr = np.corrcoef(flat_orig, flat_fg)[0, 1]

    orig_mean = original.mean()
    orig_snr = orig_mean / original.std() if original.std() > 0 else 0
    fg_clipped = np.clip(foreground, 0, None)
    fg_snr = fg_clipped.mean() / fg_clipped.std() if fg_clipped.std() > 0 else 0
    snr_ratio = fg_snr / orig_snr if orig_snr > 0 else 0

    neg_pct = 100.0 * np.sum(foreground < 0) / foreground.size
    bg_fraction = bg_mean / orig_mean if orig_mean > 0 else 0

    return {
        "bg_uniformity": bg_uniformity,
        "signal_preservation": corr,
        "snr_improvement": snr_ratio,
        "negative_pixel_pct": neg_pct,
        "bg_fraction": bg_fraction,
    }


def sweep(image_path: Path, radii: list[int], output_dir: Path):
    print(f"\n--- {image_path.name} ---")
    raw = tifffile.imread(str(image_path))
    # Downsample for speed
    if DOWNSAMPLE > 1:
        raw = raw[::DOWNSAMPLE, ::DOWNSAMPLE]
        print(f"  Downsampled {DOWNSAMPLE}x -> {raw.shape}")
    image = raw.astype(np.float32)
    stem = image_path.stem

    rows = []
    for r_orig in radii:
        r = max(1, r_orig // DOWNSAMPLE) if DOWNSAMPLE > 1 else r_orig
        t0 = time.time()
        bg = rolling_ball(image, radius=r)
        elapsed = time.time() - t0
        fg = image - bg
        m = compute_metrics(image, fg, bg)
        m["radius"] = r_orig  # Report the original (full-res equivalent) radius
        m["radius_effective"] = r
        m["time_s"] = elapsed
        rows.append(m)
        print(f"  r={r_orig:4d} (eff={r:3d})  bg_unif={m['bg_uniformity']:.4f}  "
              f"sig_pres={m['signal_preservation']:.4f}  "
              f"snr_imp={m['snr_improvement']:.2f}  "
              f"neg={m['negative_pixel_pct']:.2f}%  "
              f"time={elapsed:.1f}s")

    # Write CSV
    csv_path = output_dir / f"{stem}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV: {csv_path}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    rs = [row["radius"] for row in rows]

    axes[0, 0].plot(rs, [row["bg_uniformity"] for row in rows], "o-")
    axes[0, 0].set_title("Background Uniformity (lower=better)")
    axes[0, 0].set_xlabel("Radius")

    axes[0, 1].plot(rs, [row["signal_preservation"] for row in rows], "o-")
    axes[0, 1].set_title("Signal Preservation (higher=better)")
    axes[0, 1].set_xlabel("Radius")

    axes[1, 0].plot(rs, [row["snr_improvement"] for row in rows], "o-")
    axes[1, 0].set_title("SNR Improvement Ratio")
    axes[1, 0].set_xlabel("Radius")

    axes[1, 1].plot(rs, [row["negative_pixel_pct"] for row in rows], "o-")
    axes[1, 1].set_title("Negative Pixels %")
    axes[1, 1].set_xlabel("Radius")

    fig.suptitle(stem, fontsize=12)
    fig.tight_layout()
    plot_path = output_dir / f"{stem}_metrics.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  Plot: {plot_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in SAMPLE_FILES:
        path = TEST_DIR / fname
        if not path.exists():
            print(f"SKIP (not found): {path}")
            continue
        sweep(path, RADII, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
