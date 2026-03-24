#!/usr/bin/env python3
"""Explore photutils Background2D on test microscopy data."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

TEST_DIR = Path.home() / "Downloads" / "20250802_andrea_test_R0" / "R0"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "exploration_bg2d"
BOX_SIZES = [25, 50, 100, 200, 400]

SAMPLE_FILES = [
    "R0_0_0_Fluorescence_488_nm_Ex_-_single_band_0000.tiff",
    "R0_0_0_Fluorescence_561_nm_Ex_-_single_band_0000.tiff",
]


def process_one(image_path: Path, box_sizes: list[int], output_dir: Path):
    print(f"\n--- {image_path.name} ---")
    raw = tifffile.imread(str(image_path))
    print(f"  Shape: {raw.shape}, dtype: {raw.dtype}, range: [{raw.min()}, {raw.max()}]")

    image = raw.astype(np.float32)
    stem = image_path.stem
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()

    ds = 8  # downsample for display
    thumb = image[::ds, ::ds]

    results = {}
    for bs in box_sizes:
        t0 = time.time()
        bkg = Background2D(image, box_size=(bs, bs), filter_size=(3, 3),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        bg = bkg.background
        fg = image - bg
        elapsed = time.time() - t0
        results[bs] = (fg, bg, elapsed)
        neg_pct = 100.0 * np.sum(fg < 0) / fg.size
        print(f"  box={bs:4d}  time={elapsed:.2f}s  bg_mean={bg.mean():.0f}  neg_px={neg_pct:.2f}%")

    # Plot comparison grid
    n = len(box_sizes)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    vmax_orig = np.percentile(thumb, 99.5)
    for i, bs in enumerate(box_sizes):
        fg, bg, elapsed = results[bs]
        fg_thumb = fg[::ds, ::ds]
        bg_thumb = bg[::ds, ::ds]

        axes[i, 0].imshow(thumb, cmap="gray", vmin=0, vmax=vmax_orig)
        axes[i, 0].set_title("Original")
        axes[i, 0].set_ylabel(f"box={bs} ({elapsed:.2f}s)")

        axes[i, 1].imshow(bg_thumb, cmap="gray", vmin=0, vmax=vmax_orig)
        axes[i, 1].set_title("Background")

        vmax_fg = np.percentile(np.clip(fg_thumb, 0, None), 99.5)
        axes[i, 2].imshow(np.clip(fg_thumb, 0, None), cmap="gray", vmin=0, vmax=vmax_fg)
        axes[i, 2].set_title("Foreground")

    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{stem} — photutils Background2D", fontsize=14)
    fig.tight_layout()

    out_path = output_dir / f"{stem}_bg2d_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in SAMPLE_FILES:
        path = TEST_DIR / fname
        if not path.exists():
            print(f"SKIP (not found): {path}")
            continue
        process_one(path, BOX_SIZES, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
