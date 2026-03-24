#!/usr/bin/env python3
"""Explore rolling ball background subtraction on test microscopy data."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.restoration import rolling_ball

TEST_DIR = Path.home() / "Downloads" / "20250802_andrea_test_R0" / "R0"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "exploration"
RADII = [25, 50, 100, 200, 500]

SAMPLE_FILES = [
    "R0_0_0_Fluorescence_488_nm_Ex_-_single_band_0000.tiff",
    "R0_0_0_Fluorescence_561_nm_Ex_-_single_band_0000.tiff",
]


def process_one(image_path: Path, radii: list[int], output_dir: Path):
    """Run rolling ball at multiple radii on one image, save comparison."""
    print(f"\n--- {image_path.name} ---")
    raw = tifffile.imread(str(image_path))
    print(f"  Shape: {raw.shape}, dtype: {raw.dtype}, range: [{raw.min()}, {raw.max()}]")

    image = raw.astype(np.float32)
    stem = image_path.stem

    # Downsample for display
    ds = 8
    h, w = image.shape
    thumb = image[::ds, ::ds]

    results = {}
    for r in radii:
        t0 = time.time()
        bg = rolling_ball(image, radius=r)
        elapsed = time.time() - t0
        fg = image - bg
        results[r] = (fg, bg, elapsed)
        neg_pct = 100.0 * np.sum(fg < 0) / fg.size
        print(f"  radius={r:4d}  time={elapsed:.1f}s  bg_mean={bg.mean():.0f}  neg_px={neg_pct:.2f}%")

    # Plot comparison grid
    n = len(radii)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    vmax_orig = np.percentile(thumb, 99.5)
    for i, r in enumerate(radii):
        fg, bg, elapsed = results[r]
        fg_thumb = fg[::ds, ::ds]
        bg_thumb = bg[::ds, ::ds]

        axes[i, 0].imshow(thumb, cmap="gray", vmin=0, vmax=vmax_orig)
        axes[i, 0].set_title(f"Original")
        axes[i, 0].set_ylabel(f"r={r} ({elapsed:.1f}s)")

        axes[i, 1].imshow(bg_thumb, cmap="gray", vmin=0, vmax=vmax_orig)
        axes[i, 1].set_title(f"Background")

        vmax_fg = np.percentile(fg_thumb, 99.5)
        axes[i, 2].imshow(np.clip(fg_thumb, 0, None), cmap="gray", vmin=0, vmax=vmax_fg)
        axes[i, 2].set_title(f"Foreground")

    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(stem, fontsize=14)
    fig.tight_layout()

    out_path = output_dir / f"{stem}_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in SAMPLE_FILES:
        path = TEST_DIR / fname
        if not path.exists():
            print(f"SKIP (not found): {path}")
            continue
        process_one(path, RADII, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
