"""Quality metrics for background subtraction evaluation."""

import numpy as np
import sep


def compute_metrics(original: np.ndarray, foreground: np.ndarray, background: np.ndarray) -> dict:
    """Compute quality metrics for a background subtraction result."""
    bg_mean = background.mean()
    bg_uniformity = background.std() / bg_mean if bg_mean > 0 else float("inf")

    corr = np.corrcoef(original.ravel(), foreground.ravel())[0, 1]

    fg_clipped = np.clip(foreground, 0, None)
    orig_snr = original.mean() / original.std() if original.std() > 0 else 0
    fg_snr = fg_clipped.mean() / fg_clipped.std() if fg_clipped.std() > 0 else 0
    snr_ratio = fg_snr / orig_snr if orig_snr > 0 else 0

    neg_pct = 100.0 * np.sum(foreground < 0) / foreground.size
    bg_fraction = bg_mean / original.mean() if original.mean() > 0 else 0

    return {
        "bg_uniformity": bg_uniformity,
        "signal_preservation": corr,
        "snr_improvement": snr_ratio,
        "negative_pixel_pct": neg_pct,
        "bg_fraction": bg_fraction,
    }


def suggest_box_size(
    image: np.ndarray, candidates: list[int] | None = None
) -> tuple[int, dict]:
    """Try multiple box sizes and suggest the best one.

    Returns (best_box_size, {box_size: metrics_dict}).
    """
    if candidates is None:
        candidates = [25, 50, 100, 200, 400]

    img = np.ascontiguousarray(image, dtype=np.float32)
    all_metrics = {}

    for bs in candidates:
        bkg = sep.Background(img, bw=bs, bh=bs, fw=3, fh=3)
        bg = bkg.back()
        fg = img - bg
        all_metrics[bs] = compute_metrics(img, fg, bg)

    # Pick box size that maximizes signal preservation
    best_bs = max(all_metrics, key=lambda bs: all_metrics[bs]["signal_preservation"])
    return best_bs, all_metrics
