"""Quality metrics for background subtraction evaluation.

Note: `signal_preservation` is `corrcoef(original, foreground)`. It is useful
as a sanity number — a correlation near zero means the subtraction destroyed
all signal — but it is NOT an optimization target. The correlation is highest
when foreground ≈ original, i.e., when no background was subtracted, so
maximizing it would prefer the largest box sizes that change nothing.
"""

import numpy as np


def compute_metrics(
    original: np.ndarray, foreground: np.ndarray, background: np.ndarray
) -> dict:
    """Compute quality metrics for a background subtraction result."""
    bg_mean = background.mean()
    bg_uniformity = background.std() / bg_mean if bg_mean > 0 else float("inf")

    corr = np.corrcoef(original.ravel(), foreground.ravel())[0, 1]

    fg_clipped = np.clip(foreground, 0, None)
    orig_snr = original.mean() / original.std() if original.std() > 0 else 0
    fg_snr = fg_clipped.mean() / fg_clipped.std() if fg_clipped.std() > 0 else 0
    snr_ratio = fg_snr / orig_snr if orig_snr > 0 else 0

    neg_pct = 100.0 * np.sum(foreground < 0) / foreground.size

    return {
        "bg_uniformity": bg_uniformity,
        "signal_preservation": corr,
        "snr_improvement": snr_ratio,
        "negative_pixel_pct": neg_pct,
    }
