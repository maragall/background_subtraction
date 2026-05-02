"""Tests for BackgroundSubtractor core."""
import numpy as np
import sep
import tifffile

from bgsub import BackgroundSubtractor
from bgsub.core import _run_sep


def test_run_sep_matches_direct_call():
    rng = np.random.default_rng(0)
    img = rng.integers(100, 1000, size=(64, 64)).astype(np.uint16)
    fg, bg = _run_sep(img, box_size=16)
    # Reference
    ref_input = np.ascontiguousarray(img, dtype=np.float32)
    ref_bkg = sep.Background(ref_input, bw=16, bh=16, fw=3, fh=3)
    ref_bg = ref_bkg.back()
    np.testing.assert_array_equal(bg, ref_bg)
    np.testing.assert_array_equal(fg, ref_input - ref_bg)


def test_process_single_returns_correct_shapes(make_individual_fixture):
    root = make_individual_fixture(shape=(64, 64))
    sub = BackgroundSubtractor(root, box_size=16)
    img = np.random.default_rng(1).integers(100, 1000, size=(64, 64)).astype(np.uint16)
    fg, bg = sub.process_single(img)
    assert fg.shape == (64, 64)
    assert bg.shape == (64, 64)
    assert fg.dtype == np.float32


def test_process_all_writes_one_output_per_input_frame(make_individual_fixture):
    root = make_individual_fixture(fovs=2, channels=("488",), z_planes=3)
    sub = BackgroundSubtractor(root, box_size=16, max_workers=2)
    sub.process_all(channels=["488"])
    outputs = list(sub.output_path.glob("*.tiff"))
    assert len(outputs) == 6  # 2 fov * 3 z


def test_output_dtype_matches_input(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488",), z_planes=1)
    sub = BackgroundSubtractor(root, box_size=16, max_workers=1)
    sub.process_all(channels=["488"])
    out = next(sub.output_path.glob("*.tiff"))
    written = tifffile.imread(str(out))
    assert written.dtype == np.uint16


def test_output_clipped_to_dtype_range(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488",), z_planes=1)
    sub = BackgroundSubtractor(root, box_size=16, max_workers=1)
    sub.process_all(channels=["488"])
    out = next(sub.output_path.glob("*.tiff"))
    written = tifffile.imread(str(out))
    info = np.iinfo(np.uint16)
    assert written.min() >= info.min
    assert written.max() <= info.max


def test_progress_callback_invoked(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488",), z_planes=2)
    sub = BackgroundSubtractor(root, box_size=16, max_workers=1)
    calls = []
    sub.process_all(
        channels=["488"],
        progress_callback=lambda c, t, m: calls.append((c, t, m)),
    )
    assert len(calls) == 2
    assert calls[-1][0] == 2
    assert calls[-1][1] == 2


def test_metrics_dict_has_expected_keys(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488",), z_planes=1)
    sub = BackgroundSubtractor(root, box_size=16)
    _orig, _fg, _bg, metrics = sub.process_frame("488", 0)
    assert set(metrics.keys()) == {
        "bg_uniformity",
        "signal_preservation",
        "snr_improvement",
        "negative_pixel_pct",
    }
