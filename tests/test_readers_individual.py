"""Tests for IndividualReader. Uses synthetic fixtures from conftest.py."""
import numpy as np

from bgsub.readers import FrameRef, open_acquisition
from bgsub.readers.individual import IndividualReader


def test_open_individual_returns_correct_reader(make_individual_fixture):
    root = make_individual_fixture()
    reader = open_acquisition(root)
    assert isinstance(reader, IndividualReader)
    assert reader.format_name == "individual"


def test_iter_fovs_yields_unique_fovs(make_individual_fixture):
    root = make_individual_fixture(fovs=3, channels=("488",), z_planes=1)
    reader = open_acquisition(root)
    fovs = list(reader.iter_fovs())
    assert len(fovs) == 3
    assert {f.index for f in fovs} == {0, 1, 2}


def test_n_frames_per_fov_matches_z_planes(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488", "561"), z_planes=4)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    assert reader.n_frames_per_fov(fov, "488") == 4
    assert reader.n_frames_per_fov(fov, "561") == 4


def test_iter_frames_for_fov_yields_frame_refs_in_order(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488",), z_planes=3)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    refs = list(reader.iter_frames_for_fov(fov, "488"))
    assert len(refs) == 3
    assert all(isinstance(r, FrameRef) for r in refs)
    assert all(r.page_idx is None for r in refs)  # single-page TIFFs
    assert [r.frame_idx for r in refs] == sorted(r.frame_idx for r in refs)


def test_iter_frames_aggregates_across_fovs(make_individual_fixture):
    root = make_individual_fixture(fovs=2, channels=("488",), z_planes=3)
    reader = open_acquisition(root)
    refs = list(reader.iter_frames("488"))
    assert len(refs) == 6  # 2 fovs * 3 z


def test_get_frame_returns_2d_array(make_individual_fixture):
    root = make_individual_fixture(shape=(48, 32))
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    frame = reader.get_frame(fov, "488", 0)
    assert frame.shape == (48, 32)
    assert frame.dtype == np.float32


def test_frame_shape_and_dtype(make_individual_fixture):
    root = make_individual_fixture(shape=(48, 32))
    reader = open_acquisition(root)
    assert reader.frame_shape == (48, 32)
    assert reader.frame_dtype == np.uint16


def test_glob_ignores_unrelated_tiffs(make_individual_fixture, tmp_path):
    """frame_shape/frame_dtype must not pick up bg-subtracted output files
    that happen to live in the same directory."""
    root = make_individual_fixture(shape=(48, 32))
    reader = open_acquisition(root)
    # Drop a stray TIFF in the tiff_dir that doesn't match the Fluorescence pattern
    import tifffile
    stray = reader._tiff_dir / "stray_output.tiff"
    tifffile.imwrite(str(stray), np.zeros((1, 1), dtype=np.uint8))
    # Should still report the real frame's shape
    assert reader.frame_shape == (48, 32)
    assert reader.frame_dtype == np.uint16
