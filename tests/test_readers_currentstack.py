"""Tests for CurrentStackReader."""
import numpy as np

from bgsub.readers import FrameRef, open_acquisition
from bgsub.readers.currentstack import CurrentStackReader


def test_open_currentstack_returns_correct_reader(make_currentstack_fixture):
    root = make_currentstack_fixture()
    reader = open_acquisition(root)
    assert isinstance(reader, CurrentStackReader)
    assert reader.format_name == "currentstack"


def test_channels_parsed(make_currentstack_fixture):
    root = make_currentstack_fixture(channels=("488 nm Ex", "561 nm Ex"))
    reader = open_acquisition(root)
    assert reader.metadata.channels == ["488", "561"]


def test_iter_fovs_yields_unique(make_currentstack_fixture):
    root = make_currentstack_fixture(fovs=3)
    reader = open_acquisition(root)
    fovs = list(reader.iter_fovs())
    assert len(fovs) == 3


def test_n_frames_per_fov_matches_z(make_currentstack_fixture):
    root = make_currentstack_fixture(z_planes=4)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    assert reader.n_frames_per_fov(fov, "488") == 4


def test_iter_frames_for_fov_yields_frame_refs(make_currentstack_fixture):
    root = make_currentstack_fixture(z_planes=3)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    refs = list(reader.iter_frames_for_fov(fov, "488"))
    assert len(refs) == 3
    assert all(isinstance(r, FrameRef) for r in refs)
    assert all(r.page_idx is not None for r in refs)


def test_get_frame_returns_2d(make_currentstack_fixture):
    root = make_currentstack_fixture(shape=(48, 32))
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    frame = reader.get_frame(fov, "488", 0)
    assert frame.shape == (48, 32)
    assert frame.dtype == np.float32
