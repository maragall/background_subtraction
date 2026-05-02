"""Tests for OMETiffReader."""
import numpy as np

from bgsub.readers import FrameRef, open_acquisition
from bgsub.readers.ometiff import OMETiffReader


def test_open_ometiff_returns_correct_reader(make_ometiff_fixture):
    root = make_ometiff_fixture()
    reader = open_acquisition(root)
    assert isinstance(reader, OMETiffReader)
    assert reader.format_name == "ometiff"


def test_channels_parsed_from_ome_xml(make_ometiff_fixture):
    root = make_ometiff_fixture(channels=("405", "488", "561"))
    reader = open_acquisition(root)
    assert reader.metadata.channels == ["405", "488", "561"]


def test_iter_fovs_yields_unique_fovs(make_ometiff_fixture):
    root = make_ometiff_fixture(fovs=3)
    reader = open_acquisition(root)
    fovs = list(reader.iter_fovs())
    assert len(fovs) == 3


def test_n_frames_per_fov_matches_z(make_ometiff_fixture):
    root = make_ometiff_fixture(fovs=1, z_planes=5)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    assert reader.n_frames_per_fov(fov, "488") == 5


def test_iter_frames_for_fov_yields_frame_refs(make_ometiff_fixture):
    root = make_ometiff_fixture(fovs=1, channels=("488", "561"), z_planes=3)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    refs = list(reader.iter_frames_for_fov(fov, "488"))
    assert len(refs) == 3
    assert all(isinstance(r, FrameRef) for r in refs)
    assert all(r.page_idx is not None for r in refs)  # multi-page


def test_get_frame_returns_2d(make_ometiff_fixture):
    root = make_ometiff_fixture(shape=(48, 32))
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    frame = reader.get_frame(fov, "488", 0)
    assert frame.shape == (48, 32)
    assert frame.dtype == np.float32


def test_channel_index_cache(make_ometiff_fixture):
    """After first call, channel indices should be cached on the reader."""
    root = make_ometiff_fixture(channels=("488", "561"))
    reader = open_acquisition(root)
    # Trigger caching
    list(reader.iter_frames("488"))
    assert hasattr(reader, "_channel_indices")
    assert reader._channel_indices == {"488": 0, "561": 1}
