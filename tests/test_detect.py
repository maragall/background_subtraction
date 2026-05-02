"""Tests for format detection."""
import pytest

from bgsub.readers import detect_format, open_acquisition


def test_detect_individual(make_individual_fixture):
    root = make_individual_fixture()
    assert detect_format(root) == "individual"


def test_detect_ometiff(make_ometiff_fixture):
    root = make_ometiff_fixture()
    assert detect_format(root) == "ometiff"


def test_detect_currentstack(make_currentstack_fixture):
    root = make_currentstack_fixture()
    assert detect_format(root) == "currentstack"


def test_detect_unknown_returns_none(tmp_path):
    (tmp_path / "irrelevant.txt").write_text("hi")
    assert detect_format(tmp_path) is None


def test_open_unknown_raises(tmp_path):
    (tmp_path / "irrelevant.txt").write_text("hi")
    with pytest.raises(ValueError, match="Unknown acquisition format"):
        open_acquisition(tmp_path)


def test_open_missing_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        open_acquisition(missing)
