"""Smoke import test for the GUI module. No Qt instance created."""
import importlib

import pytest


def test_gui_app_imports():
    pytest.importorskip("PyQt5")
    mod = importlib.import_module("gui.app")
    assert hasattr(mod, "MainWindow")
    assert hasattr(mod, "MovieWorker")
    assert hasattr(mod, "PreviewWorker")
    assert hasattr(mod, "ProcessWorker")


def test_gui_does_not_export_auto_box_worker():
    """Auto-detect box-size feature was removed; the worker class should be gone."""
    pytest.importorskip("PyQt5")
    mod = importlib.import_module("gui.app")
    assert not hasattr(mod, "AutoBoxSizeWorker")
