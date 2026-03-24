"""Reader for OME-TIFF acquisitions with acquisition.yaml metadata."""

from pathlib import Path

import numpy as np
import tifffile
import yaml


def detect(folder: Path) -> bool:
    """Check if folder contains ome_tiff/ subdir with acquisition metadata."""
    folder = Path(folder)
    has_ome = (folder / "ome_tiff").is_dir()
    has_yaml = (folder / "acquisition.yaml").exists() or (
        folder / "acquisition_channels.yaml"
    ).exists()
    return has_ome and has_yaml


def load_metadata(folder: Path) -> dict:
    """Parse acquisition YAML and enumerate OME-TIFF files."""
    folder = Path(folder)
    ome_dir = folder / "ome_tiff"

    # Parse acquisition metadata
    channels_info = []
    n_z = 1
    n_t = 1
    pixel_size_um = None

    acq_path = folder / "acquisition.yaml"
    if acq_path.exists():
        with open(acq_path) as f:
            acq = yaml.safe_load(f)

        # Extract channels — may be a list or dict
        raw_channels = acq.get("channels", [])
        if isinstance(raw_channels, list):
            for ch_cfg in raw_channels:
                if ch_cfg.get("enabled", True):
                    channels_info.append({
                        "name": ch_cfg.get("name", "unknown"),
                        "exposure_ms": ch_cfg.get("camera_settings", {}).get("exposure_time_ms"),
                        "color": ch_cfg.get("display_color"),
                    })
        elif isinstance(raw_channels, dict):
            for ch_name, ch_cfg in raw_channels.items():
                if ch_cfg.get("enabled", True):
                    channels_info.append({
                        "name": ch_name,
                        "exposure_ms": ch_cfg.get("exposure_time_ms"),
                        "color": ch_cfg.get("display_color"),
                    })

        # Z-stack
        z_cfg = acq.get("z_stack", {}) or {}
        n_z = z_cfg.get("nz", 1) or 1

        # Time
        t_cfg = acq.get("time_series", {}) or {}
        n_t = t_cfg.get("nt", 1) or 1

        # Pixel size
        obj = acq.get("objective", {})
        pixel_size_um = obj.get("pixel_size_um")

    # Fallback: parse acquisition_channels.yaml
    if not channels_info:
        ch_path = folder / "acquisition_channels.yaml"
        if ch_path.exists():
            with open(ch_path) as f:
                ch_yaml = yaml.safe_load(f)
            for ch in ch_yaml.get("channels", []):
                if ch.get("enabled", True):
                    channels_info.append({
                        "name": ch.get("name", "unknown"),
                        "color": ch.get("display_color"),
                    })

    # Enumerate files
    tiff_files = sorted(ome_dir.glob("*.ome.tiff")) + sorted(ome_dir.glob("*.ome.tif"))

    # Read shape from first file
    shape = None
    dtype = None
    n_pages = 0
    if tiff_files:
        with tifffile.TiffFile(str(tiff_files[0])) as tf:
            shape = tf.pages[0].shape
            dtype = tf.pages[0].dtype
            n_pages = len(tf.pages)

    n_channels = len(channels_info) if channels_info else 1

    return {
        "format": "ome_tiff",
        "folder": folder,
        "ome_dir": ome_dir,
        "channels": channels_info,
        "n_channels": n_channels,
        "n_z": n_z,
        "n_t": n_t,
        "n_files": len(tiff_files),
        "n_pages_per_file": n_pages,
        "shape": shape,
        "dtype": dtype,
        "pixel_size_um": pixel_size_um,
        "tiff_files": tiff_files,
    }


def read_plane(tiff_path: Path, page_index: int = 0) -> np.ndarray:
    """Read a single plane from an OME-TIFF file."""
    with tifffile.TiffFile(str(tiff_path)) as tf:
        return tf.pages[page_index].asarray()
