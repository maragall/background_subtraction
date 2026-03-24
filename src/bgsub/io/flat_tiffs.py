"""Reader/writer for flat TIFF acquisitions (e.g. R0 naming convention)."""

import re
from pathlib import Path

import numpy as np
import tifffile

# Matches: R0_0_0_Fluorescence_488_nm_Ex_-_single_band_0000.tiff
_PATTERN = re.compile(
    r"^(?P<prefix>.+)_Fluorescence_(?P<wavelength>\d+_nm)_Ex_-_single_band_(?P<frame>\d+)\.tiff?$"
)


def detect(folder: Path) -> bool:
    """Check if folder contains flat TIFFs matching our naming convention."""
    folder = Path(folder)
    # Check direct children and one level down (e.g. R0/ subfolder)
    for d in [folder] + [p for p in folder.iterdir() if p.is_dir()]:
        for f in d.iterdir():
            if f.is_file() and _PATTERN.match(f.name):
                return True
    return False


def _find_tiff_dir(folder: Path) -> Path:
    """Return the directory containing the TIFFs (may be a subfolder like R0/)."""
    folder = Path(folder)
    for f in folder.iterdir():
        if f.is_file() and _PATTERN.match(f.name):
            return folder
    for d in folder.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file() and _PATTERN.match(f.name):
                    return d
    raise FileNotFoundError(f"No matching TIFFs found in {folder}")


def load_metadata(folder: Path) -> dict:
    """Parse filenames to extract channels, frame count, image shape, and file map."""
    tiff_dir = _find_tiff_dir(folder)
    file_map = {}  # (channel, frame_idx) -> Path
    channels = set()

    for f in sorted(tiff_dir.iterdir()):
        m = _PATTERN.match(f.name)
        if not m:
            continue
        ch = m.group("wavelength")  # e.g. "488_nm"
        frame = int(m.group("frame"))
        channels.add(ch)
        file_map[(ch, frame)] = f

    if not file_map:
        raise FileNotFoundError(f"No matching TIFFs in {tiff_dir}")

    channels = sorted(channels)
    frames_per_channel = {}
    for ch in channels:
        frames = sorted(fr for (c, fr) in file_map if c == ch)
        frames_per_channel[ch] = frames

    # Read shape from first file
    first_path = next(iter(file_map.values()))
    with tifffile.TiffFile(str(first_path)) as tf:
        shape = tf.pages[0].shape
        dtype = tf.pages[0].dtype

    return {
        "format": "flat_tiffs",
        "tiff_dir": tiff_dir,
        "channels": channels,
        "frames_per_channel": frames_per_channel,
        "n_frames": {ch: len(frames_per_channel[ch]) for ch in channels},
        "shape": shape,
        "dtype": dtype,
        "file_map": file_map,
    }


def read_frame(path: Path) -> np.ndarray:
    """Read a single TIFF frame."""
    return tifffile.imread(str(path))


def write_frame(image: np.ndarray, path: Path, dtype=np.uint16):
    """Write image as TIFF, clipping to dtype range."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = np.iinfo(dtype)
    clipped = np.clip(image, info.min, info.max).astype(dtype)
    tifffile.imwrite(str(path), clipped)
