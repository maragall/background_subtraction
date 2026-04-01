"""Reader for *_stack.tiff multi-page format (Squid MULTI_PAGE_TIFF output).

From petakit/Deconvolution.
"""
import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from .base import AcquisitionReader, Metadata, FOV

_PATTERN = re.compile(r"^(.+?)_(\d+)_stack\.tiff$")


def detect_currentstack(root: Path) -> bool:
    """Check if directory contains *_stack.tiff multi-page files."""
    for subdir in root.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            for f in subdir.glob("*_stack.tiff"):
                if _PATTERN.match(f.name):
                    return True
    return False


def open_currentstack(root: Path) -> "CurrentStackReader":
    """Open a *_stack.tiff acquisition."""
    root = Path(root)

    plane_dir = None
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and subdir.name.isdigit():
            if any(f for f in subdir.glob("*_stack.tiff") if _PATTERN.match(f.name)):
                plane_dir = subdir
                break

    if plane_dir is None:
        raise FileNotFoundError("No *_stack.tiff files found")

    first_file = next(f for f in sorted(plane_dir.glob("*_stack.tiff")) if _PATTERN.match(f.name))
    channels = []
    nz = 0
    with tifffile.TiffFile(str(first_file)) as tif:
        ch_set = set()
        z_set = set()
        for page in tif.pages:
            meta = json.loads(page.description)
            ch_set.add(meta["channel"])
            z_set.add(meta["z_level"])
        nz = len(z_set)
        wl_pattern = re.compile(r"(\d{3})\s*nm")
        for ch in sorted(ch_set):
            m = wl_pattern.search(ch)
            channels.append(m.group(1) if m else ch)

    json_path = root / "acquisition parameters.json"
    metadata = Metadata.from_acquisition_json(json_path, channels=channels)
    metadata.nz = nz

    return CurrentStackReader(root, metadata, plane_dir)


class CurrentStackReader(AcquisitionReader):
    """Reader for multi-page *_stack.tiff files (Squid format)."""

    def __init__(self, root: Path, metadata: Metadata, plane_dir: Path):
        super().__init__(root, metadata)
        self._plane_dir = plane_dir

    @property
    def format_name(self) -> str:
        return "currentstack"

    def iter_fovs(self) -> Iterator[FOV]:
        seen = set()
        for f in sorted(self._plane_dir.glob("*_stack.tiff")):
            m = _PATTERN.match(f.name)
            if m:
                region, idx = m.group(1), int(m.group(2))
                key = (region, idx)
                if key not in seen:
                    seen.add(key)
                    yield FOV(region=region, index=idx)

    def get_frame(self, fov: FOV, channel: str, z_idx: int) -> np.ndarray:
        """Load a single z-plane."""
        path = self._plane_dir / f"{fov.region}_{fov.index}_stack.tiff"
        if not path.exists():
            raise FileNotFoundError(f"Stack file not found: {path}")

        with tifffile.TiffFile(str(path)) as tif:
            for page in tif.pages:
                meta = json.loads(page.description)
                if channel in meta["channel"] and meta["z_level"] == z_idx:
                    return page.asarray().astype(np.float32)

        raise ValueError(f"z={z_idx} channel '{channel}' not found in {path.name}")

    def iter_frames(self, channel: str):
        """Yield (fov, z_idx, file_path, page_idx) for parallel processing."""
        for fov in self.iter_fovs():
            path = self._plane_dir / f"{fov.region}_{fov.index}_stack.tiff"
            if not path.exists():
                continue
            with tifffile.TiffFile(str(path)) as tif:
                for page_idx, page in enumerate(tif.pages):
                    meta = json.loads(page.description)
                    if channel in meta["channel"]:
                        yield fov, meta["z_level"], path, page_idx

    @property
    def frame_shape(self) -> tuple:
        first_file = next(f for f in sorted(self._plane_dir.glob("*_stack.tiff")) if _PATTERN.match(f.name))
        with tifffile.TiffFile(str(first_file)) as tf:
            return tf.pages[0].shape

    @property
    def frame_dtype(self):
        first_file = next(f for f in sorted(self._plane_dir.glob("*_stack.tiff")) if _PATTERN.match(f.name))
        with tifffile.TiffFile(str(first_file)) as tf:
            return tf.pages[0].dtype

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        path = self._plane_dir / f"{fov.region}_{fov.index}_stack.tiff"
        if not path.exists():
            raise FileNotFoundError(f"Stack file not found: {path}")

        slices = {}
        with tifffile.TiffFile(str(path)) as tif:
            for page in tif.pages:
                meta = json.loads(page.description)
                if channel in meta["channel"]:
                    slices[meta["z_level"]] = page.asarray()

        if not slices:
            raise ValueError(f"Channel '{channel}' not found in {path.name}")

        stack = np.stack([slices[z] for z in sorted(slices)], axis=0)
        return stack.astype(np.float32)
