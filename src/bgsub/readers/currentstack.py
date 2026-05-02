"""Reader for *_stack.tiff multi-page format (Squid MULTI_PAGE_TIFF output).

From petakit/Deconvolution.
"""
import json
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tifffile

from .base import FOV, AcquisitionReader, FrameRef, Metadata

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

    first_file = next(
        f for f in sorted(plane_dir.glob("*_stack.tiff")) if _PATTERN.match(f.name)
    )

    # Parse channels from page descriptions of the first file
    channels = []
    nz = 0
    wl_pattern = re.compile(r"(\d{3})\s*nm")
    with tifffile.TiffFile(str(first_file)) as tif:
        ch_set = set()
        z_set = set()
        for page in tif.pages:
            meta = json.loads(page.description)
            ch_set.add(meta["channel"])
            z_set.add(meta["z_level"])
        nz = len(z_set)
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
        # path -> {(channel_wavelength, z_level): page_idx}
        self._page_index: dict[Path, dict[tuple[str, int], int]] = {}

    @property
    def format_name(self) -> str:
        return "currentstack"

    def _path_for_fov(self, fov: FOV) -> Path:
        return self._plane_dir / f"{fov.region}_{fov.index}_stack.tiff"

    def _index_for(self, path: Path) -> dict[tuple[str, int], int]:
        if path in self._page_index:
            return self._page_index[path]
        wl_pattern = re.compile(r"(\d{3})\s*nm")
        idx: dict[tuple[str, int], int] = {}
        with tifffile.TiffFile(str(path)) as tif:
            for page_idx, page in enumerate(tif.pages):
                meta = json.loads(page.description)
                m = wl_pattern.search(meta["channel"])
                wavelength = m.group(1) if m else meta["channel"]
                idx[(wavelength, meta["z_level"])] = page_idx
        self._page_index[path] = idx
        return idx

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

    def iter_frames_for_fov(self, fov: FOV, channel: str):
        path = self._path_for_fov(fov)
        if not path.exists():
            return
        idx = self._index_for(path)
        for (ch, z), page_idx in sorted(idx.items(), key=lambda kv: kv[0][1]):
            if ch == channel:
                yield FrameRef(fov=fov, frame_idx=z, file_path=path, page_idx=page_idx)

    def n_frames_per_fov(self, fov: FOV, channel: str) -> int:
        path = self._path_for_fov(fov)
        if not path.exists():
            return 0
        idx = self._index_for(path)
        return sum(1 for (ch, _z) in idx if ch == channel)

    def get_frame(self, fov: FOV, channel: str, z_idx: int) -> np.ndarray:
        path = self._path_for_fov(fov)
        if not path.exists():
            raise FileNotFoundError(f"Stack file not found: {path}")
        idx = self._index_for(path)
        if (channel, z_idx) not in idx:
            raise ValueError(f"z={z_idx} channel '{channel}' not found in {path.name}")
        page_idx = idx[(channel, z_idx)]
        with tifffile.TiffFile(str(path)) as tif:
            return tif.pages[page_idx].asarray().astype(np.float32)

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        path = self._path_for_fov(fov)
        if not path.exists():
            raise FileNotFoundError(f"Stack file not found: {path}")
        idx = self._index_for(path)
        z_pages = sorted(((z, p) for (ch, z), p in idx.items() if ch == channel))
        if not z_pages:
            raise ValueError(f"Channel '{channel}' not found in {path.name}")
        with tifffile.TiffFile(str(path)) as tif:
            slices = [tif.pages[p].asarray() for _z, p in z_pages]
        return np.stack(slices, axis=0).astype(np.float32)

    @property
    def frame_shape(self) -> tuple:
        first_file = next(
            f for f in sorted(self._plane_dir.glob("*_stack.tiff"))
            if _PATTERN.match(f.name)
        )
        with tifffile.TiffFile(str(first_file)) as tf:
            return tf.pages[0].shape

    @property
    def frame_dtype(self):
        first_file = next(
            f for f in sorted(self._plane_dir.glob("*_stack.tiff"))
            if _PATTERN.match(f.name)
        )
        with tifffile.TiffFile(str(first_file)) as tf:
            return tf.pages[0].dtype
