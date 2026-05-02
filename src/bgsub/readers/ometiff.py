"""Reader for OME-TIFF format (from petakit/Deconvolution)."""
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from .base import AcquisitionReader, Metadata, FOV, FrameRef


def detect_ometiff(root: Path) -> bool:
    """Check if directory contains OME-TIFF format."""
    ome_dir = root / "ome_tiff"
    if ome_dir.exists():
        return any(ome_dir.glob("*.ome.tiff"))
    return False


def open_ometiff(root: Path) -> "OMETiffReader":
    """Open an acquisition in OME-TIFF format."""
    root = Path(root)
    ome_dir = root / "ome_tiff"

    first_file = next(ome_dir.glob("*.ome.tiff"))
    with tifffile.TiffFile(first_file) as tif:
        ome = tif.ome_metadata
        channels = _parse_channels_from_ome(ome)

    json_path = root / "acquisition_parameters.json"
    if not json_path.exists():
        json_path = root / "acquisition parameters.json"

    metadata = Metadata.from_acquisition_json(json_path, channels)

    reader = OMETiffReader(root, metadata)
    reader._channel_indices = {ch: i for i, ch in enumerate(channels)}
    reader._n_channels = len(channels)
    return reader


def _parse_channels_from_ome(ome_xml: str) -> list[str]:
    """Extract channel wavelengths from OME-XML, in document order, deduplicated."""
    pattern = re.compile(r'Name="Fluorescence (\d+) nm Ex"')
    seen = []
    for match in pattern.finditer(ome_xml):
        ch = match.group(1)
        if ch not in seen:
            seen.append(ch)
    return seen


class OMETiffReader(AcquisitionReader):
    """Reader for OME-TIFF format.

    Directory structure:
        root/
            acquisition_parameters.json
            ome_tiff/
                <region><idx>_<n>.ome.tiff
                ...

    Page-order assumption: pages within one OME-TIFF interleave channels then
    Z planes (DimensionOrder XYCZT), so page = z_idx * n_channels + channel_idx.
    """

    # Populated by open_ometiff. Public read-only.
    _channel_indices: dict[str, int]
    _n_channels: int

    @property
    def format_name(self) -> str:
        return "ometiff"

    @property
    def ome_dir(self) -> Path:
        return self.root / "ome_tiff"

    def _file_for_fov(self, fov: FOV) -> Path:
        return self.ome_dir / f"{fov.region}_{fov.index}.ome.tiff"

    def iter_fovs(self) -> Iterator[FOV]:
        pattern = re.compile(r"^([a-zA-Z]+\d+)_(\d+)\.ome\.tiff$")
        for f in sorted(self.ome_dir.glob("*.ome.tiff")):
            if match := pattern.match(f.name):
                yield FOV(region=match.group(1), index=int(match.group(2)))

    def _channel_index(self, channel: str) -> int:
        if channel not in self._channel_indices:
            raise ValueError(
                f"Channel {channel} not found. Available: "
                f"{list(self._channel_indices)}"
            )
        return self._channel_indices[channel]

    def _nz_for_file(self, filepath: Path) -> int:
        with tifffile.TiffFile(filepath) as tif:
            return len(tif.pages) // max(self._n_channels, 1)

    def iter_frames_for_fov(self, fov: FOV, channel: str):
        filepath = self._file_for_fov(fov)
        if not filepath.exists():
            return
        c_idx = self._channel_index(channel)
        nz = self._nz_for_file(filepath)
        for z in range(nz):
            page = z * max(self._n_channels, 1) + c_idx
            yield FrameRef(fov=fov, frame_idx=z, file_path=filepath, page_idx=page)

    def n_frames_per_fov(self, fov: FOV, channel: str) -> int:
        filepath = self._file_for_fov(fov)
        if not filepath.exists():
            return 0
        return self._nz_for_file(filepath)

    def iter_frames(self, channel: str):
        for fov in self.iter_fovs():
            yield from self.iter_frames_for_fov(fov, channel)

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        filepath = self._file_for_fov(fov)
        if not filepath.exists():
            raise FileNotFoundError(f"OME-TIFF not found: {filepath}")
        c_idx = self._channel_index(channel)
        with tifffile.TiffFile(filepath) as tif:
            data = tif.series[0].asarray()
            axes = tif.series[0].axes.upper()
        return _select_channel(data, axes, c_idx).astype(np.float32)

    def get_frame(self, fov: FOV, channel: str, z_idx: int) -> np.ndarray:
        stack = self.get_stack(fov, channel)
        if stack.ndim == 3:
            return stack[min(z_idx, stack.shape[0] - 1)]
        return stack

    @property
    def frame_shape(self) -> tuple:
        first_file = next(self.ome_dir.glob("*.ome.tiff"))
        with tifffile.TiffFile(first_file) as tf:
            return tf.pages[0].shape

    @property
    def frame_dtype(self):
        first_file = next(self.ome_dir.glob("*.ome.tiff"))
        with tifffile.TiffFile(first_file) as tf:
            return tf.pages[0].dtype


def _select_channel(data: np.ndarray, axes: str, channel_idx: int) -> np.ndarray:
    """Squeeze T if length 1, take given channel along C if present."""
    if "T" in axes:
        t_pos = axes.index("T")
        if data.shape[t_pos] == 1:
            data = np.squeeze(data, axis=t_pos)
            axes = axes.replace("T", "")
    if "C" in axes:
        c_pos = axes.index("C")
        return np.take(data, channel_idx, axis=c_pos)
    return data
