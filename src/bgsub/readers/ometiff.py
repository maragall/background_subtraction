"""Reader for OME-TIFF format (from petakit/Deconvolution)."""
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from .base import AcquisitionReader, Metadata, FOV


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

    return OMETiffReader(root, metadata)


def _parse_channels_from_ome(ome_xml: str) -> list[str]:
    """Extract channel wavelengths from OME-XML."""
    channels = []
    pattern = re.compile(r'Name="Fluorescence (\d+) nm Ex"')
    for match in pattern.finditer(ome_xml):
        channels.append(match.group(1))
    return sorted(set(channels), key=int)


class OMETiffReader(AcquisitionReader):
    """Reader for OME-TIFF format.

    Directory structure:
        root/
            acquisition_parameters.json
            ome_tiff/
                manual0_0.ome.tiff
                manual0_1.ome.tiff
                ...
    """

    @property
    def format_name(self) -> str:
        return "ometiff"

    @property
    def ome_dir(self) -> Path:
        return self.root / "ome_tiff"

    def iter_fovs(self) -> Iterator[FOV]:
        pattern = re.compile(r"^([a-zA-Z]+\d+)_(\d+)\.ome\.tiff$")
        for f in sorted(self.ome_dir.glob("*.ome.tiff")):
            if match := pattern.match(f.name):
                region = match.group(1)
                fov_idx = int(match.group(2))
                yield FOV(region=region, index=fov_idx)

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        filename = f"{fov.region}_{fov.index}.ome.tiff"
        filepath = self.ome_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"OME-TIFF not found: {filepath}")

        with tifffile.TiffFile(filepath) as tif:
            channel_idx = self._get_channel_index(tif.ome_metadata, channel)

            series = tif.series[0]
            data = series.asarray()
            axes = series.axes.upper()

            if 'T' in axes:
                t_pos = axes.index('T')
                if data.shape[t_pos] == 1:
                    data = np.squeeze(data, axis=t_pos)
                    axes = axes.replace('T', '')

            if 'C' in axes:
                c_pos = axes.index('C')
                stack = np.take(data, channel_idx, axis=c_pos)
            else:
                stack = data

        return stack.astype(np.float32)

    def get_frame(self, fov: FOV, channel: str, z_idx: int) -> np.ndarray:
        """Load a single 2D z-plane without loading the full stack."""
        filename = f"{fov.region}_{fov.index}.ome.tiff"
        filepath = self.ome_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"OME-TIFF not found: {filepath}")

        with tifffile.TiffFile(filepath) as tif:
            channel_idx = self._get_channel_index(tif.ome_metadata, channel)
            series = tif.series[0]
            data = series.asarray()
            axes = series.axes.upper()

            if 'T' in axes:
                t_pos = axes.index('T')
                if data.shape[t_pos] == 1:
                    data = np.squeeze(data, axis=t_pos)
                    axes = axes.replace('T', '')

            if 'C' in axes:
                c_pos = axes.index('C')
                stack = np.take(data, channel_idx, axis=c_pos)
            else:
                stack = data

        if stack.ndim == 3:
            idx = min(z_idx, stack.shape[0] - 1)
            return stack[idx].astype(np.float32)
        return stack.astype(np.float32)

    def iter_frames(self, channel: str):
        """Yield (fov, z_idx, file_path, page_idx) for parallel processing."""
        for fov in self.iter_fovs():
            filename = f"{fov.region}_{fov.index}.ome.tiff"
            filepath = self.ome_dir / filename
            if filepath.exists():
                with tifffile.TiffFile(filepath) as tif:
                    channel_idx = self._get_channel_index(tif.ome_metadata, channel)
                    n_channels = len(set(re.findall(r'Name="Fluorescence (\d+) nm Ex"', tif.ome_metadata)))
                    nz = len(tif.pages) // max(n_channels, 1)
                for z_idx in range(nz):
                    page = z_idx * max(n_channels, 1) + channel_idx
                    yield fov, z_idx, filepath, page

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

    def _get_channel_index(self, ome_xml: str, channel: str) -> int:
        pattern = re.compile(r'Name="Fluorescence (\d+) nm Ex"')
        channels = []
        for match in pattern.finditer(ome_xml):
            channels.append(match.group(1))

        seen = set()
        unique_channels = []
        for ch in channels:
            if ch not in seen:
                seen.add(ch)
                unique_channels.append(ch)

        if channel not in unique_channels:
            raise ValueError(
                f"Channel {channel} not found. Available: {unique_channels}"
            )
        return unique_channels.index(channel)
