"""Reader for individual TIFF format (one file per z-slice per channel).

From petakit/Deconvolution — supports both standard and single_band naming.
"""
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from .base import AcquisitionReader, Metadata, FOV, FrameRef


# Standard: {region}_{fov}_{z}_Fluorescence_{wavelength}_nm_Ex.tiff
_STD_PATTERN = re.compile(r"Fluorescence_(\d+)_nm_Ex")
# Single-band: {prefix}_Fluorescence_{wavelength}_nm_Ex_-_single_band_{frame}.tiff
_SB_PATTERN = re.compile(
    r"^(.+?)_Fluorescence_(\d+)_nm_Ex_-_single_band_(\d+)\.tiff?$"
)


def _find_tiff_dir(root: Path) -> Path | None:
    """Find the subdirectory containing matching TIFFs."""
    root = Path(root)
    # Check root itself
    for f in root.iterdir():
        if f.is_file() and _STD_PATTERN.search(f.name):
            return root
    # Check one level of subdirectories
    for d in sorted(root.iterdir()):
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file() and _STD_PATTERN.search(f.name):
                    return d
    return None


def detect_individual(root: Path) -> bool:
    """Check if directory contains individual TIFF format."""
    return _find_tiff_dir(root) is not None


def open_individual(root: Path) -> "IndividualReader":
    """Open an acquisition in individual TIFF format."""
    root = Path(root)
    tiff_dir = _find_tiff_dir(root)
    if tiff_dir is None:
        raise FileNotFoundError(f"No matching TIFFs in {root}")

    channels = set()
    for f in tiff_dir.iterdir():
        if m := _STD_PATTERN.search(f.name):
            channels.add(m.group(1))

    channels = sorted(channels, key=int)

    # Try loading acquisition_parameters.json; fall back to minimal metadata
    json_path = root / "acquisition_parameters.json"
    if not json_path.exists():
        json_path = root / "acquisition parameters.json"

    if json_path.exists():
        metadata = Metadata.from_acquisition_json(json_path, channels)
    else:
        # Infer minimal metadata from files
        # Count z-planes for first FOV+channel to get nz
        first_ch = channels[0] if channels else "000"
        nz = 0
        for subdir in root.iterdir():
            if subdir.is_dir():
                nz = sum(1 for f in subdir.iterdir()
                         if _STD_PATTERN.search(f.name) and first_ch in f.name)
                if nz > 0:
                    break
        metadata = Metadata(
            dxy=1.0, dz=1.0, na=1.0, magnification=1.0,
            channels=channels, nz=max(nz, 1), nt=1,
        )

    return IndividualReader(root, metadata, tiff_dir)


class IndividualReader(AcquisitionReader):
    """Reader for individual TIFF format.

    Supports both standard naming:
        {region}_{fov}_{z}_Fluorescence_{wavelength}_nm_Ex.tiff

    And single-band naming:
        {prefix}_Fluorescence_{wavelength}_nm_Ex_-_single_band_{frame}.tiff
    """

    def __init__(self, root: Path, metadata, tiff_dir: Path):
        super().__init__(root, metadata)
        self._tiff_dir = tiff_dir
        self._file_cache = {}  # (fov_str, channel) -> [(idx, path), ...]

    @property
    def format_name(self) -> str:
        return "individual"

    def iter_fovs(self) -> Iterator[FOV]:
        seen = set()
        pattern = re.compile(r"^([a-zA-Z]+\d+)_(\d+)_\d+_Fluorescence")

        for f in sorted(self._tiff_dir.glob("*_Fluorescence_*_nm_Ex*")):
            if match := pattern.match(f.name):
                region = match.group(1)
                fov_idx = int(match.group(2))
                key = (region, fov_idx)
                if key not in seen:
                    seen.add(key)
                    yield FOV(region=region, index=fov_idx)

    def _find_files(self, fov: FOV, channel: str) -> list[tuple[int, Path]]:
        """Find and sort files for a FOV+channel. Returns [(frame_idx, path), ...].

        Results are cached after first call.
        """
        key = (str(fov), channel)
        if key in self._file_cache:
            return self._file_cache[key]

        pattern = f"{fov.region}_{fov.index}_*_Fluorescence_{channel}_nm_Ex*"
        files = list(self._tiff_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files found for FOV {fov}, channel {channel}"
            )

        # Try single_band index first (timelapse), fall back to z-index
        sb_pattern = re.compile(r"_single_band_(\d+)\.tiff?$")
        z_pattern = re.compile(rf"{fov.region}_{fov.index}_(\d+)_Fluorescence")

        files_with_idx = []
        for f in files:
            sb_match = sb_pattern.search(f.name)
            if sb_match:
                idx = int(sb_match.group(1))
            elif z_match := z_pattern.search(f.name):
                idx = int(z_match.group(1))
            else:
                idx = 0
            files_with_idx.append((idx, f))

        files_with_idx.sort(key=lambda x: x[0])
        self._file_cache[key] = files_with_idx
        return files_with_idx

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        """Load z-stack for given FOV and channel."""
        files_with_z = self._find_files(fov, channel)
        slices = [tifffile.imread(f) for _, f in files_with_z]
        return np.stack(slices, axis=0).astype(np.float32)

    def get_frame(self, fov: FOV, channel: str, z_idx: int) -> np.ndarray:
        """Load a single 2D frame without loading the full stack."""
        files_with_z = self._find_files(fov, channel)
        idx = min(z_idx, len(files_with_z) - 1)
        return tifffile.imread(files_with_z[idx][1]).astype(np.float32)

    def iter_frames_for_fov(self, fov: FOV, channel: str):
        """Yield FrameRef for every 2D frame in this FOV+channel."""
        for z_idx, path in self._find_files(fov, channel):
            yield FrameRef(fov=fov, frame_idx=z_idx, file_path=path, page_idx=None)

    def n_frames_per_fov(self, fov: FOV, channel: str) -> int:
        return len(self._find_files(fov, channel))

    def iter_frames(self, channel: str):
        for fov in self.iter_fovs():
            yield from self.iter_frames_for_fov(fov, channel)

    @property
    def frame_shape(self) -> tuple:
        f = next(self._tiff_dir.glob("*_Fluorescence_*"))
        with tifffile.TiffFile(str(f)) as tf:
            return tf.pages[0].shape

    @property
    def frame_dtype(self):
        f = next(self._tiff_dir.glob("*_Fluorescence_*"))
        with tifffile.TiffFile(str(f)) as tf:
            return tf.pages[0].dtype
