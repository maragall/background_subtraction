"""BackgroundSubtractor — sep-based background subtraction for microscopy acquisitions."""

import itertools
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sep
import tifffile

from .readers import open_acquisition
from .metrics import compute_metrics

# sep uses ~3x the raw frame size in peak memory per worker.
_MEM_MULTIPLIER = 3


def _subtract_background(image: np.ndarray, box_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Run sep.Background on one 2D image. Returns (foreground, background)."""
    img = np.ascontiguousarray(image, dtype=np.float32)
    bkg = sep.Background(img, bw=box_size, bh=box_size, fw=3, fh=3)
    bg = bkg.back()
    fg = img - bg
    return fg, bg


def _process_frame_worker(args):
    """Picklable worker: read frame from file, subtract background, write output."""
    file_path_str, page_idx, output_dir, box_size, dtype_str, out_name = args
    import sep as _sep, numpy as _np, tifffile as _tf

    if page_idx >= 0:
        image = _tf.imread(file_path_str, key=page_idx)
    else:
        image = _tf.imread(file_path_str)

    img = _np.ascontiguousarray(image, dtype=_np.float32)
    bkg = _sep.Background(img, bw=box_size, bh=box_size, fw=3, fh=3)
    fg = img - bkg.back()

    out_path = Path(output_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = _np.iinfo(_np.dtype(dtype_str))
    clipped = _np.clip(fg, info.min, info.max).astype(_np.dtype(dtype_str))
    _tf.imwrite(str(out_path), clipped)
    return out_name


class BackgroundSubtractor:
    """Orchestrates sep-based background subtraction across an acquisition.

    Supports all formats via the readers module:
      - Individual TIFFs (*_Fluorescence_*_nm_Ex*.tiff)
      - OME-TIFF (ome_tiff/*.ome.tiff)
      - CurrentStack (*_stack.tiff)

    Parameters
    ----------
    input_path : str or Path
        Path to acquisition folder.
    output_path : str or Path, optional
        Where to write results. Defaults to input_path / "background_subtracted".
    box_size : int
        Box size for background grid estimation (pixels).
    max_workers : int, optional
        Number of parallel workers. Auto-detected from available memory if None.
    """

    def __init__(
        self,
        input_path,
        output_path=None,
        box_size=50,
        max_workers=None,
    ):
        self.input_path = Path(input_path)
        self.output_path = (
            Path(output_path) if output_path else self.input_path / "background_subtracted"
        )
        self.box_size = box_size
        self.max_workers = max_workers
        self._acq = None

    @property
    def acq(self):
        """Lazy-load the AcquisitionReader."""
        if self._acq is None:
            self._acq = open_acquisition(self.input_path)
        return self._acq

    @property
    def channels(self) -> list[str]:
        return self.acq.metadata.channels

    @property
    def format_name(self) -> str:
        return self.acq.format_name

    def n_frames(self, channel: str) -> int:
        """Number of 2D frames for a channel (across all FOVs)."""
        return sum(1 for _ in self.acq.iter_frames(channel))

    def n_frames_per_fov(self, channel: str) -> int:
        """Number of z-planes / timepoints per FOV for a channel."""
        fovs = list(self.acq.iter_fovs())
        if not fovs:
            return 0
        # Use cached _find_files for individual reader, fallback to iter_frames
        try:
            files = self.acq._find_files(fovs[0], channel)
            return len(files)
        except AttributeError:
            return sum(1 for fov_f, _, _, _ in self.acq.iter_frames(channel)
                       if str(fov_f) == str(fovs[0]))

    def process_single(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run background subtraction on one 2D image."""
        return _subtract_background(image, self.box_size)

    def get_frame(self, channel: str, frame_idx: int) -> np.ndarray:
        """Load a single 2D frame (first FOV, given z/time index)."""
        fovs = list(self.acq.iter_fovs())
        if not fovs:
            raise ValueError("No FOVs found")
        return self.acq.get_frame(fovs[0], channel, frame_idx)

    def process_frame(
        self, channel: str, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Load, process, and compute metrics for one frame.

        Returns (original, foreground, background, metrics).
        """
        image = self.get_frame(channel, frame_idx)
        fg, bg = self.process_single(image)
        metrics = compute_metrics(image, fg, bg)
        return image, fg, bg, metrics

    def _safe_worker_count(self) -> int:
        """Choose worker count based on available memory and frame size."""
        try:
            shape = self.acq.frame_shape
            dtype = self.acq.frame_dtype
            frame_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        except (NotImplementedError, StopIteration):
            frame_bytes = 100 * 1024 * 1024  # assume 100 MB

        per_worker_bytes = frame_bytes * _MEM_MULTIPLIER

        try:
            import psutil
            available = psutil.virtual_memory().available
        except (ImportError, Exception):
            available = 4 * 1024**3

        usable = max(0, available - 2 * 1024**3) * 0.75
        mem_limited = max(1, int(usable / per_worker_bytes))
        cpu_limited = os.cpu_count() or 1
        return min(mem_limited, cpu_limited)

    def process_all(
        self,
        channels: Optional[list[str]] = None,
        progress_callback: Optional[Callable] = None,
        max_workers: Optional[int] = None,
    ):
        """Process entire acquisition and write output TIFFs.

        Uses multiprocessing for parallel frame processing.
        progress_callback(current, total, message) for GUI integration.
        """
        acq = self.acq
        if channels is None:
            channels = acq.metadata.channels

        self.output_path.mkdir(parents=True, exist_ok=True)
        workers = max_workers or self.max_workers or self._safe_worker_count()

        try:
            dtype_str = str(acq.frame_dtype)
        except (NotImplementedError, StopIteration):
            dtype_str = "uint16"

        # Build tasks from iter_frames — each yields (fov, z_idx, file_path, page_idx)
        tasks = []
        for ch in channels:
            for fov, z_idx, file_path, page_idx in acq.iter_frames(ch):
                # Preserve original filename for individual TIFFs
                if page_idx < 0:
                    out_name = file_path.name
                else:
                    out_name = f"{file_path.stem}_page{page_idx:04d}.tiff"
                tasks.append((
                    str(file_path),
                    page_idx,
                    str(self.output_path),
                    self.box_size,
                    dtype_str,
                    out_name,
                ))

        total = len(tasks)
        current = 0

        with ProcessPoolExecutor(max_workers=workers) as pool:
            # Submit in controlled batches to avoid memory blowup.
            # Keep at most 2*workers tasks in flight.
            pending = set()
            task_iter = iter(tasks)
            max_pending = workers * 2

            # Seed the pool
            for t in itertools.islice(task_iter, max_pending):
                pending.add(pool.submit(_process_frame_worker, t))

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    out_name = future.result()
                    current += 1
                    if progress_callback:
                        progress_callback(current, total, out_name)
                # Refill
                for t in itertools.islice(task_iter, len(done)):
                    pending.add(pool.submit(_process_frame_worker, t))
