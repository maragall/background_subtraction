"""BackgroundSubtractor — sep-based background subtraction for microscopy acquisitions."""

import itertools
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import psutil
import sep
import tifffile

from .metrics import compute_metrics
from .readers import open_acquisition

# sep peak ~2x input + 1x output buffer per worker.
_MEM_PER_WORKER_MULTIPLIER = 3
# Leave 2 GB for OS + main process.
_MEM_HEADROOM_BYTES = 2 * 1024**3
# Don't fully saturate available memory; reserve 25% for spikes.
_MEM_USE_FRACTION = 0.75
# Keep workers fed without unbounded queue growth.
_PENDING_TASKS_PER_WORKER = 2
# sep's documented filter-window default.
_SEP_FILTER_SIZE = 3
# Cephla microscopy default; user-overridable in GUI/API.
_DEFAULT_BOX_SIZE = 50


def _run_sep(image: np.ndarray, box_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Run sep.Background on one 2D image. Returns (foreground, background) in float32."""
    img = np.ascontiguousarray(image, dtype=np.float32)
    bkg = sep.Background(
        img, bw=box_size, bh=box_size, fw=_SEP_FILTER_SIZE, fh=_SEP_FILTER_SIZE
    )
    bg = bkg.back()
    return img - bg, bg


def _process_frame_worker(args):
    """Picklable worker: read one frame, subtract background, write output TIFF."""
    file_path, page_idx, output_dir, box_size, dtype_str, out_name = args
    image = (
        tifffile.imread(file_path)
        if page_idx is None
        else tifffile.imread(file_path, key=page_idx)
    )
    fg, _ = _run_sep(image, box_size)
    out_path = Path(output_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = np.iinfo(np.dtype(dtype_str))
    out = np.clip(fg, info.min, info.max).astype(np.dtype(dtype_str))
    tifffile.imwrite(str(out_path), out)
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
        box_size: int = _DEFAULT_BOX_SIZE,
        max_workers: Optional[int] = None,
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
        return sum(1 for _ in self.acq.iter_frames(channel))

    def n_frames_per_fov(self, channel: str) -> int:
        fovs = list(self.acq.iter_fovs())
        return self.acq.n_frames_per_fov(fovs[0], channel) if fovs else 0

    def process_single(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _run_sep(image, self.box_size)

    def get_frame(self, channel: str, frame_idx: int) -> np.ndarray:
        fovs = list(self.acq.iter_fovs())
        if not fovs:
            raise ValueError("No FOVs found")
        return self.acq.get_frame(fovs[0], channel, frame_idx)

    def process_frame(
        self, channel: str, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        image = self.get_frame(channel, frame_idx)
        fg, bg = self.process_single(image)
        metrics = compute_metrics(image, fg, bg)
        return image, fg, bg, metrics

    def _safe_worker_count(self) -> int:
        shape = self.acq.frame_shape
        dtype = self.acq.frame_dtype
        frame_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        available = psutil.virtual_memory().available
        usable = max(0, available - _MEM_HEADROOM_BYTES) * _MEM_USE_FRACTION
        mem_limited = max(1, int(usable / (frame_bytes * _MEM_PER_WORKER_MULTIPLIER)))
        return min(mem_limited, os.cpu_count() or 1)

    def process_all(
        self,
        channels: Optional[list[str]] = None,
        progress_callback: Optional[Callable] = None,
        max_workers: Optional[int] = None,
    ):
        """Process entire acquisition and write output TIFFs.

        progress_callback(current, total, message) is invoked from the main thread
        as each frame completes.
        """
        acq = self.acq
        if channels is None:
            channels = acq.metadata.channels

        self.output_path.mkdir(parents=True, exist_ok=True)
        workers = max_workers or self.max_workers or self._safe_worker_count()
        dtype_str = str(acq.frame_dtype)

        tasks = []
        for ch in channels:
            for ref in acq.iter_frames(ch):
                if ref.page_idx is None:
                    out_name = ref.file_path.name
                else:
                    out_name = f"{ref.file_path.stem}_page{ref.page_idx:04d}.tiff"
                tasks.append((
                    str(ref.file_path),
                    ref.page_idx,
                    str(self.output_path),
                    self.box_size,
                    dtype_str,
                    out_name,
                ))

        total = len(tasks)
        current = 0
        max_pending = workers * _PENDING_TASKS_PER_WORKER

        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = set()
            task_iter = iter(tasks)
            for t in itertools.islice(task_iter, max_pending):
                pending.add(pool.submit(_process_frame_worker, t))

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    out_name = future.result()
                    current += 1
                    if progress_callback:
                        progress_callback(current, total, out_name)
                for t in itertools.islice(task_iter, len(done)):
                    pending.add(pool.submit(_process_frame_worker, t))
