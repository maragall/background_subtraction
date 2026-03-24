"""BackgroundSubtractor — background estimation and subtraction for microscopy acquisitions."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

from .io import (
    detect_flat_tiffs,
    detect_ome_tiff,
    load_flat_tiffs_metadata,
    load_ome_tiff_metadata,
    read_frame,
    read_plane,
    write_frame,
)
from .metrics import compute_metrics


class BackgroundSubtractor:
    """Orchestrates photutils Background2D subtraction across an acquisition.

    Parameters
    ----------
    input_path : str or Path
        Path to acquisition folder.
    output_path : str or Path, optional
        Where to write results. Defaults to input_path / "background_subtracted".
    box_size : int
        Box size for Background2D grid estimation (pixels).
    """

    def __init__(
        self,
        input_path,
        output_path=None,
        box_size=50,
    ):
        self.input_path = Path(input_path)
        self.output_path = (
            Path(output_path) if output_path else self.input_path / "background_subtracted"
        )
        self.box_size = box_size
        self._format = None
        self._metadata = None

    def detect_format(self) -> str:
        if self._format:
            return self._format
        if detect_ome_tiff(self.input_path):
            self._format = "ome_tiff"
        elif detect_flat_tiffs(self.input_path):
            self._format = "flat_tiffs"
        else:
            raise ValueError(
                f"Unrecognized acquisition format in {self.input_path}. "
                "Expected flat TIFFs or OME-TIFF with acquisition.yaml."
            )
        return self._format

    def load_metadata(self) -> dict:
        if self._metadata:
            return self._metadata
        fmt = self.detect_format()
        if fmt == "flat_tiffs":
            self._metadata = load_flat_tiffs_metadata(self.input_path)
        else:
            self._metadata = load_ome_tiff_metadata(self.input_path)
        return self._metadata

    def process_single(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Background2D on one 2D image. Returns (foreground, background)."""
        img = image.astype(np.float32)
        sigma_clip = SigmaClip(sigma=3.0)
        bkg = Background2D(
            img,
            box_size=(self.box_size, self.box_size),
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=MedianBackground(),
        )
        bg = bkg.background
        fg = img - bg
        return fg, bg

    def process_frame(
        self, channel: str, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load, process, and compute metrics for one frame. Returns (fg, bg, metrics)."""
        meta = self.load_metadata()
        fmt = self.detect_format()

        if fmt == "flat_tiffs":
            path = meta["file_map"][(channel, frame_idx)]
            image = read_frame(path)
        else:
            ch_idx = next(
                i for i, ch in enumerate(meta["channels"]) if ch["name"] == channel
            )
            page = frame_idx * meta["n_channels"] + ch_idx
            file_idx = page // meta["n_pages_per_file"]
            page_in_file = page % meta["n_pages_per_file"]
            image = read_plane(meta["tiff_files"][file_idx], page_in_file)

        fg, bg = self.process_single(image)
        metrics = compute_metrics(image.astype(np.float32), fg, bg)
        return fg, bg, metrics

    def process_all(
        self,
        channels: Optional[list[str]] = None,
        progress_callback: Optional[Callable] = None,
    ):
        """Process entire acquisition and write output TIFFs.

        progress_callback(current, total, message) for GUI integration.
        """
        meta = self.load_metadata()
        fmt = self.detect_format()
        self.output_path.mkdir(parents=True, exist_ok=True)

        if fmt == "flat_tiffs":
            self._process_all_flat(meta, channels, progress_callback)
        else:
            self._process_all_ome(meta, channels, progress_callback)

    def _process_all_flat(self, meta, channels, progress_callback):
        if channels is None:
            channels = meta["channels"]

        total = sum(meta["n_frames"][ch] for ch in channels)
        current = 0

        for ch in channels:
            for frame_idx in meta["frames_per_channel"][ch]:
                path = meta["file_map"][(ch, frame_idx)]
                image = read_frame(path)
                fg, _bg = self.process_single(image)

                out_path = self.output_path / path.name
                write_frame(fg, out_path, dtype=meta["dtype"])

                current += 1
                if progress_callback:
                    progress_callback(current, total, f"{ch} frame {frame_idx}")

    def _process_all_ome(self, meta, channels, progress_callback):
        available_channels = [ch["name"] for ch in meta["channels"]]
        if channels is None:
            channels = available_channels

        ch_indices = [available_channels.index(ch) for ch in channels]
        n_ch = meta["n_channels"]
        total = 0
        for _tf in meta["tiff_files"]:
            total += meta["n_pages_per_file"] * len(ch_indices) // n_ch

        self.output_path.mkdir(parents=True, exist_ok=True)
        current = 0

        for tiff_path in meta["tiff_files"]:
            n_pages = meta["n_pages_per_file"]
            for page_idx in range(n_pages):
                ch_in_page = page_idx % n_ch
                if ch_in_page not in ch_indices:
                    continue

                image = read_plane(tiff_path, page_idx)
                fg, _bg = self.process_single(image)

                out_name = f"{tiff_path.stem}_page{page_idx:04d}.tiff"
                write_frame(fg, self.output_path / out_name, dtype=meta["dtype"])

                current += 1
                if progress_callback:
                    ch_name = available_channels[ch_in_page]
                    progress_callback(current, total, f"{ch_name} page {page_idx}")
