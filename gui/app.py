#!/usr/bin/env python3
"""Background Subtraction GUI — minimal PyQt5 interface."""

import sys
import os
from pathlib import Path

# Fix Qt plugin path for conda environments on macOS
if sys.platform == "darwin" and "CONDA_PREFIX" in os.environ:
    conda_plugins = Path(os.environ["CONDA_PREFIX"]) / "plugins"
    if conda_plugins.exists() and "QT_PLUGIN_PATH" not in os.environ:
        os.environ["QT_PLUGIN_PATH"] = str(conda_plugins)

import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QGroupBox,
    QFileDialog,
    QProgressBar,
    QComboBox,
    QSlider,
    QSpinBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap


STYLE_SHEET = """
QGroupBox {
    font-weight: bold;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
}
QPushButton#runButton {
    background-color: #0071e3;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
}
QPushButton#runButton:hover {
    background-color: #0077ed;
}
QPushButton#runButton:disabled {
    background-color: #c7c7cc;
}
QPushButton#autoButton {
    background-color: #5856d6;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
}
QPushButton#autoButton:hover {
    background-color: #6866e0;
}
QPushButton#autoButton:disabled {
    background-color: #c7c7cc;
}
QProgressBar {
    border: none;
    border-radius: 4px;
    height: 6px;
}
QProgressBar::chunk {
    background-color: #0071e3;
    border-radius: 4px;
}
"""


class ProcessWorker(QThread):
    """Background worker for batch processing."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, subtractor, channels):
        super().__init__()
        self.subtractor = subtractor
        self.channels = channels

    def run(self):
        try:
            self.subtractor.process_all(
                channels=self.channels,
                progress_callback=lambda cur, tot, msg: self.progress.emit(cur, tot, msg),
            )
            self.finished.emit()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class PreviewWorker(QThread):
    """Single-frame preview worker."""

    finished = pyqtSignal(object, object, object, object)  # original, fg, bg, metrics
    error = pyqtSignal(str)

    def __init__(self, subtractor, channel, frame_idx):
        super().__init__()
        self.subtractor = subtractor
        self.channel = channel
        self.frame_idx = frame_idx

    def run(self):
        try:
            meta = self.subtractor.load_metadata()
            fmt = self.subtractor.detect_format()

            if fmt == "flat_tiffs":
                from bgsub.io import read_frame
                path = meta["file_map"][(self.channel, self.frame_idx)]
                original = read_frame(path).astype(np.float32)
            else:
                from bgsub.io import read_plane
                ch_idx = next(
                    i for i, ch in enumerate(meta["channels"]) if ch["name"] == self.channel
                )
                page = self.frame_idx * meta["n_channels"] + ch_idx
                file_idx = page // meta["n_pages_per_file"]
                page_in_file = page % meta["n_pages_per_file"]
                original = read_plane(meta["tiff_files"][file_idx], page_in_file).astype(
                    np.float32
                )

            fg, bg = self.subtractor.process_single(original)
            from bgsub.metrics import compute_metrics
            metrics = compute_metrics(original, fg, bg)
            self.finished.emit(original, fg, bg, metrics)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class AutoBoxSizeWorker(QThread):
    """Worker to auto-detect optimal box size."""

    finished = pyqtSignal(int, object)  # best_box_size, all_metrics
    error = pyqtSignal(str)

    def __init__(self, subtractor, channel, frame_idx):
        super().__init__()
        self.subtractor = subtractor
        self.channel = channel
        self.frame_idx = frame_idx

    def run(self):
        try:
            meta = self.subtractor.load_metadata()
            fmt = self.subtractor.detect_format()

            if fmt == "flat_tiffs":
                from bgsub.io import read_frame
                path = meta["file_map"][(self.channel, self.frame_idx)]
                image = read_frame(path)
            else:
                from bgsub.io import read_plane
                ch_idx = next(
                    i for i, ch in enumerate(meta["channels"]) if ch["name"] == self.channel
                )
                page = self.frame_idx * meta["n_channels"] + ch_idx
                file_idx = page // meta["n_pages_per_file"]
                page_in_file = page % meta["n_pages_per_file"]
                image = read_plane(meta["tiff_files"][file_idx], page_in_file)

            from bgsub.metrics import suggest_box_size
            best_bs, all_metrics = suggest_box_size(image)
            self.finished.emit(best_bs, all_metrics)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


def ndarray_to_qpixmap(arr: np.ndarray, max_width: int = 400) -> QPixmap:
    """Convert a 2D float array to a QPixmap for display."""
    p1, p99 = np.percentile(arr, [0.5, 99.5])
    normalized = np.clip((arr - p1) / (p99 - p1 + 1e-6) * 255, 0, 255).astype(np.uint8)

    h, w = normalized.shape
    scale = min(1.0, max_width / w)
    if scale < 1.0:
        step_h = max(1, h // int(h * scale))
        step_w = max(1, w // int(w * scale))
        normalized = normalized[::step_h, ::step_w]
        h, w = normalized.shape

    normalized = np.ascontiguousarray(normalized)
    qimg = QImage(normalized.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Background Subtraction")
        self.setMinimumSize(850, 700)
        self.setAcceptDrops(True)

        self._subtractor = None
        self._metadata = None
        self._preview_worker = None
        self._process_worker = None
        self._auto_worker = None
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._do_preview)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Input group ---
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)

        browse_row = QHBoxLayout()
        self._input_label = QLabel("No folder selected")
        self._input_label.setWordWrap(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_input)
        browse_row.addWidget(self._input_label, 1)
        browse_row.addWidget(browse_btn)
        input_layout.addLayout(browse_row)

        self._info_label = QLabel("")
        input_layout.addWidget(self._info_label)
        layout.addWidget(input_group)

        # --- Parameters group ---
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        # Box size
        box_row = QHBoxLayout()
        box_row.addWidget(QLabel("Box Size:"))
        self._box_slider = QSlider(Qt.Horizontal)
        self._box_slider.setRange(10, 500)
        self._box_slider.setValue(50)
        self._box_slider.valueChanged.connect(self._on_box_changed)
        box_row.addWidget(self._box_slider, 1)
        self._box_spin = QSpinBox()
        self._box_spin.setRange(10, 500)
        self._box_spin.setValue(50)
        self._box_spin.valueChanged.connect(self._on_box_spin_changed)
        box_row.addWidget(self._box_spin)
        self._auto_btn = QPushButton("Auto-detect")
        self._auto_btn.setObjectName("autoButton")
        self._auto_btn.setEnabled(False)
        self._auto_btn.clicked.connect(self._on_auto_box)
        box_row.addWidget(self._auto_btn)
        params_layout.addLayout(box_row)

        # Channel checkboxes (populated dynamically)
        self._channel_row = QHBoxLayout()
        self._channel_row.addWidget(QLabel("Channels:"))
        self._channel_checks = []
        self._channel_container = QWidget()
        self._channel_container_layout = QHBoxLayout(self._channel_container)
        self._channel_container_layout.setContentsMargins(0, 0, 0, 0)
        self._channel_row.addWidget(self._channel_container, 1)
        params_layout.addLayout(self._channel_row)

        # Output path
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output:"))
        self._output_label = QLabel("(auto)")
        self._output_label.setWordWrap(True)
        output_row.addWidget(self._output_label, 1)
        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(self._on_browse_output)
        output_row.addWidget(output_btn)
        params_layout.addLayout(output_row)

        layout.addWidget(params_group)

        # --- Preview group ---
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        images_row = QHBoxLayout()
        self._orig_preview = QLabel("Original")
        self._orig_preview.setAlignment(Qt.AlignCenter)
        self._orig_preview.setMinimumSize(380, 280)
        self._orig_preview.setStyleSheet("border: 1px solid #ccc;")
        self._fg_preview = QLabel("Subtracted")
        self._fg_preview.setAlignment(Qt.AlignCenter)
        self._fg_preview.setMinimumSize(380, 280)
        self._fg_preview.setStyleSheet("border: 1px solid #ccc;")
        images_row.addWidget(self._orig_preview)
        images_row.addWidget(self._fg_preview)
        preview_layout.addLayout(images_row)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Frame:"))
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.valueChanged.connect(self._schedule_preview)
        controls_row.addWidget(self._frame_slider, 1)
        self._frame_label = QLabel("0/0")
        controls_row.addWidget(self._frame_label)
        controls_row.addWidget(QLabel("Channel:"))
        self._channel_combo = QComboBox()
        self._channel_combo.currentTextChanged.connect(self._on_channel_combo_changed)
        controls_row.addWidget(self._channel_combo)
        preview_layout.addLayout(controls_row)

        self._metrics_label = QLabel("")
        preview_layout.addWidget(self._metrics_label)

        layout.addWidget(preview_group)

        # --- Run section ---
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run All")
        self._run_btn.setObjectName("runButton")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        run_row.addWidget(self._run_btn)
        layout.addLayout(run_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self.setStyleSheet(STYLE_SHEET)

    # --- Drag and drop ---
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._load_input(path)

    # --- Input ---
    def _on_browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select acquisition folder")
        if folder:
            self._load_input(folder)

    def _load_input(self, path: str):
        from bgsub.core import BackgroundSubtractor

        self._input_label.setText(path)
        self._status_label.setText("Detecting format...")

        try:
            self._subtractor = BackgroundSubtractor(path, box_size=self._box_slider.value())
            fmt = self._subtractor.detect_format()
            self._metadata = self._subtractor.load_metadata()
        except Exception as e:
            self._info_label.setText(f"Error: {e}")
            self._subtractor = None
            self._metadata = None
            self._run_btn.setEnabled(False)
            self._auto_btn.setEnabled(False)
            self._status_label.setText("")
            return

        # Populate channels
        for cb in self._channel_checks:
            self._channel_container_layout.removeWidget(cb)
            cb.deleteLater()
        self._channel_checks.clear()
        self._channel_combo.clear()

        if fmt == "flat_tiffs":
            channels = self._metadata["channels"]
            info = f"Format: flat_tiffs | Channels: {', '.join(channels)}"
            n_frames = list(self._metadata["n_frames"].values())[0]
            info += f" | Frames/ch: {n_frames}"
            info += f" | Size: {self._metadata['shape']}"
        else:
            channels = [ch["name"] for ch in self._metadata["channels"]]
            info = f"Format: ome_tiff | Channels: {', '.join(channels)}"
            info += f" | Z: {self._metadata['n_z']} | T: {self._metadata['n_t']}"
            info += f" | Files: {self._metadata['n_files']}"
            n_frames = self._metadata["n_pages_per_file"] // max(len(channels), 1)

        self._info_label.setText(info)

        for ch in channels:
            cb = QCheckBox(ch)
            cb.setChecked(True)
            self._channel_checks.append(cb)
            self._channel_container_layout.addWidget(cb)
            self._channel_combo.addItem(ch)

        self._frame_slider.setRange(0, max(0, n_frames - 1))
        self._frame_slider.setValue(0)
        self._frame_label.setText(f"0/{max(0, n_frames - 1)}")

        self._output_label.setText(str(self._subtractor.output_path))

        self._run_btn.setEnabled(True)
        self._auto_btn.setEnabled(True)
        self._status_label.setText("Ready.")
        self._schedule_preview()

    # --- Output ---
    def _on_browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder and self._subtractor:
            self._subtractor.output_path = Path(folder)
            self._output_label.setText(folder)

    # --- Box size ---
    def _on_box_changed(self, value):
        self._box_spin.blockSignals(True)
        self._box_spin.setValue(value)
        self._box_spin.blockSignals(False)
        if self._subtractor:
            self._subtractor.box_size = value
        self._schedule_preview()

    def _on_box_spin_changed(self, value):
        self._box_slider.blockSignals(True)
        self._box_slider.setValue(value)
        self._box_slider.blockSignals(False)
        if self._subtractor:
            self._subtractor.box_size = value
        self._schedule_preview()

    def _on_auto_box(self):
        if not self._subtractor or not self._metadata:
            return
        self._auto_btn.setEnabled(False)
        self._status_label.setText("Auto-detecting optimal box size...")

        ch = self._channel_combo.currentText()
        frame = self._frame_slider.value()

        self._auto_worker = AutoBoxSizeWorker(self._subtractor, ch, frame)
        self._auto_worker.finished.connect(self._on_auto_box_done)
        self._auto_worker.error.connect(self._on_worker_error)
        self._auto_worker.start()

    def _on_auto_box_done(self, best_box, all_metrics):
        self._box_slider.setValue(best_box)
        self._auto_btn.setEnabled(True)
        self._status_label.setText(f"Auto-detected box size: {best_box}")

    # --- Preview ---
    def _on_channel_combo_changed(self):
        self._schedule_preview()

    def _schedule_preview(self):
        self._preview_timer.start()

    def _do_preview(self):
        if not self._subtractor or not self._metadata:
            return

        ch = self._channel_combo.currentText()
        frame = self._frame_slider.value()
        self._frame_label.setText(f"{frame}/{self._frame_slider.maximum()}")

        if not ch:
            return

        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.terminate()
            self._preview_worker.wait()

        self._preview_worker = PreviewWorker(self._subtractor, ch, frame)
        self._preview_worker.finished.connect(self._update_preview)
        self._preview_worker.error.connect(self._on_worker_error)
        self._preview_worker.start()

    def _update_preview(self, original, fg, bg, metrics):
        preview_w = self._orig_preview.width() - 4
        self._orig_preview.setPixmap(ndarray_to_qpixmap(original, max_width=preview_w))
        self._fg_preview.setPixmap(
            ndarray_to_qpixmap(np.clip(fg, 0, None), max_width=preview_w)
        )
        self._metrics_label.setText(
            f"SNR: {metrics['snr_improvement']:.2f}x | "
            f"BG uniformity: {metrics['bg_uniformity']:.4f} | "
            f"Signal preservation: {metrics['signal_preservation']:.4f} | "
            f"Neg pixels: {metrics['negative_pixel_pct']:.2f}%"
        )

    # --- Run ---
    def _on_run(self):
        if not self._subtractor:
            return

        selected_channels = [
            cb.text() for cb in self._channel_checks if cb.isChecked()
        ]
        if not selected_channels:
            self._status_label.setText("No channels selected.")
            return

        self._run_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("Processing...")

        self._process_worker = ProcessWorker(self._subtractor, selected_channels)
        self._process_worker.progress.connect(self._on_progress)
        self._process_worker.finished.connect(self._on_run_finished)
        self._process_worker.error.connect(self._on_worker_error)
        self._process_worker.start()

    def _on_progress(self, current, total, message):
        pct = int(100 * current / total) if total > 0 else 0
        self._progress_bar.setValue(pct)
        self._status_label.setText(f"{message} ({current}/{total})")

    def _on_run_finished(self):
        self._run_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._status_label.setText(
            f"Done! Output: {self._subtractor.output_path}"
        )

    def _on_worker_error(self, msg):
        self._run_btn.setEnabled(True)
        self._auto_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"Error: {msg.split(chr(10))[0]}")
        print(f"Worker error:\n{msg}", file=sys.stderr)


def main():
    src_dir = str(Path(__file__).parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
