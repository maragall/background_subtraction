"""Background Subtraction GUI — Cephla-styled PyQt5 interface."""
import sys
import os
from pathlib import Path

if sys.platform == "darwin" and "CONDA_PREFIX" in os.environ:
    conda_plugins = Path(os.environ["CONDA_PREFIX"]) / "plugins"
    if conda_plugins.exists() and "QT_PLUGIN_PATH" not in os.environ:
        os.environ["QT_PLUGIN_PATH"] = str(conda_plugins)

import cv2
import numpy as np
import sep
import tifffile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar,
    QGroupBox, QSpinBox, QSlider,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter

from bgsub.core import BackgroundSubtractor

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
QPushButton {
    border: 1px solid #999;
    border-radius: 6px;
    padding: 8px 16px;
    background: white;
}
QPushButton:hover {
    background: #f0f0f0;
    border-color: #666;
}
QPushButton:disabled {
    background: #f5f5f5;
    border-color: #ccc;
    color: #aaa;
}
QPushButton:pressed {
    background: #e0e0e0;
}
QProgressBar {
    border: none;
    border-radius: 4px;
    height: 6px;
    background: rgba(49, 196, 243, 0.15);
}
QProgressBar::chunk {
    background-color: rgba(49, 196, 243, 0.6);
    border-radius: 4px;
}
"""

SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 15px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(128, 128, 128, 0.25),
        stop:1 rgba(128, 128, 128, 0.1)
    );
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 38px;
    background: #999999;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(100, 100, 100, 0.25),
        stop:1 rgba(100, 100, 100, 0.1)
    );
}
"""

PREVIEW_W = 400
_NORMALIZATION_PERCENTILE = 99.0   # robust upper bound, ignores hot pixels
_NORMALIZATION_SAMPLES = 10        # enough frames for stable median estimate
_MOVIE_FPS = 30                    # standard playback rate
_MOVIE_FONT_SCALE = 0.8
_MOVIE_FONT_THICKNESS = 2
_THUMBNAIL_PERCENTILES = (0.5, 99) # preview contrast window
_PREVIEW_DEBOUNCE_MS = 200         # coalesce slider drags

_DROP_LABEL_IDLE_STYLE = (
    "QLabel { border: 2px dashed #aaa; border-radius: 6px; "
    "color: #888; background: #fafafa; }"
)
_DROP_LABEL_HOVER_STYLE = (
    "QLabel { border: 2px dashed #34c759; border-radius: 6px; "
    "color: #34c759; background: #f0fff4; }"
)
_DROP_LABEL_LOADED_STYLE = (
    "QLabel { border: 2px solid #34c759; border-radius: 6px; "
    "color: #333; background: #f0fff4; }"
)


# ── Workers ───────────────────────────────────────────────────────────────

def _make_thumbnail(arr, max_w=PREVIEW_W):
    """Downsample + normalize to uint8. Runs in worker thread."""
    h, w = arr.shape
    scale = min(1.0, max_w / w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(arr.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    p_lo, p_hi = np.percentile(small, _THUMBNAIL_PERCENTILES)
    out = np.clip((small - p_lo) / (p_hi - p_lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


class PreviewWorker(QThread):
    finished = pyqtSignal(bytes, int, int, bytes, int, int, dict)
    error = pyqtSignal(str)

    def __init__(self, subtractor, channel, frame_idx):
        super().__init__()
        self.subtractor = subtractor
        self.channel = channel
        self.frame_idx = frame_idx

    def run(self):
        try:
            original, fg, bg, metrics = self.subtractor.process_frame(
                self.channel, self.frame_idx
            )
            before = _make_thumbnail(original)
            after = _make_thumbnail(np.clip(fg, 0, None))
            self.finished.emit(
                before.tobytes(), before.shape[1], before.shape[0],
                after.tobytes(), after.shape[1], after.shape[0],
                metrics,
            )
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class ProcessWorker(QThread):
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
                progress_callback=lambda c, t, m: self.progress.emit(c, t, m),
            )
            self.finished.emit()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class MovieWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, subtractor, output_path):
        super().__init__()
        self.subtractor = subtractor
        self.output_path = Path(output_path)

    def _read_ref(self, ref):
        if ref.page_idx is None:
            return tifffile.imread(str(ref.file_path))
        return tifffile.imread(str(ref.file_path), key=ref.page_idx)

    def _compute_normalization(self, refs):
        """Sample frames evenly, return (raw_hi, fg_hi) percentile medians."""
        step = max(1, len(refs) // _NORMALIZATION_SAMPLES)
        sample_refs = [refs[i] for i in range(0, len(refs), step)]
        raw_his, fg_his = [], []
        for ref in sample_refs:
            img = self._read_ref(ref).astype(np.float32)
            fg, _ = self.subtractor.process_single(img)
            fg_pos = np.clip(fg, 0, None)
            raw_his.append(np.percentile(img, _NORMALIZATION_PERCENTILE))
            pos = fg_pos[fg_pos > 0]
            if len(pos) > 0:
                fg_his.append(np.percentile(pos, _NORMALIZATION_PERCENTILE))
        return (
            float(np.median(raw_his)) if raw_his else 1.0,
            float(np.median(fg_his)) if fg_his else 1.0,
        )

    def run(self):
        try:
            acq = self.subtractor.acq
            channels = acq.metadata.channels
            fovs = list(acq.iter_fovs())
            if not fovs:
                self.error.emit("No FOVs found")
                return

            self.output_path.mkdir(parents=True, exist_ok=True)

            for ch in channels:
                refs = list(acq.iter_frames_for_fov(fovs[0], ch))
                if not refs:
                    continue

                first = self._read_ref(refs[0])
                h, w = first.shape
                # Half-res per pane, two panes side by side, even dimensions
                pw = (w // 2 // 2) * 2
                ph = (h // 2 // 2) * 2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = self.output_path / f"bgsub_{ch}.mp4"
                writer = cv2.VideoWriter(
                    str(out_path), fourcc, _MOVIE_FPS, (pw * 2, ph), isColor=False
                )
                if not writer.isOpened():
                    self.error.emit(f"Failed to open video writer for {out_path}")
                    return

                raw_hi, fg_hi = self._compute_normalization(refs)

                for i, ref in enumerate(refs):
                    img = self._read_ref(ref).astype(np.float32)
                    fg, _ = self.subtractor.process_single(img)
                    fg = np.clip(fg, 0, None)

                    raw_u8 = np.clip(img / (raw_hi + 1e-6) * 255, 0, 255).astype(np.uint8)
                    fg_u8 = np.clip(fg / (fg_hi + 1e-6) * 255, 0, 255).astype(np.uint8)

                    raw_r = cv2.resize(raw_u8, (pw, ph), interpolation=cv2.INTER_AREA)
                    fg_r = cv2.resize(fg_u8, (pw, ph), interpolation=cv2.INTER_AREA)
                    frame = np.hstack([raw_r, fg_r])
                    cv2.putText(
                        frame, "Raw", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, _MOVIE_FONT_SCALE, 255,
                        _MOVIE_FONT_THICKNESS,
                    )
                    cv2.putText(
                        frame, "Subtracted", (pw + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, _MOVIE_FONT_SCALE, 255,
                        _MOVIE_FONT_THICKNESS,
                    )
                    writer.write(frame)

                    if (i + 1) % _MOVIE_FPS == 0 or i == len(refs) - 1:
                        self.progress.emit(f"{ch} {i+1}/{len(refs)}")
                writer.release()

            self.finished.emit(str(self.output_path))
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ── Axis Slider ───────────────────────────────────────────────────────────

class AxisSlider(QWidget):
    """Label + slider + spinbox for one axis. Only visible when axis has >1 value."""
    valueChanged = pyqtSignal(int)

    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self._label = QLabel(label_text)
        self._label.setFixedWidth(50)
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider, 1)

        self._spin = QSpinBox()
        self._spin.setMinimum(0)
        self._spin.setMaximum(0)
        self._spin.setFixedWidth(70)
        self._spin.valueChanged.connect(self._on_spin)
        layout.addWidget(self._spin)

        self._info = QLabel("0/0")
        self._info.setFixedWidth(60)
        self._info.setStyleSheet("color: #86868b; font-size: 11px;")
        layout.addWidget(self._info)

        self.setVisible(False)

    def setup(self, max_val, start=None):
        if max_val <= 0:
            self.setVisible(False)
            return
        self.setVisible(True)
        self._slider.blockSignals(True)
        self._spin.blockSignals(True)
        self._slider.setMaximum(max_val)
        self._spin.setMaximum(max_val)
        val = start if start is not None else max_val // 2
        self._slider.setValue(val)
        self._spin.setValue(val)
        self._info.setText(f"{val}/{max_val}")
        self._slider.blockSignals(False)
        self._spin.blockSignals(False)

    def _on_slider(self, val):
        self._spin.blockSignals(True)
        self._spin.setValue(val)
        self._spin.blockSignals(False)
        self._info.setText(f"{val}/{self._slider.maximum()}")
        self.valueChanged.emit(val)

    def _on_spin(self, val):
        self._slider.blockSignals(True)
        self._slider.setValue(val)
        self._slider.blockSignals(False)
        self._info.setText(f"{val}/{self._slider.maximum()}")
        self.valueChanged.emit(val)

    def set_label(self, text: str):
        self._label.setText(text)

    def value(self):
        return self._slider.value()

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)
        self._spin.setEnabled(enabled)


# ── Main Window ───────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cephla Background Subtraction")
        self.setMinimumWidth(860)

        self._subtractor = None
        self._preview_worker = None
        self._process_worker = None
        self._movie_worker = None
        self._pending_movie = False  # True = run movie after subtraction finishes

        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(_PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._do_preview)

        self._setup_ui()
        self._set_cephla_icon()
        self.setStyleSheet(STYLE_SHEET)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Acquisition ──────────────────────────────────────────────
        acq_group = QGroupBox("Acquisition")
        acq_layout = QVBoxLayout(acq_group)

        self.drop_label = QLabel("Drop acquisition folder here\nor click to browse")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFixedHeight(60)
        self.drop_label.setStyleSheet(_DROP_LABEL_IDLE_STYLE)
        self.drop_label.setCursor(Qt.PointingHandCursor)
        self.drop_label.setAcceptDrops(True)
        self.drop_label.mousePressEvent = lambda _: self._browse_acquisition()
        self.drop_label.dragEnterEvent = self._drag_enter
        self.drop_label.dragLeaveEvent = self._drag_leave
        self.drop_label.dropEvent = self._drop
        acq_layout.addWidget(self.drop_label)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #86868b; font-size: 11px;")
        acq_layout.addWidget(self.info_label)

        layout.addWidget(acq_group)

        # ── Preview ──────────────────────────────────────────────────
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        images_row = QHBoxLayout()
        self._before_label = QLabel("Before")
        self._before_label.setAlignment(Qt.AlignCenter)
        self._before_label.setMinimumSize(PREVIEW_W, 300)
        self._before_label.setStyleSheet("border: 1px solid #ddd; background: #111;")
        self._after_label = QLabel("After")
        self._after_label.setAlignment(Qt.AlignCenter)
        self._after_label.setMinimumSize(PREVIEW_W, 300)
        self._after_label.setStyleSheet("border: 1px solid #ddd; background: #111;")
        images_row.addWidget(self._before_label)
        images_row.addWidget(self._after_label)
        preview_layout.addLayout(images_row)

        # Channel selector
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.setEnabled(False)
        self.channel_combo.currentTextChanged.connect(self._schedule_preview)
        ch_row.addWidget(self.channel_combo, 1)
        preview_layout.addLayout(ch_row)

        # Axis sliders (ndviewer-style, with spinbox for punch-in)
        slider_container = QWidget()
        slider_container.setStyleSheet(SLIDER_STYLE)
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(2)

        self._fov_slider = AxisSlider("FOV")
        self._fov_slider.valueChanged.connect(self._schedule_preview)
        slider_layout.addWidget(self._fov_slider)

        self._frame_slider = AxisSlider("Frame")
        self._frame_slider.valueChanged.connect(self._schedule_preview)
        slider_layout.addWidget(self._frame_slider)

        preview_layout.addWidget(slider_container)

        self._metrics_label = QLabel("")
        self._metrics_label.setStyleSheet("color: #86868b; font-size: 11px;")
        preview_layout.addWidget(self._metrics_label)

        layout.addWidget(preview_group)

        # ── Parameters ───────────────────────────────────────────────
        params_group = QGroupBox("Parameters")
        params_layout = QHBoxLayout(params_group)

        params_layout.addWidget(QLabel("Box Size:"))
        self.box_spin = QSpinBox()
        self.box_spin.setRange(10, 500)
        self.box_spin.setValue(50)
        self.box_spin.setToolTip("Background mesh box size in pixels")
        self.box_spin.valueChanged.connect(self._schedule_preview)
        params_layout.addWidget(self.box_spin)

        params_layout.addStretch()

        params_layout.addWidget(QLabel("Output:"))
        self.output_label = QLabel("(auto)")
        self.output_label.setStyleSheet("color: #86868b;")
        params_layout.addWidget(self.output_label)
        output_btn = QPushButton("Change...")
        output_btn.setToolTip("Choose where subtracted files are saved")
        output_btn.clicked.connect(self._select_output)
        params_layout.addWidget(output_btn)

        layout.addWidget(params_group)

        # ── Actions ──────────────────────────────────────────────────
        action_row = QHBoxLayout()

        self.run_btn = QPushButton("Run Background Subtraction")
        self.run_btn.setEnabled(False)
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setToolTip("Process all channels and frames")
        self.run_btn.clicked.connect(self._run_processing)
        action_row.addWidget(self.run_btn)

        action_row.addStretch()

        self.movie_btn = QPushButton("Generate Movie")
        self.movie_btn.setEnabled(False)
        self.movie_btn.setCursor(Qt.PointingHandCursor)
        self.movie_btn.setToolTip(
            "Generate a side-by-side before/after MP4 for each channel"
        )
        self.movie_btn.clicked.connect(self._generate_movie)
        action_row.addWidget(self.movie_btn)

        layout.addLayout(action_row)

        # ── Progress ─────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #86868b;")
        layout.addWidget(self.status_label)

        self.view_btn = QPushButton("View Output")
        self.view_btn.setVisible(False)
        self.view_btn.setCursor(Qt.PointingHandCursor)
        self.view_btn.clicked.connect(self._view_output)
        layout.addWidget(self.view_btn)

        layout.addStretch()

        # ── Branding ─────────────────────────────────────────────────
        brand_widget = QWidget()
        brand_layout = QHBoxLayout(brand_widget)
        brand_layout.setContentsMargins(0, 4, 0, 6)
        brand_layout.setSpacing(5)
        brand_layout.addStretch()
        logo_path = Path(__file__).parent / "cephla_logo.svg"
        if logo_path.exists():
            try:
                from PyQt5.QtSvg import QSvgRenderer
                logo_label = QLabel()
                renderer = QSvgRenderer(str(logo_path))
                pm = QPixmap(16, 16)
                pm.fill(Qt.transparent)
                p = QPainter(pm)
                renderer.render(p)
                p.end()
                logo_label.setPixmap(pm)
                brand_layout.addWidget(logo_label)
            except ImportError:
                pass
        brand_text = QLabel("cephla")
        brand_text.setStyleSheet("color: #31c4f3; font-size: 10px; letter-spacing: 3px;")
        brand_layout.addWidget(brand_text)
        brand_layout.addStretch()
        layout.addWidget(brand_widget)

    def _set_cephla_icon(self):
        logo_path = Path(__file__).parent / "cephla_logo.svg"
        if logo_path.exists():
            try:
                from PyQt5.QtSvg import QSvgRenderer
                renderer = QSvgRenderer(str(logo_path))
                pixmap = QPixmap(64, 64)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                renderer.render(painter)
                painter.end()
                self.setWindowIcon(QIcon(pixmap))
            except ImportError:
                pass

    # ── Drag and drop ─────────────────────────────────────────────────

    def _drag_enter(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).is_dir():
                    event.acceptProposedAction()
                    self.drop_label.setStyleSheet(_DROP_LABEL_HOVER_STYLE)
                    return
        event.ignore()

    def _drag_leave(self, event):
        if self._subtractor:
            self.drop_label.setStyleSheet(_DROP_LABEL_LOADED_STYLE)
        else:
            self.drop_label.setStyleSheet(_DROP_LABEL_IDLE_STYLE)

    def _drop(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).is_dir():
                self._load_acquisition(path)
                return
        self._drag_leave(event)

    # ── Acquisition ───────────────────────────────────────────────────

    def _browse_acquisition(self):
        path = QFileDialog.getExistingDirectory(self, "Select Acquisition Folder")
        if path:
            self._load_acquisition(path)

    def _load_acquisition(self, path):
        self.status_label.setText("Loading...")
        QApplication.processEvents()

        try:
            self._subtractor = BackgroundSubtractor(
                path, box_size=self.box_spin.value()
            )
            acq = self._subtractor.acq
        except Exception as e:
            self.info_label.setText(f"Error: {e}")
            self._subtractor = None
            self.status_label.setText("")
            return

        self.drop_label.setText(Path(path).name)
        self.drop_label.setStyleSheet(_DROP_LABEL_LOADED_STYLE)

        channels = self._subtractor.channels
        meta = acq.metadata

        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        for ch in channels:
            self.channel_combo.addItem(ch)
        self.channel_combo.setEnabled(True)
        self.channel_combo.blockSignals(False)

        # Setup axis sliders
        n_fovs = sum(1 for _ in acq.iter_fovs())
        n_frames = self._subtractor.n_frames_per_fov(channels[0])

        self._fov_slider.setup(max(0, n_fovs - 1), start=0)

        has_json = (Path(path) / "acquisition_parameters.json").exists() or \
                   (Path(path) / "acquisition parameters.json").exists()
        if has_json and meta.nz > 1:
            self._frame_slider.set_label("Z")
        elif has_json and meta.nt > 1:
            self._frame_slider.set_label("Time")
        else:
            self._frame_slider.set_label("FOV")
        self._frame_slider.setup(max(0, n_frames - 1), start=n_frames // 2)

        self.info_label.setText(
            f"Format: {self._subtractor.format_name} | "
            f"Channels: {', '.join(channels)} | "
            f"FOVs: {n_fovs} | Frames/FOV: {n_frames}"
        )
        self.output_label.setText(str(self._subtractor.output_path))

        self.run_btn.setEnabled(True)
        self.movie_btn.setEnabled(True)
        self.status_label.setText("")

        QTimer.singleShot(50, self._do_preview)

    def _select_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path and self._subtractor:
            self._subtractor.output_path = Path(path)
            self.output_label.setText(path)

    # ── Preview ───────────────────────────────────────────────────────

    def _get_frame_idx(self):
        return self._frame_slider.value()

    def _schedule_preview(self):
        if self._subtractor:
            self._preview_timer.start()

    def _do_preview(self):
        if not self._subtractor:
            return
        ch = self.channel_combo.currentText()
        if not ch:
            return

        self._subtractor.box_size = self.box_spin.value()
        frame = self._get_frame_idx()

        if self._preview_worker and self._preview_worker.isRunning():
            return

        self._preview_worker = PreviewWorker(self._subtractor, ch, frame)
        self._preview_worker.finished.connect(self._update_preview)
        self._preview_worker.error.connect(self._on_error)
        self._preview_worker.start()

    def _update_preview(self, before_bytes, bw, bh, after_bytes, aw, ah, metrics):
        before_img = QImage(before_bytes, bw, bh, bw, QImage.Format_Grayscale8)
        after_img = QImage(after_bytes, aw, ah, aw, QImage.Format_Grayscale8)
        self._before_label.setPixmap(QPixmap.fromImage(before_img))
        self._after_label.setPixmap(QPixmap.fromImage(after_img))
        self._metrics_label.setText(
            f"SNR: {metrics['snr_improvement']:.2f}x  |  "
            f"BG uniformity: {metrics['bg_uniformity']:.4f}  |  "
            f"Signal preservation: {metrics['signal_preservation']:.4f}  |  "
            f"Neg pixels: {metrics['negative_pixel_pct']:.2f}%"
        )

    # ── Buttons disable/enable ────────────────────────────────────────

    def _set_running(self, running):
        has_sub = self._subtractor is not None
        self.run_btn.setEnabled(not running and has_sub)
        self.movie_btn.setEnabled(not running and has_sub)
        self._fov_slider.setEnabled(not running)
        self._frame_slider.setEnabled(not running)
        self.channel_combo.setEnabled(not running and has_sub)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(0)

    # ── Full processing ───────────────────────────────────────────────

    def _run_processing(self):
        if not self._subtractor:
            return
        self._subtractor.box_size = self.box_spin.value()
        channels = self._subtractor.channels
        if not channels:
            return
        self._process_worker = ProcessWorker(self._subtractor, channels)
        self._process_worker.progress.connect(self._on_progress)
        self._process_worker.finished.connect(self._on_run_finished)
        self._process_worker.error.connect(self._on_error)
        self._set_running(True)
        self.progress_bar.setMaximum(100)
        self._process_worker.start()

    def _on_progress(self, current, total, message):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"{message} ({current}/{total})")

    def _on_run_finished(self):
        self._set_running(False)
        self.view_btn.setVisible(True)
        self.status_label.setText(f"Done! Output: {self._subtractor.output_path}")

        # If movie was pending, chain into movie generation
        if self._pending_movie:
            self._pending_movie = False
            self._start_movie()

    # ── Generate Movie ────────────────────────────────────────────────

    def _has_subtracted_output(self):
        """Check if the output folder already has subtracted images."""
        out = self._subtractor.output_path
        if not out.exists():
            return False
        return any(out.glob("*.tiff")) or any(out.glob("*.tif"))

    def _generate_movie(self):
        if not self._subtractor:
            return
        self._subtractor.box_size = self.box_spin.value()

        if self._has_subtracted_output():
            self._start_movie()
        else:
            self._pending_movie = True
            self.status_label.setText("Running background subtraction first...")
            self._run_processing()

    def _start_movie(self):
        movie_dir = self._subtractor.output_path.parent / "movies"
        self._movie_worker = MovieWorker(self._subtractor, movie_dir)
        self._movie_worker.progress.connect(
            lambda msg: self.status_label.setText(f"Movie: {msg}")
        )
        self._movie_worker.finished.connect(self._on_movie_finished)
        self._movie_worker.error.connect(self._on_error)
        self._set_running(True)
        self._movie_worker.start()

    def _on_movie_finished(self, output_path):
        self._set_running(False)
        self.status_label.setText(f"Movies saved to {output_path}")

    # ── View output ───────────────────────────────────────────────────

    def _view_output(self):
        import subprocess
        output_dir = str(self._subtractor.output_path)
        if sys.platform == "darwin":
            subprocess.Popen(["open", output_dir])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", output_dir])
        else:
            subprocess.Popen(["xdg-open", output_dir])

    # ── Error handling ────────────────────────────────────────────────

    def _on_error(self, message):
        self._set_running(False)
        self.status_label.setText(f"Error: {message.split(chr(10))[0]}")
        print(f"Worker error:\n{message}", file=sys.stderr)


# ── Entry point ───────────────────────────────────────────────────────────

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
