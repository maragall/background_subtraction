# bgsub — production-readiness cleanup

**Date:** 2026-05-02
**Scope:** `src/bgsub/`, `gui/`, `pyproject.toml`, new `tests/`, new `README.md`
**Goal:** Remove dead code, fix correctness bugs, eliminate leaky abstractions, name and justify every magic constant, add a synthetic-fixture test suite. The sep-based algorithm and the GUI's overall flow are not changing.

---

## 1. Problems being fixed

### Correctness
1. **`suggest_box_size` is broken.** It picks the box size that maximizes `corrcoef(original, foreground)`. That correlation is highest when foreground ≈ original — i.e., when the smallest amount of background was subtracted. The "Auto-detect" button is biased toward over-large box sizes. (`src/bgsub/metrics.py:51`)
2. **Missing required deps.** `opencv-python` is hard-required by both preview and movie generation but absent from `pyproject.toml`. `psutil` is referenced with a soft `try/except ImportError` fallback; we will require it.
3. **Missing `README.md`.** Referenced by `pyproject.toml` but not present.

### Dead / stale code
4. `src/bgsub/io/` (`flat_tiffs.py`, `ome_tiff.py`, `__init__.py`) — no in-tree consumer; superseded by `src/bgsub/readers/`.
5. `pyyaml` dependency — only used by the dead `io/` module.
6. `import shutil` in `core.py` — unused.
7. `scripts/explore_*.py`, `scripts/quality_metrics.py` — old exploration code with deps (`photutils`, `skimage`) not declared in `pyproject.toml`.

### Leaky abstractions
8. `acq._find_files(...)` is called from `core.py` (`n_frames_per_fov`) and `gui/app.py` (`MovieWorker`). It exists only on `IndividualReader` — would crash on OME-TIFF or CurrentStack. The base class doesn't declare it. The current `n_frames_per_fov` masks this with `try/except AttributeError`.
9. `gui.app.MovieWorker` reimplements the entire sep pipeline instead of calling `BackgroundSubtractor.process_single`.
10. `_subtract_background` and `_process_frame_worker` in `core.py` duplicate the same sep call logic. `sep.Background(... fw=3, fh=3)` appears in **5 places** across the codebase.
11. GUI axis-state hack: `_t_slider._label.setText("Z")` (private-attribute access) and `_z_slider.isVisible()` (visibility used as state) for deciding which slider holds the frame index.
12. `page_idx = -1` sentinel for "single-page TIFF" — should be `None`.

### Baseless / undocumented constants
- `_MEM_MULTIPLIER = 3`
- `available - 2 * 1024**3` (memory headroom)
- `* 0.75` (memory use fraction)
- `4 * 1024**3` (psutil-missing fallback)
- `100 MB` (frame-size fallback)
- `box_size = 50` (default, hardcoded twice)
- `fw=3, fh=3` (sep filter, in 5 places)
- `[25, 50, 100, 200, 400]` (auto-detect candidates)
- `workers * 2` (pool backpressure)
- 99th-percentile, 10-sample movie normalization
- `(0.5, 99)` thumbnail percentiles
- `fps = 30` (movie)
- Drop-label CSS strings repeated three times inline

### Other
- In-method imports in `gui/app.py` (`from bgsub.core import BackgroundSubtractor` inside `_load_acquisition`).
- `except (ImportError, Exception)` (redundant) and `except (NotImplementedError, StopIteration)` (swallows real errors) in `core._safe_worker_count`.
- `OMETiffReader.iter_frames` re-parses OME-XML for every FOV.
- `CurrentStackReader` re-reads JSON page descriptions multiple times per stack.
- `IndividualReader.frame_shape` / `frame_dtype` glob `*.tiff`, can pick up non-acquisition files.
- No tests at all.

---

## 2. Design

### 2.1 Module structure

**Delete:**
- `src/bgsub/io/` (entire directory)
- `scripts/` directory (all of `explore_background2d.py`, `explore_rolling_ball.py`, `quality_metrics.py`, `__init__.py`)
- `"scripts"` entry from `[tool.hatch.build.targets.wheel].packages`
- `pyyaml` from `pyproject.toml` deps
- `import shutil` from `core.py`

**Add:**
- `README.md` — minimal: install, GUI launch, supported formats, link to repo. Used by `pyproject.toml`'s `readme` field.
- `tests/` directory (see §2.7).
- `opencv-python>=4.5` and `psutil>=5.9` to required deps.

### 2.2 Public reader API

In `src/bgsub/readers/base.py`:

```python
@dataclass(frozen=True)
class FrameRef:
    fov: FOV
    frame_idx: int          # z-index or t-index — interpretation is reader-specific
    file_path: Path
    page_idx: int | None    # None = single-page TIFF; int = multi-page page index
```

`AcquisitionReader` gains two new abstract methods, replacing the leaky private call:

```python
@abstractmethod
def iter_frames_for_fov(self, fov: FOV, channel: str) -> Iterator[FrameRef]:
    """Yield FrameRef for every 2D frame in this FOV+channel, in frame_idx order."""

@abstractmethod
def n_frames_per_fov(self, fov: FOV, channel: str) -> int:
    """Number of 2D frames for one FOV+channel."""
```

`iter_frames(channel)` is no longer abstract. Default implementation:

```python
def iter_frames(self, channel: str) -> Iterator[FrameRef]:
    for fov in self.iter_fovs():
        yield from self.iter_frames_for_fov(fov, channel)
```

(Existing `iter_frames` returns 4-tuples; we change it to yield `FrameRef`. Both `core.py` and `gui/app.py` are updated to consume `FrameRef`.)

`frame_shape` and `frame_dtype` remain abstract (each reader already implements them). `IndividualReader` switches its glob from `*.tiff` to `*_Fluorescence_*` so it can't pick up output files in an in-place output directory.

### 2.3 `core.py`

**Single sep helper** — used by both in-process and worker paths:

```python
def _run_sep(image: np.ndarray, box_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (foreground, background). Both float32, same shape as input."""
    img = np.ascontiguousarray(image, dtype=np.float32)
    bkg = sep.Background(img, bw=box_size, bh=box_size,
                         fw=_SEP_FILTER_SIZE, fh=_SEP_FILTER_SIZE)
    bg = bkg.back()
    return img - bg, bg
```

**Worker** becomes a thin wrapper around the helper:

```python
def _process_frame_worker(file_path, page_idx, output_dir, box_size, dtype_str, out_name):
    image = (tifffile.imread(file_path) if page_idx is None
             else tifffile.imread(file_path, key=page_idx))
    fg, _ = _run_sep(image, box_size)
    info = np.iinfo(np.dtype(dtype_str))
    out = np.clip(fg, info.min, info.max).astype(np.dtype(dtype_str))
    tifffile.imwrite(str(Path(output_dir) / out_name), out)
    return out_name
```

(Imports stay at module top — fine for `ProcessPoolExecutor` with the default `spawn` start method on macOS, since the worker function is module-level.)

**Module-level constants** with one-line justifications:

```python
_SEP_FILTER_SIZE = 3                    # sep's documented filter window default
_DEFAULT_BOX_SIZE = 50                  # Cephla microscopy default; user-overridable in GUI/API
_MEM_PER_WORKER_MULTIPLIER = 3          # sep peak ~2x input + 1x output buffer
_MEM_HEADROOM_BYTES = 2 * 1024**3       # leave 2 GB for OS + main process
_MEM_USE_FRACTION = 0.75                # don't fully saturate available memory
_PENDING_TASKS_PER_WORKER = 2           # keep workers fed without unbounded queue growth
```

**`_safe_worker_count`** — no try/except (psutil now required, frame_shape always available since acquisitions always have ≥1 file):

```python
def _safe_worker_count(self) -> int:
    shape = self.acq.frame_shape
    dtype = self.acq.frame_dtype
    frame_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    available = psutil.virtual_memory().available
    usable = max(0, available - _MEM_HEADROOM_BYTES) * _MEM_USE_FRACTION
    mem_limited = max(1, int(usable / (frame_bytes * _MEM_PER_WORKER_MULTIPLIER)))
    return min(mem_limited, os.cpu_count() or 1)
```

**`n_frames_per_fov`** uses the new public reader API. The `try/except AttributeError` is gone:

```python
def n_frames_per_fov(self, channel: str) -> int:
    fovs = list(self.acq.iter_fovs())
    return self.acq.n_frames_per_fov(fovs[0], channel) if fovs else 0
```

**`process_all`** consumes `FrameRef` instead of 4-tuples:

```python
for ref in self.acq.iter_frames(channel):
    out_name = (ref.file_path.name if ref.page_idx is None
                else f"{ref.file_path.stem}_page{ref.page_idx:04d}.tiff")
    tasks.append((str(ref.file_path), ref.page_idx, str(self.output_path),
                  self.box_size, dtype_str, out_name))
```

`max_pending = workers * _PENDING_TASKS_PER_WORKER`.

### 2.4 `metrics.py`

- Delete `suggest_box_size` (auto-detect removed).
- Drop `bg_fraction` from the metrics dict (unused by GUI; nothing else consumes it).
- Add a one-line module note that `signal_preservation` is `corrcoef(original, foreground)` — useful as a sanity check, **not** an optimization target.

### 2.5 `gui/app.py`

**Remove:**
- `AutoBoxSizeWorker` class.
- "Auto-detect" button and its handler `_on_auto_box`.
- Lazy import `from bgsub.metrics import suggest_box_size`.

**Hoist imports** to module top:
- `from bgsub.core import BackgroundSubtractor`
- `import cv2` (now a declared dep)
- `import sep` (already a dep — no need to alias as `_sep`)

The macOS Qt-plugin shim stays where it is (must run before `from PyQt5.QtWidgets import ...`).

**Axis-state cleanup** — replace the visibility-as-state hack:

`AxisSlider` gains a public `set_label(text: str)` method. `_t_slider._label.setText("Z")` becomes `self._frame_slider.set_label("Z")`. The GUI keeps the existing two-slider layout (FOV slider + frame slider) — we drop the unused `_z_slider` entirely (it was always hidden in current code; the frame slider was always doing the work). `_get_frame_idx()` becomes a single line: `return self._frame_slider.value()`.

**Movie worker rewrite** — uses public reader API + `process_single`:

```python
class MovieWorker(QThread):
    def run(self):
        acq = self.subtractor.acq
        for ch in acq.metadata.channels:
            refs = list(acq.iter_frames_for_fov(next(acq.iter_fovs()), ch))
            if not refs: continue
            # sample-based normalization
            sample_idxs = list(range(0, len(refs),
                                     max(1, len(refs) // _NORMALIZATION_SAMPLES)))
            raw_hi, fg_hi = self._compute_normalization(refs, sample_idxs)
            # encode side-by-side
            ...
            for ref in refs:
                image = self._read_ref(ref)
                fg, _ = self.subtractor.process_single(image)
                ...
```

The duplicated sep call is gone. `_read_ref(ref)` handles `page_idx is None` vs int.

**Drop-label styles** — three repeated CSS strings become three module-level constants: `_DROP_LABEL_IDLE_STYLE`, `_DROP_LABEL_HOVER_STYLE`, `_DROP_LABEL_LOADED_STYLE`.

**Movie/preview constants** at module top:

```python
_NORMALIZATION_PERCENTILE = 99.0   # robust upper bound, ignores hot pixels
_NORMALIZATION_SAMPLES = 10        # enough frames for stable median estimate
_MOVIE_FPS = 30                    # standard playback rate
_THUMBNAIL_PERCENTILES = (0.5, 99) # preview contrast window
_PREVIEW_DEBOUNCE_MS = 200         # coalesce slider drags
```

### 2.6 Readers

- `IndividualReader._find_files` renamed to `_files_for_fov` (still private). Public wrappers `iter_frames_for_fov` and `n_frames_per_fov` call it. The `frame_shape`/`frame_dtype` glob narrows to `*_Fluorescence_*`.
- `OMETiffReader`: parse channel list **once** in `open_ometiff` (already partially done) and cache `_channel_indices: dict[str, int]` and `_n_channels: int` on the reader. `iter_frames_for_fov` uses the cache instead of re-running the regex per FOV. `_get_channel_index` becomes a dict lookup.
- `CurrentStackReader`: build a per-file index `{(channel, z_level): page_idx}` lazily on first access, cache it. `iter_frames_for_fov`, `get_frame`, `get_stack` all read from the index — no more JSON parsing in tight loops.

### 2.7 Tests (`tests/`)

`pyproject.toml` adds:
```toml
[project.optional-dependencies]
dev = ["ruff>=0.1", "pytest>=7"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`tests/conftest.py` — fixture builders that produce synthetic acquisitions in `tmp_path`:
- `make_individual_fixture(tmp_path, fovs=2, channels=("488","561"), z_planes=3, shape=(64,64))`
- `make_ometiff_fixture(tmp_path, ...)` — writes synthetic OME-XML
- `make_currentstack_fixture(tmp_path, ...)` — multi-page TIFF with JSON page descriptions
- `make_acquisition_parameters_json(tmp_path, ...)` — minimum-viable JSON

Test files:

| File | Coverage |
|---|---|
| `test_readers_individual.py` | `iter_fovs`, `n_frames_per_fov`, `iter_frames_for_fov`, `get_frame`, `frame_shape`, `frame_dtype` |
| `test_readers_ometiff.py` | Same interface; verifies channel cache hit; verifies [Z][C] page-order assumption with both axis orderings in OME-XML |
| `test_readers_currentstack.py` | Same interface; verifies page-index cache |
| `test_detect.py` | All three formats detected from their fixtures; unknown dir raises `ValueError` |
| `test_core.py` | `_run_sep` matches direct `sep.Background` call; `process_single` is its public face; `process_all` writes one output per input frame; output dtype matches `acq.frame_dtype`; output values clipped to dtype range; metrics dict has the documented keys |
| `test_gui_smoke.py` | `import gui.app` and `from gui.app import MainWindow` succeed; if `PyQt5` not installed, test is skipped, not failed |

GUI behavior is **not** tested beyond the import smoke test. The user accepted this trade-off.

### 2.8 Constants migration table

| Was | Becomes | Justification |
|---|---|---|
| `_MEM_MULTIPLIER = 3` | `_MEM_PER_WORKER_MULTIPLIER = 3` | sep peak ≈ 2× input + 1× output buffer |
| inline `2 * 1024**3` | `_MEM_HEADROOM_BYTES` | OS + main proc reservation |
| inline `0.75` | `_MEM_USE_FRACTION` | leave headroom for spikes |
| inline `4 * 1024**3` (fallback) | (deleted; psutil required) | — |
| inline `100 MB` (fallback) | (deleted; frame_shape always available) | — |
| inline `box_size=50` (×2) | `_DEFAULT_BOX_SIZE` (single source) | Cephla default |
| inline `fw=3, fh=3` (×5) | `_SEP_FILTER_SIZE = 3` (single source) | sep documented default |
| inline `workers * 2` | `_PENDING_TASKS_PER_WORKER = 2` | backpressure |
| inline `99` | `_NORMALIZATION_PERCENTILE = 99.0` | robust upper bound |
| inline `n_frames // 10` | `_NORMALIZATION_SAMPLES = 10` | stable median |
| inline movie `fps=30` | `_MOVIE_FPS = 30` | standard playback |
| inline `(0.5, 99)` | `_THUMBNAIL_PERCENTILES = (0.5, 99)` | preview contrast window |
| inline preview `setInterval(200)` | `_PREVIEW_DEBOUNCE_MS = 200` | coalesce slider drags |
| `page_idx = -1` (sentinel) | `page_idx: int \| None` (None = single-page) | typed |

---

## 3. Out of scope

- Algorithm changes (sep stays).
- New formats / new readers.
- Replacing PyQt5 with another framework.
- GUI integration tests beyond import smoke.
- Performance optimization beyond removing redundant XML/JSON re-parsing in readers.
- Packaging beyond declaring deps and ensuring README exists (no PyPI release work).

## 4. Acceptance

- `pip install -e .[dev]` from a fresh venv succeeds.
- `pytest` runs all tests green.
- `ruff check .` is clean.
- `python -m bgsub` launches the GUI; preview, full-run, and movie generation work end-to-end on at least the Individual format.
- No `_find_files` access from outside `IndividualReader`.
- `grep -rn "fw=" src/bgsub gui` returns exactly one match (the `_run_sep` helper).
- `suggest_box_size` and the Auto-detect button are gone.
- `src/bgsub/io/` is gone. `pyyaml` is gone from deps.
- `README.md` exists.
