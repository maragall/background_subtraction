# bgsub Production-Readiness Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take `bgsub` from "algorithm works" to production-ready: kill dead code, fix the broken auto-detect heuristic, declare missing deps, define a public reader API, deduplicate the sep pipeline, name and justify every magic constant, and add a synthetic-fixture pytest suite.

**Architecture:** The sep-based subtraction algorithm and PyQt5 GUI flow are unchanged. We add a public `FrameRef` / `iter_frames_for_fov` / `n_frames_per_fov` API on `AcquisitionReader`, route `core.py` and `gui.app.MovieWorker` through it, collapse 5 duplicated `sep.Background(...)` calls into a single `_run_sep` helper, and replace the broken `suggest_box_size` (auto-detect) with no auto-detect at all.

**Tech Stack:** Python 3.9+, numpy, sep, tifffile, PyQt5, opencv-python, psutil, pytest. Build: hatchling. Lint: ruff.

**Spec:** `docs/superpowers/specs/2026-05-02-bgsub-production-cleanup-design.md`

---

## File Structure

**New files:**
- `README.md` — minimal install + usage docs
- `tests/__init__.py`
- `tests/conftest.py` — synthetic fixture builders for all 3 acquisition formats
- `tests/test_readers_individual.py`
- `tests/test_readers_ometiff.py`
- `tests/test_readers_currentstack.py`
- `tests/test_detect.py`
- `tests/test_core.py`
- `tests/test_gui_smoke.py`

**Modified files:**
- `pyproject.toml` — add `opencv-python`, `psutil`; add `pytest` to dev; remove `pyyaml`; remove `"scripts"` from wheel packages; add `[tool.pytest.ini_options]`
- `src/bgsub/readers/base.py` — add `FrameRef`, add `iter_frames_for_fov`/`n_frames_per_fov` (default `NotImplementedError`, then made abstract once all readers implement them)
- `src/bgsub/readers/individual.py` — public `iter_frames_for_fov`/`n_frames_per_fov`; narrow glob
- `src/bgsub/readers/ometiff.py` — channel-index cache; public `iter_frames_for_fov`/`n_frames_per_fov`
- `src/bgsub/readers/currentstack.py` — page-index cache; public `iter_frames_for_fov`/`n_frames_per_fov`
- `src/bgsub/core.py` — single `_run_sep` helper; named constants; consume `FrameRef`; drop unused `import shutil`
- `src/bgsub/metrics.py` — delete `suggest_box_size`; drop `bg_fraction`
- `gui/app.py` — hoist imports; drop auto-detect; drop unused `_z_slider`; clean axis label API; rewrite `MovieWorker` to use public API + `process_single`; named constants; drop-label CSS constants

**Deleted files/dirs:**
- `src/bgsub/io/` (entire directory)
- `scripts/explore_background2d.py`, `scripts/explore_rolling_ball.py`, `scripts/quality_metrics.py`, `scripts/__init__.py`

---

## Task 1: Set up test infrastructure and update `pyproject.toml`

**Files:**
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/pyproject.toml`
- Create: `/Users/julioamaragall/Downloads/background_subtraction/README.md`
- Create: `/Users/julioamaragall/Downloads/background_subtraction/tests/__init__.py` (empty)

- [ ] **Step 1: Update `pyproject.toml`**

Replace the `dependencies`, `optional-dependencies`, `scripts`, and `[tool.hatch.build.targets.wheel]` blocks. Final state:

```toml
dependencies = [
    "numpy>=1.21",
    "tifffile>=2023.0",
    "sep>=1.2",
    "opencv-python>=4.5",
    "psutil>=5.9",
]

[project.optional-dependencies]
gui = ["PyQt5>=5.15"]
dev = ["ruff>=0.1", "pytest>=7"]

[project.scripts]
bgsub-gui = "gui.app:main"

[project.urls]
Repository = "https://github.com/maragall/background_subtraction"

[tool.hatch.build.targets.wheel]
packages = ["src/bgsub", "gui"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
ignore = ["E501"]
```

(`pyyaml` removed; `opencv-python` and `psutil` added; `pytest` added to dev; `"scripts"` dropped from wheel packages; `[tool.pytest.ini_options]` added.)

- [ ] **Step 2: Create minimal `README.md`**

```markdown
# bgsub

Minimal background subtraction for fluorescence microscopy acquisitions, built on
[sep](https://sep.readthedocs.io/).

## Install

```bash
pip install -e .[gui]
```

## Run

GUI:
```bash
bgsub-gui
# or: python -m bgsub
```

Programmatic:
```python
from bgsub import BackgroundSubtractor

sub = BackgroundSubtractor("/path/to/acquisition", box_size=50)
sub.process_all()
```

## Supported acquisition formats

- **Individual TIFFs**: `*_Fluorescence_<wavelength>_nm_Ex*.tiff`
- **OME-TIFF**: `ome_tiff/*.ome.tiff` + `acquisition_parameters.json`
- **CurrentStack**: `*_stack.tiff` multi-page (Squid `MULTI_PAGE_TIFF` output)

## Development

```bash
pip install -e .[dev,gui]
pytest
ruff check .
```

## License

BSD-3-Clause
```

- [ ] **Step 3: Create empty `tests/__init__.py`**

```bash
touch /Users/julioamaragall/Downloads/background_subtraction/tests/__init__.py
```

- [ ] **Step 4: Verify install + collect**

```bash
cd /Users/julioamaragall/Downloads/background_subtraction
pip install -e .[dev,gui]
pytest --collect-only
```

Expected: install succeeds; `pytest --collect-only` reports 0 tests collected (no errors).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md tests/__init__.py
git commit -m "chore: declare opencv/psutil deps, drop pyyaml, add README and pytest config"
```

---

## Task 2: Synthetic-fixture builders in `tests/conftest.py`

**Files:**
- Create: `/Users/julioamaragall/Downloads/background_subtraction/tests/conftest.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
"""Synthetic fixture builders. No tests yet — these are pytest fixtures
that produce real on-disk acquisitions in tmp_path that the readers can open.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile


def _write_acquisition_parameters_json(folder: Path, nz: int = 1, nt: int = 1) -> None:
    """Minimum-viable acquisition_parameters.json that satisfies Metadata.from_acquisition_json."""
    data = {
        "objective": {"magnification": 20, "NA": 0.45},
        "sensor_pixel_size_um": 6.5,
        "dz(um)": 1.0,
        "Nz": nz,
        "Nt": nt,
    }
    (folder / "acquisition_parameters.json").write_text(json.dumps(data))


def _make_image(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.integers(100, 1000, size=shape, dtype=np.int32)).astype(np.uint16)


@pytest.fixture
def make_individual_fixture(tmp_path):
    """Build an Individual-format acquisition under tmp_path/individual.

    Layout:
        tmp_path/individual/
            acquisition_parameters.json
            R0/
                R0_<fov>_<z>_Fluorescence_<channel>_nm_Ex_-_single_band_0000.tiff
    """
    def _make(fovs=2, channels=("488", "561"), z_planes=3, shape=(64, 64)):
        root = tmp_path / "individual"
        sub = root / "R0"
        sub.mkdir(parents=True)
        _write_acquisition_parameters_json(root, nz=z_planes)
        for fov_idx in range(fovs):
            for z in range(z_planes):
                for ch in channels:
                    name = (
                        f"R0_{fov_idx}_{z}_Fluorescence_{ch}_nm_Ex"
                        f"_-_single_band_0000.tiff"
                    )
                    img = _make_image(shape, seed=fov_idx * 100 + z * 10 + int(ch))
                    tifffile.imwrite(str(sub / name), img)
        return root
    return _make


@pytest.fixture
def make_ometiff_fixture(tmp_path):
    """Build an OME-TIFF acquisition under tmp_path/ometiff.

    Each FOV is one .ome.tiff with Z*C pages. OME-XML lists each
    'Fluorescence <wavelength> nm Ex' channel exactly once.
    """
    def _make(fovs=2, channels=("488", "561"), z_planes=3, shape=(64, 64)):
        root = tmp_path / "ometiff"
        ome_dir = root / "ome_tiff"
        ome_dir.mkdir(parents=True)
        _write_acquisition_parameters_json(root, nz=z_planes)

        ch_xml = "".join(
            f'<Channel ID="Channel:0:{i}" Name="Fluorescence {ch} nm Ex" '
            f'SamplesPerPixel="1"/>'
            for i, ch in enumerate(channels)
        )
        ome_xml_template = (
            '<?xml version="1.0"?>'
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            f'<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" '
            f'Type="uint16" SizeX="{shape[1]}" SizeY="{shape[0]}" '
            f'SizeZ="{z_planes}" SizeC="{len(channels)}" SizeT="1">'
            f'{ch_xml}</Pixels></Image></OME>'
        )

        for fov_idx in range(fovs):
            # Page order matches DimensionOrder XYCZT → C varies fastest, then Z
            pages = []
            for z in range(z_planes):
                for c, ch in enumerate(channels):
                    pages.append(_make_image(shape, seed=fov_idx * 100 + z * 10 + c))
            arr = np.stack(pages, axis=0)
            path = ome_dir / f"R0_{fov_idx}.ome.tiff"
            tifffile.imwrite(str(path), arr, description=ome_xml_template)
        return root
    return _make


@pytest.fixture
def make_currentstack_fixture(tmp_path):
    """Build a CurrentStack acquisition under tmp_path/currentstack.

    Each FOV is a multi-page TIFF where each page has a JSON description
    with 'channel' and 'z_level' keys.
    """
    def _make(fovs=2, channels=("488 nm Ex", "561 nm Ex"), z_planes=3, shape=(64, 64)):
        root = tmp_path / "currentstack"
        plane_dir = root / "0"
        plane_dir.mkdir(parents=True)
        # uses "acquisition parameters.json" (with space) per the reader
        data = {
            "objective": {"magnification": 20, "NA": 0.45},
            "sensor_pixel_size_um": 6.5,
            "dz(um)": 1.0,
            "Nz": z_planes,
            "Nt": 1,
        }
        (root / "acquisition parameters.json").write_text(json.dumps(data))

        for fov_idx in range(fovs):
            path = plane_dir / f"R0_{fov_idx}_stack.tiff"
            with tifffile.TiffWriter(str(path)) as tw:
                for z in range(z_planes):
                    for ch in channels:
                        img = _make_image(shape, seed=fov_idx * 100 + z * 10 + len(ch))
                        desc = json.dumps({"channel": ch, "z_level": z})
                        tw.write(img, description=desc)
        return root
    return _make
```

- [ ] **Step 2: Sanity-check fixtures**

```bash
cd /Users/julioamaragall/Downloads/background_subtraction
pytest --collect-only
```

Expected: 0 tests collected, no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add synthetic acquisition fixture builders"
```

---

## Task 3: Add `FrameRef` dataclass to base reader

**Files:**
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/src/bgsub/readers/base.py`

- [ ] **Step 1: Add `FrameRef` and stub the new methods**

In `src/bgsub/readers/base.py`, insert `FrameRef` after the `FOV` dataclass and add two new methods to `AcquisitionReader` that default to `NotImplementedError`. We do **not** make them `@abstractmethod` yet — readers will be migrated in Tasks 4-6 first, then we tighten in Task 7.

Add this after the `FOV` dataclass (around line 73 of the current file):

```python
@dataclass(frozen=True)
class FrameRef:
    """Reference to a single 2D frame inside an acquisition.

    page_idx is None for single-page TIFFs (read whole file) and an integer
    for multi-page TIFFs (read that page).
    """
    fov: "FOV"
    frame_idx: int
    file_path: Path
    page_idx: int | None
```

In the `AcquisitionReader` class, add after the existing `iter_frames` method:

```python
    def iter_frames_for_fov(self, fov: FOV, channel: str) -> Iterator["FrameRef"]:
        """Yield FrameRef for every 2D frame in this FOV+channel, in frame_idx order.

        Subclasses must override.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement iter_frames_for_fov"
        )

    def n_frames_per_fov(self, fov: FOV, channel: str) -> int:
        """Number of 2D frames for one FOV+channel.

        Subclasses must override.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement n_frames_per_fov"
        )
```

Also: `Path` is already imported at the top of the file. The `FrameRef` forward reference (`"FOV"`) is needed because `FOV` is defined later... actually `FOV` comes first; just use `FOV` directly (no quotes).

After verifying `FOV` is defined before where you insert `FrameRef`, simplify to:

```python
@dataclass(frozen=True)
class FrameRef:
    fov: FOV
    frame_idx: int
    file_path: Path
    page_idx: int | None
```

(`FOV` is defined at line 65 in the current file; insert `FrameRef` immediately after it, before `class AcquisitionReader`.)

- [ ] **Step 2: Update `__init__.py` to re-export `FrameRef`**

In `src/bgsub/readers/__init__.py`:

```python
"""Acquisition format readers (adapted from petakit/Deconvolution)."""
from .base import Metadata, FOV, FrameRef, AcquisitionReader
from .detect import open_acquisition, detect_format

__all__ = [
    "Metadata",
    "FOV",
    "FrameRef",
    "AcquisitionReader",
    "open_acquisition",
    "detect_format",
]
```

- [ ] **Step 3: Verify imports**

```bash
cd /Users/julioamaragall/Downloads/background_subtraction
python -c "from bgsub.readers import FrameRef, AcquisitionReader; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/bgsub/readers/base.py src/bgsub/readers/__init__.py
git commit -m "refactor(readers): add FrameRef dataclass and stub public per-FOV API"
```

---

## Task 4: Migrate `IndividualReader` to public API + tests

**Files:**
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/src/bgsub/readers/individual.py`
- Create: `/Users/julioamaragall/Downloads/background_subtraction/tests/test_readers_individual.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for IndividualReader. Uses synthetic fixtures from conftest.py."""
import numpy as np
import pytest

from bgsub.readers import FrameRef, open_acquisition
from bgsub.readers.individual import IndividualReader


def test_open_individual_returns_correct_reader(make_individual_fixture):
    root = make_individual_fixture()
    reader = open_acquisition(root)
    assert isinstance(reader, IndividualReader)
    assert reader.format_name == "individual"


def test_iter_fovs_yields_unique_fovs(make_individual_fixture):
    root = make_individual_fixture(fovs=3, channels=("488",), z_planes=1)
    reader = open_acquisition(root)
    fovs = list(reader.iter_fovs())
    assert len(fovs) == 3
    assert {f.index for f in fovs} == {0, 1, 2}


def test_n_frames_per_fov_matches_z_planes(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488", "561"), z_planes=4)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    assert reader.n_frames_per_fov(fov, "488") == 4
    assert reader.n_frames_per_fov(fov, "561") == 4


def test_iter_frames_for_fov_yields_frame_refs_in_order(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488",), z_planes=3)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    refs = list(reader.iter_frames_for_fov(fov, "488"))
    assert len(refs) == 3
    assert all(isinstance(r, FrameRef) for r in refs)
    assert all(r.page_idx is None for r in refs)  # single-page TIFFs
    assert [r.frame_idx for r in refs] == sorted(r.frame_idx for r in refs)


def test_iter_frames_aggregates_across_fovs(make_individual_fixture):
    root = make_individual_fixture(fovs=2, channels=("488",), z_planes=3)
    reader = open_acquisition(root)
    refs = list(reader.iter_frames("488"))
    assert len(refs) == 6  # 2 fovs * 3 z


def test_get_frame_returns_2d_array(make_individual_fixture):
    root = make_individual_fixture(shape=(48, 32))
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    frame = reader.get_frame(fov, "488", 0)
    assert frame.shape == (48, 32)
    assert frame.dtype == np.float32


def test_frame_shape_and_dtype(make_individual_fixture):
    root = make_individual_fixture(shape=(48, 32))
    reader = open_acquisition(root)
    assert reader.frame_shape == (48, 32)
    assert reader.frame_dtype == np.uint16


def test_glob_ignores_unrelated_tiffs(make_individual_fixture, tmp_path):
    """frame_shape/frame_dtype must not pick up bg-subtracted output files
    that happen to live in the same directory."""
    root = make_individual_fixture(shape=(48, 32))
    reader = open_acquisition(root)
    # Drop a stray TIFF in the tiff_dir that doesn't match the Fluorescence pattern
    import tifffile
    stray = reader._tiff_dir / "stray_output.tiff"
    tifffile.imwrite(str(stray), np.zeros((1, 1), dtype=np.uint8))
    # Should still report the real frame's shape
    assert reader.frame_shape == (48, 32)
    assert reader.frame_dtype == np.uint16
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_readers_individual.py -v
```

Expected: most tests fail. Current code uses `iter_frames` returning 4-tuples, no `iter_frames_for_fov`, no `n_frames_per_fov`. `test_glob_ignores_unrelated_tiffs` will fail because current code globs `*.tiff`.

- [ ] **Step 3: Update `IndividualReader`**

In `src/bgsub/readers/individual.py`, make these changes:

1. Add `FrameRef` to the `from .base import` line:
```python
from .base import AcquisitionReader, Metadata, FOV, FrameRef
```

2. Replace the existing `iter_frames` method with a public per-FOV API:

```python
    def iter_frames_for_fov(self, fov: FOV, channel: str):
        """Yield FrameRef for every 2D frame in this FOV+channel."""
        for z_idx, path in self._find_files(fov, channel):
            yield FrameRef(fov=fov, frame_idx=z_idx, file_path=path, page_idx=None)

    def n_frames_per_fov(self, fov: FOV, channel: str) -> int:
        return len(self._find_files(fov, channel))

    def iter_frames(self, channel: str):
        for fov in self.iter_fovs():
            yield from self.iter_frames_for_fov(fov, channel)
```

3. Narrow the glob in `frame_shape` and `frame_dtype`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_readers_individual.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bgsub/readers/individual.py tests/test_readers_individual.py
git commit -m "refactor(readers): public FrameRef API on IndividualReader + tests"
```

---

## Task 5: Migrate `OMETiffReader` to public API + tests + caching

**Files:**
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/src/bgsub/readers/ometiff.py`
- Create: `/Users/julioamaragall/Downloads/background_subtraction/tests/test_readers_ometiff.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for OMETiffReader."""
import numpy as np

from bgsub.readers import FrameRef, open_acquisition
from bgsub.readers.ometiff import OMETiffReader


def test_open_ometiff_returns_correct_reader(make_ometiff_fixture):
    root = make_ometiff_fixture()
    reader = open_acquisition(root)
    assert isinstance(reader, OMETiffReader)
    assert reader.format_name == "ometiff"


def test_channels_parsed_from_ome_xml(make_ometiff_fixture):
    root = make_ometiff_fixture(channels=("405", "488", "561"))
    reader = open_acquisition(root)
    assert reader.metadata.channels == ["405", "488", "561"]


def test_iter_fovs_yields_unique_fovs(make_ometiff_fixture):
    root = make_ometiff_fixture(fovs=3)
    reader = open_acquisition(root)
    fovs = list(reader.iter_fovs())
    assert len(fovs) == 3


def test_n_frames_per_fov_matches_z(make_ometiff_fixture):
    root = make_ometiff_fixture(fovs=1, z_planes=5)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    assert reader.n_frames_per_fov(fov, "488") == 5


def test_iter_frames_for_fov_yields_frame_refs(make_ometiff_fixture):
    root = make_ometiff_fixture(fovs=1, channels=("488", "561"), z_planes=3)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    refs = list(reader.iter_frames_for_fov(fov, "488"))
    assert len(refs) == 3
    assert all(isinstance(r, FrameRef) for r in refs)
    assert all(r.page_idx is not None for r in refs)  # multi-page


def test_get_frame_returns_2d(make_ometiff_fixture):
    root = make_ometiff_fixture(shape=(48, 32))
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    frame = reader.get_frame(fov, "488", 0)
    assert frame.shape == (48, 32)
    assert frame.dtype == np.float32


def test_channel_index_cache(make_ometiff_fixture):
    """After first call, channel indices should be cached on the reader."""
    root = make_ometiff_fixture(channels=("488", "561"))
    reader = open_acquisition(root)
    # Trigger caching
    list(reader.iter_frames("488"))
    assert hasattr(reader, "_channel_indices")
    assert reader._channel_indices == {"488": 0, "561": 1}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_readers_ometiff.py -v
```

Expected: failures on `iter_frames_for_fov`, `n_frames_per_fov`, FrameRef shape, cache attribute.

- [ ] **Step 3: Rewrite `OMETiffReader`**

Replace `src/bgsub/readers/ometiff.py` with this version (caches channel indices on the reader; new public API; existing OME-XML parsing logic preserved for `get_stack`/`get_frame`):

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_readers_ometiff.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bgsub/readers/ometiff.py tests/test_readers_ometiff.py
git commit -m "refactor(readers): public FrameRef API on OMETiffReader with channel cache"
```

---

## Task 6: Migrate `CurrentStackReader` to public API + tests + caching

**Files:**
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/src/bgsub/readers/currentstack.py`
- Create: `/Users/julioamaragall/Downloads/background_subtraction/tests/test_readers_currentstack.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for CurrentStackReader."""
import numpy as np

from bgsub.readers import FrameRef, open_acquisition
from bgsub.readers.currentstack import CurrentStackReader


def test_open_currentstack_returns_correct_reader(make_currentstack_fixture):
    root = make_currentstack_fixture()
    reader = open_acquisition(root)
    assert isinstance(reader, CurrentStackReader)
    assert reader.format_name == "currentstack"


def test_channels_parsed(make_currentstack_fixture):
    root = make_currentstack_fixture(channels=("488 nm Ex", "561 nm Ex"))
    reader = open_acquisition(root)
    assert reader.metadata.channels == ["488", "561"]


def test_iter_fovs_yields_unique(make_currentstack_fixture):
    root = make_currentstack_fixture(fovs=3)
    reader = open_acquisition(root)
    fovs = list(reader.iter_fovs())
    assert len(fovs) == 3


def test_n_frames_per_fov_matches_z(make_currentstack_fixture):
    root = make_currentstack_fixture(z_planes=4)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    assert reader.n_frames_per_fov(fov, "488") == 4


def test_iter_frames_for_fov_yields_frame_refs(make_currentstack_fixture):
    root = make_currentstack_fixture(z_planes=3)
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    refs = list(reader.iter_frames_for_fov(fov, "488"))
    assert len(refs) == 3
    assert all(isinstance(r, FrameRef) for r in refs)
    assert all(r.page_idx is not None for r in refs)


def test_get_frame_returns_2d(make_currentstack_fixture):
    root = make_currentstack_fixture(shape=(48, 32))
    reader = open_acquisition(root)
    fov = next(reader.iter_fovs())
    frame = reader.get_frame(fov, "488", 0)
    assert frame.shape == (48, 32)
    assert frame.dtype == np.float32
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_readers_currentstack.py -v
```

Expected: failures.

- [ ] **Step 3: Rewrite `CurrentStackReader`**

Replace `src/bgsub/readers/currentstack.py` with this version (lazy per-file index of `(channel, z_level) → page_idx`):

```python
"""Reader for *_stack.tiff multi-page format (Squid MULTI_PAGE_TIFF output).

From petakit/Deconvolution.
"""
import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from .base import AcquisitionReader, Metadata, FOV, FrameRef

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

    def iter_frames(self, channel: str):
        for fov in self.iter_fovs():
            yield from self.iter_frames_for_fov(fov, channel)

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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_readers_currentstack.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bgsub/readers/currentstack.py tests/test_readers_currentstack.py
git commit -m "refactor(readers): public FrameRef API on CurrentStackReader with page-index cache"
```

---

## Task 7: Test format detection

**Files:**
- Create: `/Users/julioamaragall/Downloads/background_subtraction/tests/test_detect.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for format detection."""
import pytest

from bgsub.readers import detect_format, open_acquisition


def test_detect_individual(make_individual_fixture):
    root = make_individual_fixture()
    assert detect_format(root) == "individual"


def test_detect_ometiff(make_ometiff_fixture):
    root = make_ometiff_fixture()
    assert detect_format(root) == "ometiff"


def test_detect_currentstack(make_currentstack_fixture):
    root = make_currentstack_fixture()
    assert detect_format(root) == "currentstack"


def test_detect_unknown_returns_none(tmp_path):
    (tmp_path / "irrelevant.txt").write_text("hi")
    assert detect_format(tmp_path) is None


def test_open_unknown_raises(tmp_path):
    (tmp_path / "irrelevant.txt").write_text("hi")
    with pytest.raises(ValueError, match="Unknown acquisition format"):
        open_acquisition(tmp_path)


def test_open_missing_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        open_acquisition(missing)
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_detect.py -v
```

Expected: all 6 pass (no source changes needed).

- [ ] **Step 3: Commit**

```bash
git add tests/test_detect.py
git commit -m "test: format detection for all 3 readers"
```

---

## Task 8: Refactor `core.py` — single sep helper, named constants, public reader API

**Files:**
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/src/bgsub/core.py`
- Create: `/Users/julioamaragall/Downloads/background_subtraction/tests/test_core.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for BackgroundSubtractor core."""
import numpy as np
import sep
import tifffile

from bgsub import BackgroundSubtractor
from bgsub.core import _run_sep


def test_run_sep_matches_direct_call():
    rng = np.random.default_rng(0)
    img = rng.integers(100, 1000, size=(64, 64)).astype(np.uint16)
    fg, bg = _run_sep(img, box_size=16)
    # Reference
    ref_input = np.ascontiguousarray(img, dtype=np.float32)
    ref_bkg = sep.Background(ref_input, bw=16, bh=16, fw=3, fh=3)
    ref_bg = ref_bkg.back()
    np.testing.assert_array_equal(bg, ref_bg)
    np.testing.assert_array_equal(fg, ref_input - ref_bg)


def test_process_single_returns_correct_shapes(make_individual_fixture):
    root = make_individual_fixture(shape=(64, 64))
    sub = BackgroundSubtractor(root, box_size=16)
    img = np.random.default_rng(1).integers(100, 1000, size=(64, 64)).astype(np.uint16)
    fg, bg = sub.process_single(img)
    assert fg.shape == (64, 64)
    assert bg.shape == (64, 64)
    assert fg.dtype == np.float32


def test_process_all_writes_one_output_per_input_frame(make_individual_fixture):
    root = make_individual_fixture(fovs=2, channels=("488",), z_planes=3)
    sub = BackgroundSubtractor(root, box_size=16, max_workers=2)
    sub.process_all(channels=["488"])
    outputs = list(sub.output_path.glob("*.tiff"))
    assert len(outputs) == 6  # 2 fov * 3 z


def test_output_dtype_matches_input(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488",), z_planes=1)
    sub = BackgroundSubtractor(root, box_size=16, max_workers=1)
    sub.process_all(channels=["488"])
    out = next(sub.output_path.glob("*.tiff"))
    written = tifffile.imread(str(out))
    assert written.dtype == np.uint16


def test_output_clipped_to_dtype_range(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488",), z_planes=1)
    sub = BackgroundSubtractor(root, box_size=16, max_workers=1)
    sub.process_all(channels=["488"])
    out = next(sub.output_path.glob("*.tiff"))
    written = tifffile.imread(str(out))
    info = np.iinfo(np.uint16)
    assert written.min() >= info.min
    assert written.max() <= info.max


def test_progress_callback_invoked(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488",), z_planes=2)
    sub = BackgroundSubtractor(root, box_size=16, max_workers=1)
    calls = []
    sub.process_all(
        channels=["488"],
        progress_callback=lambda c, t, m: calls.append((c, t, m)),
    )
    assert len(calls) == 2
    assert calls[-1][0] == 2
    assert calls[-1][1] == 2


def test_metrics_dict_has_expected_keys(make_individual_fixture):
    root = make_individual_fixture(fovs=1, channels=("488",), z_planes=1)
    sub = BackgroundSubtractor(root, box_size=16)
    _orig, _fg, _bg, metrics = sub.process_frame("488", 0)
    assert set(metrics.keys()) == {
        "bg_uniformity",
        "signal_preservation",
        "snr_improvement",
        "negative_pixel_pct",
    }
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_core.py -v
```

Expected: failures (`_run_sep` does not exist; `bg_fraction` still in metrics).

- [ ] **Step 3: Replace `src/bgsub/core.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_core.py -v
```

Expected: 6 of 7 tests pass. The `test_metrics_dict_has_expected_keys` test will still fail because `metrics.py` still emits `bg_fraction` — that's fixed in Task 9.

- [ ] **Step 5: Commit**

```bash
git add src/bgsub/core.py tests/test_core.py
git commit -m "refactor(core): single _run_sep helper, named constants, public reader API"
```

---

## Task 9: Clean up `metrics.py` — drop `suggest_box_size` and `bg_fraction`

**Files:**
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/src/bgsub/metrics.py`
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/src/bgsub/__init__.py`

- [ ] **Step 1: Replace `src/bgsub/metrics.py`**

```python
"""Quality metrics for background subtraction evaluation.

Note: `signal_preservation` is `corrcoef(original, foreground)`. It is useful
as a sanity number — a correlation near zero means the subtraction destroyed
all signal — but it is NOT an optimization target. The correlation is highest
when foreground ≈ original, i.e., when no background was subtracted, so
maximizing it would prefer the largest box sizes that change nothing.
"""

import numpy as np


def compute_metrics(
    original: np.ndarray, foreground: np.ndarray, background: np.ndarray
) -> dict:
    """Compute quality metrics for a background subtraction result."""
    bg_mean = background.mean()
    bg_uniformity = background.std() / bg_mean if bg_mean > 0 else float("inf")

    corr = np.corrcoef(original.ravel(), foreground.ravel())[0, 1]

    fg_clipped = np.clip(foreground, 0, None)
    orig_snr = original.mean() / original.std() if original.std() > 0 else 0
    fg_snr = fg_clipped.mean() / fg_clipped.std() if fg_clipped.std() > 0 else 0
    snr_ratio = fg_snr / orig_snr if orig_snr > 0 else 0

    neg_pct = 100.0 * np.sum(foreground < 0) / foreground.size

    return {
        "bg_uniformity": bg_uniformity,
        "signal_preservation": corr,
        "snr_improvement": snr_ratio,
        "negative_pixel_pct": neg_pct,
    }
```

- [ ] **Step 2: Update `src/bgsub/__init__.py`**

```python
"""bgsub — Minimal background subtraction for fluorescence microscopy."""

__version__ = "0.1.0"

from .core import BackgroundSubtractor
from .metrics import compute_metrics
```

(`suggest_box_size` removed.)

- [ ] **Step 3: Run all tests so far**

```bash
pytest -v
```

Expected: all reader, detect, and core tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/bgsub/metrics.py src/bgsub/__init__.py
git commit -m "refactor(metrics): drop suggest_box_size and bg_fraction"
```

---

## Task 10: Make new reader methods abstract on base

**Files:**
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/src/bgsub/readers/base.py`

- [ ] **Step 1: Tighten the base class**

In `src/bgsub/readers/base.py`, change `iter_frames_for_fov` and `n_frames_per_fov` from defaulting to `NotImplementedError` to being `@abstractmethod`. Also turn `frame_shape`/`frame_dtype` `NotImplementedError` properties into abstract properties for consistency. Update the `iter_frames` default to use `iter_frames_for_fov`.

Final state of the affected portion:

```python
class AcquisitionReader(ABC):
    """Abstract base for acquisition format readers."""

    def __init__(self, root: Path, metadata: Metadata):
        self.root = root
        self.metadata = metadata

    @property
    @abstractmethod
    def format_name(self) -> str: ...

    @abstractmethod
    def iter_fovs(self) -> Iterator[FOV]: ...

    @abstractmethod
    def get_stack(self, fov: FOV, channel: str) -> np.ndarray: ...

    @abstractmethod
    def iter_frames_for_fov(self, fov: FOV, channel: str) -> Iterator[FrameRef]: ...

    @abstractmethod
    def n_frames_per_fov(self, fov: FOV, channel: str) -> int: ...

    @property
    @abstractmethod
    def frame_shape(self) -> tuple: ...

    @property
    @abstractmethod
    def frame_dtype(self): ...

    def get_frame(self, fov: FOV, channel: str, z_idx: int) -> np.ndarray:
        stack = self.get_stack(fov, channel)
        if stack.ndim == 3:
            return stack[min(z_idx, stack.shape[0] - 1)]
        return stack

    def iter_frames(self, channel: str) -> Iterator[FrameRef]:
        for fov in self.iter_fovs():
            yield from self.iter_frames_for_fov(fov, channel)

    def get_all_channels(self, fov: FOV) -> dict[str, np.ndarray]:
        return {ch: self.get_stack(fov, ch) for ch in self.metadata.channels}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"format='{self.format_name}', "
            f"fovs={sum(1 for _ in self.iter_fovs())}, "
            f"channels={self.metadata.channels})"
        )
```

- [ ] **Step 2: Remove now-redundant `iter_frames` overrides**

Each of the three readers had its own `iter_frames(channel)` that just delegates to `iter_frames_for_fov`. Remove those overrides:

- `src/bgsub/readers/individual.py`: delete the `def iter_frames(self, channel):` block (the one we added in Task 4 step 3).
- `src/bgsub/readers/ometiff.py`: delete the `def iter_frames(self, channel):` block.
- `src/bgsub/readers/currentstack.py`: delete the `def iter_frames(self, channel):` block.

The base-class default now provides this. (Each reader's specialized `iter_frames_for_fov` is what does the work.)

- [ ] **Step 3: Run all tests**

```bash
pytest -v
```

Expected: all tests still pass — instantiation must still work, and `iter_frames` calls still aggregate across FOVs via the base default.

- [ ] **Step 4: Commit**

```bash
git add src/bgsub/readers/base.py src/bgsub/readers/individual.py src/bgsub/readers/ometiff.py src/bgsub/readers/currentstack.py
git commit -m "refactor(readers): make iter_frames_for_fov/n_frames_per_fov abstract; default iter_frames"
```

---

## Task 11: GUI cleanup — drop auto-detect, clean axis API, hoist imports, add named constants

**Files:**
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/gui/app.py`

This is the biggest mechanical change. Apply edits in order.

- [ ] **Step 1: Hoist top-of-module imports**

Replace lines 11-18 (the existing `import numpy as np` block) with:

```python
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
```

(The macOS Qt-plugin shim above the imports stays where it is.)

- [ ] **Step 2: Add module-level constants and drop-label CSS**

Replace `PREVIEW_W = 400` (line 85) with:

```python
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
```

- [ ] **Step 3: Update `_make_thumbnail` to use named percentile constants**

Replace the body of `_make_thumbnail`:

```python
def _make_thumbnail(arr, max_w=PREVIEW_W):
    """Downsample + normalize to uint8. Runs in worker thread."""
    h, w = arr.shape
    scale = min(1.0, max_w / w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(arr.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    p_lo, p_hi = np.percentile(small, _THUMBNAIL_PERCENTILES)
    out = np.clip((small - p_lo) / (p_hi - p_lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)
```

(Drops the in-function `import cv2`.)

- [ ] **Step 4: Delete `AutoBoxSizeWorker` class entirely**

Remove the class block at lines 251-269 in the current file (the `class AutoBoxSizeWorker(QThread):` and its body).

- [ ] **Step 5: Add `set_label` to `AxisSlider`**

Inside `class AxisSlider(QWidget)`, add a method just before `value(self)`:

```python
    def set_label(self, text: str):
        self._label.setText(text)
```

- [ ] **Step 6: Drop `_z_slider`, simplify axis state**

Inside `MainWindow`:

(a) In `_setup_ui`, delete the block creating `self._z_slider` (the three lines starting `self._z_slider = AxisSlider("Z")` and adding it to the layout). Rename `self._t_slider` to `self._frame_slider` everywhere in `_setup_ui` (search-and-replace within the method).

Block to delete:
```python
        self._z_slider = AxisSlider("Z")
        self._z_slider.valueChanged.connect(self._schedule_preview)
        slider_layout.addWidget(self._z_slider)
```

Rename within `_setup_ui`:
```python
        self._t_slider = AxisSlider("Frame")
        self._t_slider.valueChanged.connect(self._schedule_preview)
        slider_layout.addWidget(self._t_slider)
```
becomes:
```python
        self._frame_slider = AxisSlider("Frame")
        self._frame_slider.valueChanged.connect(self._schedule_preview)
        slider_layout.addWidget(self._frame_slider)
```

(b) In `_load_acquisition`, replace the slider-setup block (lines ~649-661 of the current file) with:

```python
        self._fov_slider.setup(max(0, n_fovs - 1), start=0)

        if has_json and meta.nz > 1:
            self._frame_slider.set_label("Z")
        elif has_json and meta.nt > 1:
            self._frame_slider.set_label("Time")
        else:
            self._frame_slider.set_label("FOV")
        self._frame_slider.setup(max(0, n_frames - 1), start=n_frames // 2)
```

(c) Replace `_get_frame_idx`:

```python
    def _get_frame_idx(self):
        return self._frame_slider.value()
```

(d) In `_set_running`, replace `self._z_slider.setEnabled(...)` with nothing (delete the line) and rename `self._t_slider` to `self._frame_slider`. Final method body:

```python
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
```

(`self.auto_btn` reference is also removed because the button is gone — see step 7.)

- [ ] **Step 7: Drop the Auto-detect button**

In `_setup_ui`, delete the block (around lines 469-476):

```python
        self.auto_btn = QPushButton("Auto-detect")
        self.auto_btn.setEnabled(False)
        self.auto_btn.setCursor(Qt.PointingHandCursor)
        self.auto_btn.setToolTip("Try several box sizes and pick the best")
        self.auto_btn.clicked.connect(self._on_auto_box)
        params_layout.addWidget(self.auto_btn)
```

Delete the entire `_on_auto_box` and `_on_auto_box_done` methods. Delete `self._auto_worker = None` from `__init__`. In `_load_acquisition`, delete `self.auto_btn.setEnabled(True)`.

- [ ] **Step 8: Update preview debounce constant**

In `MainWindow.__init__`, change:
```python
        self._preview_timer.setInterval(200)
```
to:
```python
        self._preview_timer.setInterval(_PREVIEW_DEBOUNCE_MS)
```

- [ ] **Step 9: Use the drop-label style constants**

Replace every `self.drop_label.setStyleSheet("QLabel { border: 2px dashed #aaa...` etc. with references to the new constants (`_DROP_LABEL_IDLE_STYLE`, `_DROP_LABEL_HOVER_STYLE`, `_DROP_LABEL_LOADED_STYLE`). Three call sites in `_setup_ui`, `_drag_enter`, `_drag_leave`, `_load_acquisition`.

- [ ] **Step 10: Drop the in-method import**

In `_load_acquisition`, delete the line `from bgsub.core import BackgroundSubtractor` (now at module top).

- [ ] **Step 11: Manual smoke check**

```bash
cd /Users/julioamaragall/Downloads/background_subtraction
python -c "from gui.app import MainWindow; print('ok')"
```

Expected: `ok` (no Qt instance is created on import).

- [ ] **Step 12: Commit**

```bash
git add gui/app.py
git commit -m "refactor(gui): drop auto-detect, hoist imports, named constants, clean axis state"
```

---

## Task 12: Rewrite `MovieWorker` to use public reader API + `process_single`

**Files:**
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/gui/app.py`

- [ ] **Step 1: Replace `MovieWorker.run`**

Replace the entire `class MovieWorker(QThread)` block in `gui/app.py` with:

```python
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
```

(All `_sep.Background` calls are gone. The duplicated sep pipeline is gone. `acq._find_files` access is gone. Progress emits at every `_MOVIE_FPS`-th frame — once per second of movie.)

- [ ] **Step 2: Smoke check**

```bash
python -c "from gui.app import MovieWorker; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add gui/app.py
git commit -m "refactor(gui): MovieWorker uses public reader API and BackgroundSubtractor.process_single"
```

---

## Task 13: GUI smoke test

**Files:**
- Create: `/Users/julioamaragall/Downloads/background_subtraction/tests/test_gui_smoke.py`

- [ ] **Step 1: Write test**

```python
"""Smoke import test for the GUI module. No Qt instance created."""
import importlib

import pytest


def test_gui_app_imports():
    pytest.importorskip("PyQt5")
    mod = importlib.import_module("gui.app")
    assert hasattr(mod, "MainWindow")
    assert hasattr(mod, "MovieWorker")
    assert hasattr(mod, "PreviewWorker")
    assert hasattr(mod, "ProcessWorker")


def test_gui_does_not_export_auto_box_worker():
    """Auto-detect box-size feature was removed; the worker class should be gone."""
    pytest.importorskip("PyQt5")
    mod = importlib.import_module("gui.app")
    assert not hasattr(mod, "AutoBoxSizeWorker")
```

- [ ] **Step 2: Run**

```bash
pytest tests/test_gui_smoke.py -v
```

Expected: 2 pass (or both skipped if PyQt5 not installed).

- [ ] **Step 3: Commit**

```bash
git add tests/test_gui_smoke.py
git commit -m "test: GUI smoke import + auto-detect removal check"
```

---

## Task 14: Delete dead code

**Files:**
- Delete: `/Users/julioamaragall/Downloads/background_subtraction/src/bgsub/io/` (entire directory)
- Delete: `/Users/julioamaragall/Downloads/background_subtraction/scripts/` (entire directory)
- Modify: `/Users/julioamaragall/Downloads/background_subtraction/src/bgsub/core.py` (drop unused `import shutil`)

- [ ] **Step 1: Verify nothing imports the dead modules**

```bash
cd /Users/julioamaragall/Downloads/background_subtraction
grep -rn "from bgsub.io\|from .io\|bgsub\.io\|bgsub/scripts\|from scripts\|from .scripts" src/ gui/ tests/ || echo "none"
```

Expected: `none`.

- [ ] **Step 2: Verify `import shutil` is unused in `core.py`**

```bash
grep -n "shutil" src/bgsub/core.py
```

Expected: only the `import shutil` line (which we removed in Task 8 already; this step confirms). If still present, edit `src/bgsub/core.py` and remove the line.

- [ ] **Step 3: Delete the directories**

```bash
git rm -r src/bgsub/io
git rm -r scripts
```

- [ ] **Step 4: Run full test suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: remove dead src/bgsub/io and scripts/ exploration code"
```

---

## Task 15: Final acceptance verification

**Files:** none (verification only)

- [ ] **Step 1: Clean install in a fresh environment**

```bash
cd /Users/julioamaragall/Downloads/background_subtraction
python -m venv /tmp/bgsub-verify
/tmp/bgsub-verify/bin/pip install -e .[dev,gui]
```

Expected: install succeeds with no warnings about missing README.

- [ ] **Step 2: Full pytest run**

```bash
/tmp/bgsub-verify/bin/pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Lint**

```bash
/tmp/bgsub-verify/bin/ruff check .
```

Expected: clean (or fix in-place and re-run).

- [ ] **Step 4: Spec acceptance grep checks**

```bash
# No leaky access to private reader internals
! grep -rn "_find_files\|_files_for_fov" gui/ src/bgsub/core.py
# sep filter referenced only from one place
test "$(grep -rn 'fw=' src/bgsub gui | wc -l | tr -d ' ')" = "1"
# suggest_box_size is gone
! grep -rn "suggest_box_size\|AutoBoxSizeWorker\|Auto-detect" src/ gui/
# pyyaml is gone from deps
! grep -n "pyyaml" pyproject.toml
# io module is gone
test ! -d src/bgsub/io
# scripts dir is gone
test ! -d scripts
# README exists
test -f README.md
```

Expected: all checks pass (each line either prints nothing or returns success).

- [ ] **Step 5: Manual GUI smoke**

If a real acquisition folder is available locally:
```bash
/tmp/bgsub-verify/bin/bgsub-gui
```
Drop in the acquisition. Verify:
- Format detected and channels listed.
- Preview updates when you move sliders.
- "Run Background Subtraction" produces output TIFFs.
- "Generate Movie" produces an mp4 per channel.

If no real acquisition is available, this step is skipped — the synthetic-fixture tests cover the non-Qt code paths and the `python -c "from gui.app import MainWindow"` smoke confirms the GUI module is importable.

- [ ] **Step 6: Tag the cleanup**

```bash
git log --oneline -20
```

Expected: clean commit history of the refactor. No commit needed for verification.

---

## Self-Review Checklist (filled in)

**1. Spec coverage:** every spec section maps to at least one task.

| Spec section | Task |
|---|---|
| §1 correctness #1 (`suggest_box_size` broken) | Task 9 |
| §1 correctness #2 (missing deps) | Task 1 |
| §1 correctness #3 (missing README) | Task 1 |
| §1 dead code #4-6 (`io/`, pyyaml, shutil) | Tasks 1, 14 |
| §1 dead code #7 (scripts/) | Task 14 |
| §1 leaks #8 (`_find_files`) | Tasks 4, 8, 12 |
| §1 leaks #9 (MovieWorker) | Task 12 |
| §1 leaks #10 (sep duplication) | Task 8 |
| §1 leaks #11 (axis state hack) | Task 11 |
| §1 leaks #12 (page_idx sentinel) | Tasks 3, 4, 5, 6, 8 |
| §1 magic constants | Tasks 8, 11 |
| §1 in-method imports | Task 11 |
| §1 swallowing exception handlers | Task 8 |
| §1 reader inefficiencies | Tasks 5, 6 |
| §1 frame_shape/frame_dtype glob | Task 4 |
| §1 no tests | Tasks 2-13 |
| §2.7 test scope | Tasks 4, 5, 6, 7, 8, 13 |

**2. Placeholder scan:** no `TBD` / `TODO` / "implement later" / "fill in details" in any task body. Every code-changing step contains the actual code.

**3. Type / signature consistency:**
- `FrameRef` defined in Task 3, consumed in Tasks 4-6, 8, 12.
- `iter_frames_for_fov(fov, channel)` signature is identical across all readers and base.
- `n_frames_per_fov(fov, channel)` on reader (Tasks 4-6, 10) vs `n_frames_per_fov(channel)` on `BackgroundSubtractor` (Task 8) — naming match, different argument count is intentional (subtractor is a higher-level convenience that picks the first FOV).
- `set_label` on `AxisSlider` defined in Task 11 step 5; called in Task 11 step 6.
- `_run_sep` signature `(image, box_size) -> (fg, bg)` consistent in Task 8 and consumed in Task 12.
- `_compute_normalization(refs) -> (raw_hi, fg_hi)` defined in Task 12, called once in same task.

No issues found.
