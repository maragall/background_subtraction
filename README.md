# bgsub

Minimal background subtraction for fluorescence microscopy acquisitions, built on
[sep](https://sep.readthedocs.io/).

## Install

```bash
pip install -e '.[gui]'
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
pip install -e '.[dev,gui]'
pytest
ruff check .
```

## License

BSD-3-Clause
