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
