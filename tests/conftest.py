from __future__ import annotations

import os
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest

from s4l_dti.data import download_ixi_025


# Fixture to download data and clean up afterwards
@pytest.fixture(scope="session")
def download_data(tmp_path_factory) -> Generator[dict[str, Path]]:
    tmp_dir = tmp_path_factory.mktemp("downloaded_data")

    data_dir = tmp_dir
    if "CACHED_DOWNLOAD_DIR" in os.environ:
        data_dir = Path(os.environ["CACHED_DOWNLOAD_DIR"])

    files = download_ixi_025(data_dir)

    # Yield the directory path to the tests
    yield files

    # Cleanup after the session
    shutil.rmtree(tmp_dir)
    print(f"Cleaned up {tmp_dir}")


@pytest.fixture(scope="session")
def multishell_data(tmp_path_factory) -> Generator[dict[str, Path]]:
    """ISBI 2013 2-shell phantom dataset (b=0/1500/2500, 64 volumes, 50^3)."""
    import nibabel as nib
    import numpy as np
    from dipy.data import read_isbi2013_2shell

    img, gtab = read_isbi2013_2shell()
    tmp_dir = tmp_path_factory.mktemp("multishell_data")

    dwi_file = tmp_dir / "dwi.nii.gz"
    bval_file = tmp_dir / "dwi.bval"
    bvec_file = tmp_dir / "dwi.bvec"

    nib.save(img, dwi_file)
    np.savetxt(bval_file, gtab.bvals[np.newaxis, :], fmt="%d")
    np.savetxt(bvec_file, gtab.bvecs.T, fmt="%.6f")

    yield {"dwi": dwi_file, "bvec": bvec_file, "bval": bval_file}

    shutil.rmtree(tmp_dir)
    print(f"Cleaned up {tmp_dir}")
