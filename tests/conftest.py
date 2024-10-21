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
