# Copyright (c) 2024 The Foundation for Research on Information Technologies in Society (IT'IS).
#
# This file is part of s4l-dti
# (see https://github.com/dyollb/s4l-dti).
#
# This software is released under the MIT License.
#  https://opensource.org/licenses/MIT

from __future__ import annotations

import shutil
import tempfile
import typing
from pathlib import Path
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import requests
from dipy.data import fetch_stanford_hardi, fetch_stanford_labels, fetch_stanford_t1


def download_stanford_data(
    download_dir: Path = Path.home() / "Models" / "StanfordData",
) -> dict[str, Path]:
    """Download stanford dwi data, t1 image and labelfield

    Args:
        download_dir: specify folder where data is copied
    """

    def _fetch_and_copy(fetch_fun) -> list[Path]:
        files, dipy_dir = fetch_fun()
        for f in files.keys():
            shutil.copyfile(Path(dipy_dir) / f, Path(download_dir) / f)
        return [download_dir / f for f in files.keys()]

    download_dir.mkdir(exist_ok=True, parents=True)
    dwi_files = _fetch_and_copy(fetch_stanford_hardi)
    t1_files = _fetch_and_copy(fetch_stanford_t1)
    label_files = _fetch_and_copy(fetch_stanford_labels)

    dwi = next(f for f in dwi_files if f.name.endswith(".nii.gz"))
    bvec = next(f for f in dwi_files if f.name.endswith(".bvec"))
    bval = next(f for f in dwi_files if f.name.endswith(".bval"))
    t1 = next(f for f in t1_files if f.name.endswith(".nii.gz"))
    labels = next(f for f in label_files if f.name.endswith(".nii.gz"))
    return {"dwi": dwi, "bvec": bvec, "bval": bval, "t1": t1, "labels": labels}


def download_ixi_025(
    download_dir: Path = Path.home() / "Models" / "IXI025",
    force: bool = False,
) -> dict[str, Path]:
    """Download IXI025 data

    Args:
        download_dir: specify folder where data is copied
    """
    need_download = (
        force or (not download_dir.exists()) or (len(list(download_dir.glob("*"))) == 0)
    )
    if need_download:
        base_url = "https://github.com/ITISFoundation/IXI025/releases/download/v1.0.0/"
        file_name = "IXI025-Model_v1.0.0.zip"

        url = f"{base_url}/{file_name}"
        R = requests.get(f"{base_url}/{file_name}", allow_redirects=True)
        if R.status_code != 200:
            raise ConnectionError(
                f"could not download {url}\nerror code: {R.status_code}"
            )

        with tempfile.TemporaryDirectory() as tempdir:
            zip_path = Path(tempdir) / file_name
            zip_path.write_bytes(R.content)

            with ZipFile(zip_path, "r") as zip_obj:
                zip_obj.extractall(path=tempdir)

            zip_path.unlink()

            shutil.copytree(
                Path(tempdir) / "IXI025-Model_v1.0.0", download_dir, dirs_exist_ok=True
            )

        concatenate_dwi(
            download_dir / "dwi",
            download_dir / "dwi" / "IXI025-Guys-0852-DWI.nii.gz",
            skip_last=True,
        )

    dwi = next((download_dir / "dwi").glob("*-DWI.nii.gz"))
    bvec = next((download_dir / "dwi").glob("*-DWI.bvec"))
    bval = next((download_dir / "dwi").glob("*-DWI.bval"))
    labels = next((download_dir / "seg").glob("*.nii.gz"))
    t1 = next((download_dir / "anat").glob("*T1.nii.gz"))
    return {"dwi": dwi, "bvec": bvec, "bval": bval, "t1": t1, "labels": labels}


def concatenate_dwi(
    input_dir: Path,
    dwi_file: Path,
    glob: str = "IXI*-DTI-[0-9][0-9].nii.gz",
    skip_last: bool = True,
):
    """Concatenate DWI files into single image

    Note for the IXI dataset we must skip the last component:
    https://neurostars.org/t/ixi-dataset-diffusion-data-number-of-directions/3506
    """
    files = list(sorted(input_dir.glob(glob)))
    if skip_last:
        files = files[:-1]

    components = typing.cast(list[nib.Nifti1Image], [nib.load(f) for f in files])

    n_components = len(components)
    shape = list(components[0].dataobj.shape)
    affine = components[0].affine
    data = np.arange(np.prod(shape) * n_components, dtype=np.float32).reshape(
        shape + [n_components]
    )
    for i, c in enumerate(components):
        data[:, :, :, i] = c.get_fdata()
    dwi_raw = nib.Nifti1Image(data, affine)
    nib.save(dwi_raw, dwi_file)


def main():
    import typer

    from s4l_dti.cli import register_command

    app = typer.Typer()

    register_command(app, download_stanford_data)
    register_command(app, download_ixi_025)

    app()
