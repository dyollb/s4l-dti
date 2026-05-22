# Copyright (c) 2024 The Foundation for Research on Information Technologies in Society (IT'IS).
#
# This file is part of s4l-dti
# (see https://github.com/dyollb/s4l-dti).
#
# This software is released under the MIT License.
#  https://opensource.org/licenses/MIT

import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

import nibabel as nib
import numpy as np
from dipy.data import fetch_stanford_hardi, fetch_stanford_labels, fetch_stanford_t1


def download_file(url: str, dest_folder: Path, timeout: float | None = None):
    """Download a file from a URL to a local folder.

    Args:
        url: HTTP(S) URL of the file to download.
        dest_folder: Destination directory. Created if it does not exist.
        timeout: Optional timeout in seconds for the HTTP request.

    Returns:
        Path to the downloaded file inside ``dest_folder``.
    """
    dest_folder.mkdir(exist_ok=True, parents=True)
    file_name = dest_folder / url.split("/")[-1]

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            # If the HTTP response status is not 200, HTTPError is raised
            with open(file_name, "wb") as out_file:
                out_file.write(response.read())

        print(f"Downloaded {file_name}")

    except HTTPError as e:
        print(f"HTTP error occurred: {e.code} {e.reason}")
    except URLError as e:
        print(f"Failed to reach the server: {e.reason}")
    except TimeoutError:
        print(f"Failed to download {url} due to a timeout.")

    return file_name


def download_stanford_data(
    download_dir: Path = Path.home() / "Models" / "StanfordData",
) -> dict[str, Path]:
    """Download the Stanford HARDI dataset (DWI, T1, and label image).

    Args:
        download_dir: Directory where the data files are copied.

    Returns:
        Dictionary with keys ``dwi``, ``bvec``, ``bval``, ``t1``, and
        ``labels`` mapping to the corresponding file paths.
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
    """Download the IXI025 diffusion dataset.

    Args:
        download_dir: Directory where the data files are extracted.
        force: Re-download and overwrite even if files already exist.

    Returns:
        Dictionary with keys ``dwi``, ``bvec``, ``bval``, ``t1``, and
        ``labels`` mapping to the corresponding file paths.
    """
    need_download = (
        force or (not download_dir.exists()) or (len(list(download_dir.glob("*"))) == 0)
    )
    if need_download:
        url = "https://github.com/ITISFoundation/IXI025/releases/download/v1.0.0/IXI025-Model_v1.0.0.zip"

        with tempfile.TemporaryDirectory() as tempdir:
            zip_path = download_file(url=url, dest_folder=Path(tempdir))

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
    """Concatenate individual DWI component files into a single 4D NIfTI image.

    Note:
        For the IXI dataset the last component must be skipped due to a known
        issue with the gradient table. See
        https://neurostars.org/t/ixi-dataset-diffusion-data-number-of-directions/3506

    Args:
        input_dir: Directory containing the individual DWI component files.
        dwi_file: Output path for the concatenated 4D NIfTI image.
        glob: Glob pattern to match component files within ``input_dir``.
        skip_last: If ``True``, the last matched file is excluded before
            concatenating.
    """
    files = list(sorted(input_dir.glob(glob)))
    if skip_last:
        files = files[:-1]

    components = [cast(nib.Nifti1Image, nib.load(f)) for f in files]

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
