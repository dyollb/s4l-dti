# Copyright (c) 2024 The Foundation for Research on Information Technologies in Society (IT'IS).
#
# This file is part of s4l-scripts
# (see https://github.com/dyollb/s4l-scripts).
#
# This software is released under the MIT License.
#  https://opensource.org/licenses/MIT

from __future__ import annotations

import tempfile
from os import fspath
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu

from .register import extract_channel, resample_to


def reconstruct_dti(
    img_file: Path,
    bvec_file: Path,
    bval_file: Path,
    s4l_dti_file: Path,
    mask_file: Path | None = None,
) -> None:
    """Reconstruct DTI from DWI files"""

    bvals, bvecs = read_bvals_bvecs(fspath(bval_file), fspath(bvec_file))
    img = nib.load(img_file)
    assert isinstance(img, nib.Nifti1Image)
    data = img.get_fdata()

    if mask_file:
        maskimg = sitk.ReadImage(mask_file, sitk.sitkUInt16) != 0
        reference = extract_channel(sitk.ReadImage(img_file))
        maskimg = resample_to(maskimg, reference, nearest_neighbor=True)
        with tempfile.TemporaryDirectory() as tempdir:
            sitk.WriteImage(maskimg, Path(tempdir) / "mask.nii.gz")
            mask_nib = nib.load(Path(tempdir) / "mask.nii.gz")
            mask = mask_nib.get_fdata()  # type: ignore [attr-defined]
        maskdata = np.where(mask[..., np.newaxis], data, 0.0)
    else:
        maskdata, _ = median_otsu(
            data,
            vol_idx=range(1, 15),
            median_radius=4,
            numpass=4,
            autocrop=False,
            dilate=2,
        )

    gtab = gradient_table(bvals, bvecs)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

    # lower_triangular returns DTI tensor components in order:
    #   Dxx, Dxy, Dyy, Dxz, Dyz, Dzz
    D = tenfit.lower_triangular()

    # Sim4Life expects this order: XX, YY, ZZ, XY, YZ, ZX
    ids = [0, 2, 5, 1, 4, 3]
    D_s4l = D[..., ids]
    image2 = nib.Nifti1Image(D_s4l, img.affine)
    nib.save(image2, s4l_dti_file)


def main():
    import typer

    typer.run(reconstruct_dti)
