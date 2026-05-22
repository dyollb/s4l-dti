# Copyright (c) 2024 The Foundation for Research on Information Technologies in Society (IT'IS).
#
# This file is part of s4l-scripts
# (see https://github.com/dyollb/s4l-scripts).
#
# This software is released under the MIT License.
#  https://opensource.org/licenses/MIT

import tempfile
import warnings
from os import fspath
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from dipy.core.gradients import gradient_table, unique_bvals_magnitude
from dipy.io import read_bvals_bvecs
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu

from .register import extract_channel, resample_to

_HIGH_BVAL_THRESHOLD = 1500
_B0_THRESHOLD = 50


class HighBValueWarning(UserWarning):
    """Raised when DTI is used with b-values where Gaussian diffusion breaks down."""


def reconstruct_dti(
    img_file: Path,
    bvec_file: Path,
    bval_file: Path,
    s4l_dti_file: Path,
    mask_file: Path | None = None,
    model: str = "auto",
    max_bval: int | None = None,
) -> None:
    """Reconstruct diffusion tensor from DWI files.

    Args:
        img_file: Path to 4D DWI NIfTI file.
        bvec_file: Path to gradient directions file (FSL format).
        bval_file: Path to b-values file (FSL format).
        s4l_dti_file: Path for output DTI tensor NIfTI (Sim4Life ordering).
        mask_file: Optional binary mask. If None, median_otsu is used.
        model: Fitting model - "auto" (default), "dti", or "dki".
            "auto" selects DKI when >=2 non-zero b-value shells are present,
            otherwise DTI. DKI (Diffusional Kurtosis Imaging) correctly handles
            non-Gaussian diffusion at high b-values, while standard DTI assumes
            Gaussian diffusion and is biased for b >= 1500.
        max_bval: If set, discard volumes with b-value above this threshold
            before fitting. For example, max_bval=1100 on data with shells
            b=0/1000/2500 will keep only b=0 and b=1000 volumes.
    """

    bvals, bvecs = read_bvals_bvecs(fspath(bval_file), fspath(bvec_file))
    img = nib.load(img_file)
    assert isinstance(img, nib.Nifti1Image)
    data = img.get_fdata()

    # Filter volumes by max_bval if requested
    if max_bval is not None:
        keep = bvals <= max_bval
        if not np.any(keep):
            raise ValueError(
                f"max_bval={max_bval} would discard all volumes. "
                f"B-values present: {np.unique(bvals)}"
            )
        non_b0_kept = np.sum(bvals[keep] > _B0_THRESHOLD)
        if non_b0_kept == 0:
            raise ValueError(
                f"max_bval={max_bval} discards all non-b0 volumes. "
                f"At least one diffusion-weighted volume (b > {_B0_THRESHOLD}) "
                f"is required for tensor fitting. "
                f"B-values present: {np.unique(bvals)}"
            )
        bvals = bvals[keep]
        bvecs = bvecs[keep]
        data = data[..., keep]

    # Determine model to use
    shells = unique_bvals_magnitude(bvals)
    non_zero_shells = shells[shells > _B0_THRESHOLD]

    if model == "auto":
        use_dki = len(non_zero_shells) >= 2
    elif model == "dki":
        use_dki = True
        if len(shells) < 3:
            raise ValueError(
                f"DKI requires at least 3 b-value levels (including b=0), "
                f"but only {len(shells)} found: {shells}. "
                f"Use model='dti' for single-shell data."
            )
    elif model == "dti":
        use_dki = False
    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'auto', 'dti', or 'dki'.")

    # Warn if DTI is used with high b-values
    if not use_dki and np.any(non_zero_shells >= _HIGH_BVAL_THRESHOLD):
        warnings.warn(
            f"Standard DTI fitting assumes Gaussian diffusion, which breaks down "
            f"at b >= {_HIGH_BVAL_THRESHOLD}. The data contains b-value shells: "
            f"{shells.tolist()}. Consider using model='dki' for unbiased tensor "
            f"estimation, or set max_bval to restrict to lower shells.",
            HighBValueWarning,
            stacklevel=2,
        )

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
        n_volumes = data.shape[-1]
        maskdata, _ = median_otsu(
            data,
            vol_idx=range(1, min(n_volumes, 15)),
            median_radius=4,
            numpass=4,
            autocrop=False,
            dilate=2,
        )

    gtab = gradient_table(bvals, bvecs)

    if use_dki:
        tenmodel = DiffusionKurtosisModel(gtab)
    else:
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
    from typing import Annotated

    import typer

    def _main(
        img_file: Path,
        bvec_file: Path,
        bval_file: Path,
        s4l_dti_file: Path,
        mask_file: Annotated[
            Path | None, typer.Option(help="Binary mask image")
        ] = None,
        model: Annotated[
            str,
            typer.Option(
                help="Fitting model: 'auto' selects DKI for multi-shell data, "
                "'dti' forces standard tensor, 'dki' forces kurtosis tensor"
            ),
        ] = "auto",
        max_bval: Annotated[
            int | None,
            typer.Option(help="Discard volumes with b-value above this threshold"),
        ] = None,
    ) -> None:
        reconstruct_dti(
            img_file=img_file,
            bvec_file=bvec_file,
            bval_file=bval_file,
            s4l_dti_file=s4l_dti_file,
            mask_file=mask_file,
            model=model,
            max_bval=max_bval,
        )

    typer.run(_main)
