# Copyright (c) 2024 The Foundation for Research on Information Technologies in Society (IT'IS).
#
# This file is part of s4l-dti
# (see https://github.com/dyollb/s4l-dti).
#
# This software is released under the MIT License.
#  https://opensource.org/licenses/MIT

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.denoise.localpca import mppca


def denoise_mppca(
    image_file: Path,
    output_file: Path,
    mask_file: Path | None = None,
    patch_radius: int = 2,
    pca_method: str = "eig",
) -> None:
    """Run denoising using Marcenko-Pastur PCA

    Refer to dipy for more options and alternative algorithms:
    https://workshop.dipy.org/documentation/1.7.0/interfaces/denoise_flow/
    """
    img = nib.load(image_file)
    assert isinstance(img, nib.Nifti1Image)
    mask = None
    if mask_file:
        mask = nib.load(mask_file)
    denoised = mppca(
        img.get_fdata(),
        mask=mask.get_fdata() if mask else None,  # type: ignore [attr-defined]
        patch_radius=patch_radius,
        pca_method=pca_method,
    )
    img = nib.Nifti1Image(denoised, img.affine, dtype=np.float32)
    nib.save(img, output_file)


def main():
    import typer

    typer.run(denoise_mppca)


if __name__ == "__main__":
    main()
