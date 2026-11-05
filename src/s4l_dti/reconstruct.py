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
from typing import Literal

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


BvecFrame = Literal["fsl", "world"]


def _bvecs_to_world(
    bvecs: np.ndarray, affine: np.ndarray, bvec_frame: BvecFrame
) -> np.ndarray:
    """Express gradient directions in the world (scanner) frame of ``affine``.

    FSL b-vectors are given along the voxel axes, except that the first
    component is negated when the affine is right-handed (det > 0). They are
    rotated into world coordinates with the orthogonal part of the affine
    (polar decomposition, so voxel scaling and shear are ignored).
    """
    if bvec_frame == "world":
        return bvecs
    if bvec_frame != "fsl":
        raise ValueError(f"Unknown bvec_frame '{bvec_frame}'. Choose 'fsl' or 'world'.")

    linear = np.asarray(affine, dtype=float)[:3, :3]
    u, _, vt = np.linalg.svd(linear)
    rotation = u @ vt

    voxel_bvecs = np.array(bvecs, dtype=float)
    if np.linalg.det(linear) > 0:
        voxel_bvecs[:, 0] = -voxel_bvecs[:, 0]
    return voxel_bvecs @ rotation.T


def _flip_to_right_handed(image: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return ``image`` re-oriented so its affine is right-handed (det > 0).

    If the affine already has a non-negative determinant the image is
    returned unchanged. Otherwise exactly one voxel axis is reversed with a
    world-preserving re-orientation: the data and affine are updated together
    so every value stays at the same physical location. The flipped axis is
    the one that leaves the volume closest to canonical RAS+.

    The tensor components are expressed in the world frame (see
    ``_bvecs_to_world``), which a world-preserving voxel flip does not
    change, so they are intentionally left untouched.
    """
    affine = image.affine
    if np.linalg.det(affine[:3, :3]) >= 0:
        return image

    canonical = ("R", "A", "S")
    best_axis, best_score = 0, -1
    for axis in range(3):
        flipped = affine.copy()
        flipped[:3, axis] = -flipped[:3, axis]
        score = sum(
            code == want for code, want in zip(nib.aff2axcodes(flipped), canonical)
        )
        if score > best_score:
            best_axis, best_score = axis, score

    ornt = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int8)
    ornt[best_axis, 1] = -1
    return image.as_reoriented(ornt)


def reconstruct_dti(
    img_file: Path,
    bvec_file: Path,
    bval_file: Path,
    s4l_dti_file: Path,
    mask_file: Path | None = None,
    model: str = "auto",
    max_bval: int | None = None,
    otsu_mask_dilation: int = 2,
    use_mask_as_otsu_control: bool = False,
    otsu_mask_save_path: Path | None = None,
    fix_mirror: bool = True,
    bvec_frame: BvecFrame = "fsl",
) -> None:
    """Reconstruct diffusion tensor from DWI files.

    The tensor components are written in the world (scanner) frame of the
    DWI affine, which is the frame Sim4Life uses for tensor fields.

    Args:
        img_file: Path to 4D DWI NIfTI file.
        bvec_file: Path to gradient directions file (FSL format).
        bval_file: Path to b-values file (FSL format).
        s4l_dti_file: Path for output DTI tensor NIfTI (Sim4Life ordering,
            world-frame components).
        mask_file: Optional binary mask. If None, median_otsu is used.
        otsu_mask_dilation: Initial Otsu dilation iterations (default 2).
        use_mask_as_otsu_control: Use the resampled mask_file only as a
            containment control for median_otsu. Segment once, then grow the
            mask with six-neighbor dilation until every control voxel is inside.
            Requires mask_file and positive otsu_mask_dilation; raises ValueError if the Otsu
            mask cannot grow to cover the control.
        otsu_mask_save_path: Optional output path for the final Otsu mask. [Debugging Tool]
        fix_mirror: If True (default), flip the output tensor so its affine is
            right-handed (det > 0), guarding against mirrored transforms causing
            bugs downstream in Sim4Life. The flip preserves physical locations
            and, since components are world-frame, tensor orientations.
        bvec_frame: Frame of the b-vectors in ``bvec_file``. "fsl" (default)
            means FSL/dcm2niix convention (voxel axes, x negated for
            right-handed affines); they are rotated into the world frame before
            fitting. "world" means they are already in the world frame.
        model: Fitting model - "auto" (default), "dti", or "dki".
            "auto" selects DKI when >=2 non-zero b-value shells are present,
            otherwise DTI. DKI (Diffusional Kurtosis Imaging) correctly handles
            non-Gaussian diffusion at high b-values, while standard DTI assumes
            Gaussian diffusion and is biased for b >= 1500.
        max_bval: If set, discard volumes with b-value above this threshold
            before fitting. For example, max_bval=1100 on data with shells
            b=0/1000/2500 will keep only b=0 and b=1000 volumes.
    """

    if use_mask_as_otsu_control:
        if mask_file is None:
            raise ValueError("use_mask_as_otsu_control requires mask_file.")
        if otsu_mask_dilation < 1:
            raise ValueError(
                "otsu_mask_dilation must be positive in Otsu control mode."
            )

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

    control_mask = None
    if mask_file:
        maskimg = sitk.ReadImage(mask_file, sitk.sitkUInt16) != 0
        reference = extract_channel(sitk.ReadImage(img_file))
        maskimg = resample_to(maskimg, reference, nearest_neighbor=True)
        with tempfile.TemporaryDirectory() as tempdir:
            sitk.WriteImage(maskimg, Path(tempdir) / "mask.nii.gz")
            mask_nib = nib.load(Path(tempdir) / "mask.nii.gz")
            assert isinstance(mask_nib, nib.Nifti1Image)
            mask = np.asarray(mask_nib.get_fdata(), dtype=bool)
        if use_mask_as_otsu_control:
            control_mask = mask != 0
        else:
            maskdata = np.where(mask[..., np.newaxis], data, 0.0)

    if mask_file is None or use_mask_as_otsu_control:
        n_volumes = data.shape[-1]
        maskdata, mask = median_otsu(
            data,
            vol_idx=range(1, min(n_volumes, 15)),
            median_radius=4,
            numpass=4,
            autocrop=False,
            dilate=otsu_mask_dilation,
        )
        mask = np.asarray(mask, dtype=bool)
        if control_mask is not None:
            del maskdata
            while np.any(control_mask & ~mask):
                expanded = mask.copy()
                expanded[1:, :, :] |= mask[:-1, :, :]
                expanded[:-1, :, :] |= mask[1:, :, :]
                expanded[:, 1:, :] |= mask[:, :-1, :]
                expanded[:, :-1, :] |= mask[:, 1:, :]
                expanded[:, :, 1:] |= mask[:, :, :-1]
                expanded[:, :, :-1] |= mask[:, :, 1:]
                if np.array_equal(expanded, mask):
                    raise ValueError(
                        "Otsu mask cannot cover the control mask: "
                        f"mask is empty or stopped growing at dilation={otsu_mask_dilation}."
                    )
                mask = expanded
                otsu_mask_dilation += 1

            maskdata = np.where(mask[..., np.newaxis], data, 0.0)

        if otsu_mask_save_path:
            mask_image = nib.Nifti1Image(mask.astype(np.uint8), img.affine)
            nib.save(mask_image, otsu_mask_save_path)

    gtab = gradient_table(bvals, _bvecs_to_world(bvecs, img.affine, bvec_frame))

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
    if fix_mirror:
        image2 = _flip_to_right_handed(image2)
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
        otsu_mask_dilation: Annotated[
            int, typer.Option(help="Initial Otsu mask dilation iterations")
        ] = 2,
        use_mask_as_otsu_control: Annotated[
            bool,
            typer.Option(
                help="Increase Otsu dilation until it contains the supplied mask"
            ),
        ] = False,
        otsu_mask_save_path: Annotated[
            Path | None, typer.Option(help="Save the final Otsu mask")
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
        fix_mirror: Annotated[
            bool,
            typer.Option(
                help="Flip the tensor so its affine is right-handed (det > 0)"
            ),
        ] = True,
        bvec_frame: Annotated[
            str,
            typer.Option(
                help="B-vector frame: 'fsl' (voxel axes, FSL convention) or "
                "'world' (already in scanner frame)"
            ),
        ] = "fsl",
    ) -> None:
        reconstruct_dti(
            img_file=img_file,
            bvec_file=bvec_file,
            bval_file=bval_file,
            s4l_dti_file=s4l_dti_file,
            mask_file=mask_file,
            otsu_mask_dilation=otsu_mask_dilation,
            use_mask_as_otsu_control=use_mask_as_otsu_control,
            otsu_mask_save_path=otsu_mask_save_path,
            model=model,
            max_bval=max_bval,
            fix_mirror=fix_mirror,
            bvec_frame=bvec_frame,  # type: ignore[arg-type]
        )

    typer.run(_main)
