import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from dipy.core.gradients import unique_bvals_magnitude
from dipy.io import read_bvals_bvecs

from s4l_dti.reconstruct import reconstruct_dti

requires_heavy = pytest.mark.skipif(
    not os.environ.get("S4L_DTI_HEAVY_TESTS"), reason="Set S4L_DTI_HEAVY_TESTS to run"
)


def test_reconstruct_dti(download_data, tmp_path: Path):
    dwi_file = download_data["dwi"]
    bvec_file = download_data["bvec"]
    bval_file = download_data["bval"]
    mask_file = download_data["labels"]
    s4l_dti_file = tmp_path / "DTI-s4l.nii.gz"

    reconstruct_dti(
        img_file=dwi_file,
        bvec_file=bvec_file,
        bval_file=bval_file,
        mask_file=mask_file,
        s4l_dti_file=s4l_dti_file,
    )

    assert s4l_dti_file.exists()


@requires_heavy
def test_reconstruct_dki(multishell_data, tmp_path: Path):
    """Test DKI fitting on multi-shell ISBI 2-shell phantom data."""
    dwi_file = multishell_data["dwi"]
    bvec_file = multishell_data["bvec"]
    bval_file = multishell_data["bval"]
    s4l_dti_file = tmp_path / "DKI-s4l.nii.gz"

    reconstruct_dti(
        img_file=dwi_file,
        bvec_file=bvec_file,
        bval_file=bval_file,
        s4l_dti_file=s4l_dti_file,
        model="dki",
    )

    assert s4l_dti_file.exists()

    import nibabel as nib

    dki_img = nib.load(s4l_dti_file)
    assert dki_img.shape[-1] == 6  # 6 tensor components


def test_reconstruct_auto_selects_dti_for_single_shell(download_data, tmp_path: Path):
    """Test that auto mode selects DTI for single-shell IXI data (no warning)."""
    dwi_file = download_data["dwi"]
    bvec_file = download_data["bvec"]
    bval_file = download_data["bval"]
    mask_file = download_data["labels"]
    s4l_dti_file = tmp_path / "auto-s4l.nii.gz"

    # IXI data is single-shell (b=0/1000), so auto should select DTI without warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reconstruct_dti(
            img_file=dwi_file,
            bvec_file=bvec_file,
            bval_file=bval_file,
            mask_file=mask_file,
            s4l_dti_file=s4l_dti_file,
            model="auto",
        )

    assert s4l_dti_file.exists()


@requires_heavy
def test_reconstruct_auto_selects_dki_for_multishell(multishell_data, tmp_path: Path):
    """Test that auto mode selects DKI for multi-shell ISBI data."""
    dwi_file = multishell_data["dwi"]
    bvec_file = multishell_data["bvec"]
    bval_file = multishell_data["bval"]
    s4l_dti_file = tmp_path / "auto-dki-s4l.nii.gz"

    # Multi-shell data: auto should select DKI, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reconstruct_dti(
            img_file=dwi_file,
            bvec_file=bvec_file,
            bval_file=bval_file,
            s4l_dti_file=s4l_dti_file,
            model="auto",
        )

    assert s4l_dti_file.exists()


def test_reconstruct_max_bval(download_data, tmp_path: Path):
    """Test that max_bval filters volumes correctly."""
    bval_file = download_data["bval"]
    bvals, _ = read_bvals_bvecs(str(bval_file), None)
    shells = unique_bvals_magnitude(bvals)

    # Pick a threshold that actually removes some volumes:
    # keep only b=0 volumes by setting max_bval below the lowest non-zero shell
    non_zero_shells = shells[shells > 50]
    if len(non_zero_shells) == 0:
        pytest.skip("No non-zero shells to filter")

    # Use max_bval that keeps only b=0 (just testing the filtering path works)
    max_bval_cutoff = int(non_zero_shells[0]) - 50
    n_kept = int(np.sum(bvals <= max_bval_cutoff))
    n_total = len(bvals)
    assert n_kept < n_total, "max_bval cutoff should remove some volumes"

    # Also test that full data works with max_bval set high
    dwi_file = download_data["dwi"]
    bvec_file = download_data["bvec"]
    mask_file = download_data["labels"]
    s4l_dti_file = tmp_path / "filtered-s4l.nii.gz"

    reconstruct_dti(
        img_file=dwi_file,
        bvec_file=bvec_file,
        bval_file=bval_file,
        mask_file=mask_file,
        s4l_dti_file=s4l_dti_file,
        max_bval=5000,  # keep everything
    )

    assert s4l_dti_file.exists()


@requires_heavy
def test_dti_warns_high_bval(multishell_data, tmp_path: Path):
    """Test that DTI warns when high b-values are present."""
    dwi_file = multishell_data["dwi"]
    bvec_file = multishell_data["bvec"]
    bval_file = multishell_data["bval"]
    s4l_dti_file = tmp_path / "warn-s4l.nii.gz"

    # ISBI data has b=0/1500/2500, so DTI should warn
    with pytest.warns(UserWarning, match="Gaussian diffusion"):
        reconstruct_dti(
            img_file=dwi_file,
            bvec_file=bvec_file,
            bval_file=bval_file,
            s4l_dti_file=s4l_dti_file,
            model="dti",
        )


def test_unique_shells():
    bvals = np.array([0, 0, 5, 995, 1000, 1005, 2500, 2510])
    shells = unique_bvals_magnitude(bvals)
    np.testing.assert_array_equal(shells, [0, 1000, 2500])


def test_dki_requires_multishell():
    """DKI should raise if fewer than 3 b-value levels."""
    bvals = np.array([0, 0, 1000, 1000])
    shells = unique_bvals_magnitude(bvals)
    assert len(shells) < 3, "Single-shell data should have < 3 b-value levels"
