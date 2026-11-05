"""Tests for the mirrored-affine fix in `reconstruct_dti`.

A negative-determinant (mirrored) input affine must not leak into the
Sim4Life tensor output: by default `reconstruct_dti` flips exactly one voxel
axis with a world-preserving re-orientation so the saved affine is
right-handed (det > 0).
"""

from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

import s4l_dti.reconstruct as reconstruction


def _write_inputs(tmp_path, affine):
    """Create a mirrored/right-handed 5x5x5x7 DWI + bval/bvec and mock the fit."""
    data = np.arange(5 * 5 * 5 * 7, dtype=float).reshape(5, 5, 5, 7) + 1
    img_file = tmp_path / "dwi.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), img_file)
    bval_file = tmp_path / "dwi.bval"
    bvec_file = tmp_path / "dwi.bvec"
    np.savetxt(bval_file, [[0, 1000, 1000, 1000, 1000, 1000, 1000]])
    directions = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    directions[4:] /= np.sqrt(2)
    np.savetxt(bvec_file, directions.T)

    model = Mock()
    model.fit.return_value.lower_triangular.return_value = np.ones((5, 5, 5, 6))

    distance = np.abs(np.indices((5, 5, 5)) - 2).sum(axis=0)

    def fake_median_otsu(data, *, dilate, **kwargs):
        mask = distance <= dilate
        return np.where(mask[..., np.newaxis], data, 0.0), mask

    otsu = Mock(side_effect=fake_median_otsu)
    return img_file, bval_file, bvec_file, data, model, otsu


@pytest.fixture
def mirrored_inputs(tmp_path, monkeypatch):
    """5x5x5x7 DWI with a mirrored affine (diag(-2,2,2) -> det = -8)."""
    return _inputs_for(tmp_path, monkeypatch, np.diag([-2.0, 2.0, 2.0, 1.0]))


@pytest.fixture
def right_handed_inputs(tmp_path, monkeypatch):
    """5x5x5x7 DWI with a right-handed affine (diag(2,2,2) -> det = 8)."""
    return _inputs_for(tmp_path, monkeypatch, np.diag([2.0, 2.0, 2.0, 1.0]))


def _inputs_for(tmp_path, monkeypatch, affine):
    img_file, bval_file, bvec_file, data, model, otsu = _write_inputs(tmp_path, affine)
    monkeypatch.setattr(reconstruction, "TensorModel", Mock(return_value=model))
    monkeypatch.setattr(reconstruction, "median_otsu", otsu)
    kwargs = dict(
        img_file=img_file,
        bvec_file=bvec_file,
        bval_file=bval_file,
        s4l_dti_file=tmp_path / "tensor.nii.gz",
    )
    return kwargs, data, affine, model


def det3(affine):
    return float(np.linalg.det(np.asarray(affine)[:3, :3]))


def test_mirrored_affine_is_flipped_by_default(mirrored_inputs):
    kwargs, _, affine, _ = mirrored_inputs
    assert det3(affine) < 0  # sanity: the input really is mirrored

    reconstruction.reconstruct_dti(**kwargs)

    saved = nib.load(kwargs["s4l_dti_file"])
    assert det3(saved.affine) > 0


def test_flag_off_preserves_mirrored_affine(mirrored_inputs, tmp_path):
    kwargs, _, affine, _ = mirrored_inputs
    out = tmp_path / "uncorrected.nii.gz"

    reconstruction.reconstruct_dti(**{**kwargs, "s4l_dti_file": out}, fix_mirror=False)

    saved = nib.load(out)
    assert det3(saved.affine) < 0
    np.testing.assert_allclose(saved.affine, affine)


def test_flip_preserves_world_positions(mirrored_inputs, tmp_path):
    kwargs, _, _, model = mirrored_inputs
    # A tensor that varies along array axis 0 so a flip along that axis is
    # observable in the data.
    grad = np.arange(5, dtype=float)
    D = np.stack([np.broadcast_to(grad[:, None, None], (5, 5, 5))] * 6, axis=-1)
    model.fit.return_value.lower_triangular.return_value = D

    out_on = tmp_path / "corrected.nii.gz"
    out_off = tmp_path / "uncorrected.nii.gz"
    reconstruction.reconstruct_dti(**{**kwargs, "s4l_dti_file": out_on})
    reconstruction.reconstruct_dti(
        **{**kwargs, "s4l_dti_file": out_off}, fix_mirror=False
    )

    on = nib.load(out_on)
    off = nib.load(out_off)
    assert det3(on.affine) > 0
    assert det3(off.affine) < 0

    F = off.get_fdata()
    G = on.get_fdata()
    # The corrected volume is the un-corrected one mirrored along axis 0.
    np.testing.assert_allclose(G, F[::-1, :, :, :])

    # Every voxel of the corrected image lands on the same physical point as
    # the corresponding mirrored voxel of the un-corrected image.
    i, j, k = np.indices(F.shape[:3])
    coords = np.column_stack([i.ravel(), j.ravel(), k.ravel()])
    mirrored = coords.copy()
    mirrored[:, 0] = (F.shape[0] - 1) - mirrored[:, 0]
    world_on = nib.affines.apply_affine(on.affine, coords)
    world_off = nib.affines.apply_affine(off.affine, mirrored)
    np.testing.assert_allclose(world_on, world_off)


def test_right_handed_affine_is_left_untouched(right_handed_inputs):
    kwargs, _, affine, _ = right_handed_inputs

    reconstruction.reconstruct_dti(**kwargs)

    saved = nib.load(kwargs["s4l_dti_file"])
    np.testing.assert_allclose(saved.affine, affine)


def test_cli_forwards_fix_mirror(monkeypatch):
    typer = pytest.importorskip("typer")
    from typer.testing import CliRunner

    reconstruct = Mock()
    monkeypatch.setattr(reconstruction, "reconstruct_dti", reconstruct)

    captured_args: list[str] = []

    def run_cli(function):
        app = typer.Typer()
        app.command()(function)
        result = CliRunner().invoke(app, list(captured_args))
        assert result.exit_code == 0, result.output

    monkeypatch.setattr(typer, "run", run_cli)

    base = ["dwi.nii.gz", "dwi.bvec", "dwi.bval", "tensor.nii.gz"]

    captured_args[:] = base
    reconstruction.main()
    assert reconstruct.call_args.kwargs["fix_mirror"] is True

    captured_args[:] = base + ["--no-fix-mirror"]
    reconstruction.main()
    assert reconstruct.call_args.kwargs["fix_mirror"] is False
