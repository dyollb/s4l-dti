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


_S4L_TO_MATRIX = [[0, 3, 5], [3, 1, 4], [5, 4, 2]]


def _principal_direction(components):
    tensor = np.asarray(components)[_S4L_TO_MATRIX]
    return np.linalg.eigh(tensor)[1][:, -1]


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _fit_anisotropic(tmp_path, monkeypatch, affine, bvec_direction, **kwargs):
    """Fit a real tensor whose principal axis is ``bvec_direction`` in the
    frame of the b-vector file, and return the saved tensor image."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvecs = np.vstack([[0.0, 0.0, 0.0], directions])
    bvals = np.r_[0.0, np.full(30, 1000.0)]

    v = _unit(bvec_direction)
    D = 1e-3 * (0.2 * np.eye(3) + 1.5 * np.outer(v, v))
    signal = 1000 * np.exp(-bvals * np.einsum("ni,ij,nj->n", bvecs, D, bvecs))
    data = np.broadcast_to(signal, (5, 5, 5, signal.size)).copy()

    img_file = tmp_path / "dwi.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), img_file)
    np.savetxt(tmp_path / "dwi.bval", bvals[np.newaxis])
    np.savetxt(tmp_path / "dwi.bvec", bvecs.T)

    def full_mask(data, **_):
        return data, np.ones(data.shape[:3], dtype=bool)

    monkeypatch.setattr(reconstruction, "median_otsu", full_mask)
    out = tmp_path / kwargs.pop("name", "tensor.nii.gz")
    reconstruction.reconstruct_dti(
        img_file=img_file,
        bvec_file=tmp_path / "dwi.bvec",
        bval_file=tmp_path / "dwi.bval",
        s4l_dti_file=out,
        model="dti",
        **kwargs,
    )
    return nib.load(out)


# Each case: affine, principal direction in the FSL b-vector frame, and the
# expected physical (world) direction, derived by hand from the FSL convention
# (voxel axes; x negated when det(affine) > 0).
_ROT_Z = [[0, -2, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]
_MIRROR_SWAP = [[0, -2, 0, 0], [-2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]


@pytest.mark.parametrize(
    "affine, fsl_direction, world_direction",
    [
        pytest.param(np.diag([-2.0, 2, 2, 1]), [-1, 1, 0], [1, 1, 0], id="LAS-xy"),
        pytest.param(np.diag([-2.0, 2, 2, 1]), [-1, 0, 1], [1, 0, 1], id="LAS-xz"),
        pytest.param(np.diag([2.0, 2, 2, 1]), [-1, 1, 0], [1, 1, 0], id="RAS-xy"),
        pytest.param(np.diag([2.0, 2, 2, 1]), [0, 1, 1], [0, 1, 1], id="RAS-yz"),
        pytest.param(np.array(_ROT_Z, float), [-2, -1, 0], [1, 2, 0], id="rot-z"),
        pytest.param(
            np.array(_MIRROR_SWAP, float), [-2, -1, 0], [1, 2, 0], id="mirrored-swap"
        ),
    ],
)
@pytest.mark.parametrize("fix_mirror", [True, False])
def test_tensor_direction_is_physical(
    tmp_path, monkeypatch, affine, fsl_direction, world_direction, fix_mirror
):
    saved = _fit_anisotropic(
        tmp_path, monkeypatch, affine, fsl_direction, fix_mirror=fix_mirror
    )
    principal = _principal_direction(saved.get_fdata()[2, 2, 2])
    assert abs(principal @ _unit(world_direction)) == pytest.approx(1, abs=1e-6)


def test_world_frame_bvecs_are_used_as_is(tmp_path, monkeypatch):
    saved = _fit_anisotropic(
        tmp_path,
        monkeypatch,
        np.array(_MIRROR_SWAP, float),
        [1, 2, 0],
        bvec_frame="world",
    )
    principal = _principal_direction(saved.get_fdata()[2, 2, 2])
    assert abs(principal @ _unit([1, 2, 0])) == pytest.approx(1, abs=1e-6)


def test_unknown_bvec_frame_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="bvec_frame"):
        _fit_anisotropic(
            tmp_path, monkeypatch, np.eye(4), [1, 0, 0], bvec_frame="voxel"
        )


def test_flip_keeps_world_frame_components(tmp_path, monkeypatch):
    affine = np.diag([-2.0, 2, 2, 1])
    on = _fit_anisotropic(tmp_path, monkeypatch, affine, [-1, 1, 1], name="on.nii.gz")
    off = _fit_anisotropic(
        tmp_path, monkeypatch, affine, [-1, 1, 1], fix_mirror=False, name="off.nii.gz"
    )
    assert det3(on.affine) > 0
    np.testing.assert_allclose(on.get_fdata(), off.get_fdata()[::-1])
