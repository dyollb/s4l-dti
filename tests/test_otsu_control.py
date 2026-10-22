from pathlib import Path
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

import s4l_dti.reconstruct as reconstruction


@pytest.fixture
def reconstruction_inputs(tmp_path, monkeypatch):
    data = np.arange(5 * 5 * 5 * 7, dtype=float).reshape(5, 5, 5, 7) + 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
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
    monkeypatch.setattr(reconstruction, "TensorModel", Mock(return_value=model))

    distance = np.abs(np.indices((5, 5, 5)) - 2).sum(axis=0)

    def fake_median_otsu(data, *, dilate, **kwargs):
        mask = distance <= dilate
        return np.where(mask[..., np.newaxis], data, 0.0), mask

    otsu = Mock(side_effect=fake_median_otsu)
    monkeypatch.setattr(reconstruction, "median_otsu", otsu)
    kwargs = dict(
        img_file=img_file,
        bvec_file=bvec_file,
        bval_file=bval_file,
        s4l_dti_file=tmp_path / "tensor.nii.gz",
        otsu_mask_save_path=tmp_path / "otsu.nii.gz",
    )
    return kwargs, data, affine, model, otsu


@pytest.mark.parametrize(
    ("control_voxel", "initial_dilation", "expected_dilations"),
    [
        ((4, 2, 2), 2, [2]),
        ((4, 4, 2), 2, [2, 3, 4]),
        ((4, 4, 4), 3, [3, 4, 5, 6]),
    ],
)
def test_control_mask_adaptive_dilation(
    reconstruction_inputs,
    tmp_path,
    control_voxel,
    initial_dilation,
    expected_dilations,
):
    kwargs, data, affine, model, otsu = reconstruction_inputs
    control = np.zeros(data.shape[:3], dtype=np.uint8)
    control[control_voxel] = 7
    mask_file = tmp_path / "control.nii.gz"
    nib.save(nib.Nifti1Image(control, affine), mask_file)

    reconstruction.reconstruct_dti(
        **kwargs,
        mask_file=mask_file,
        use_mask_as_otsu_control=True,
        otsu_mask_dilation=initial_dilation,
    )

    assert [call.kwargs["dilate"] for call in otsu.call_args_list] == expected_dilations
    saved = nib.load(kwargs["otsu_mask_save_path"])
    mask = saved.get_fdata().astype(bool)
    expected = (
        np.abs(np.indices(control.shape) - 2).sum(axis=0) <= expected_dilations[-1]
    )
    np.testing.assert_array_equal(mask, expected)
    assert not np.any((control != 0) & ~mask)
    assert np.any(mask & (control == 0))
    np.testing.assert_allclose(saved.affine, affine)
    np.testing.assert_array_equal(
        model.fit.call_args.args[0], np.where(mask[..., np.newaxis], data, 0.0)
    )
    assert kwargs["s4l_dti_file"].exists()


@pytest.mark.parametrize("with_mask", [False, True])
def test_masking_without_control_preserves_behavior(
    reconstruction_inputs, tmp_path, with_mask
):
    kwargs, data, affine, model, otsu = reconstruction_inputs
    if with_mask:
        mask = np.zeros(data.shape[:3], dtype=np.uint8)
        mask[1, 1, 1] = 1
        mask_file = tmp_path / "mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), mask_file)
        kwargs["mask_file"] = mask_file
    else:
        mask = np.abs(np.indices(data.shape[:3]) - 2).sum(axis=0) <= 2

    reconstruction.reconstruct_dti(**kwargs)

    assert otsu.call_count == (0 if with_mask else 1)
    assert kwargs["otsu_mask_save_path"].exists() is not with_mask
    np.testing.assert_array_equal(
        model.fit.call_args.args[0], np.where(mask[..., np.newaxis], data, 0.0)
    )


def test_control_mode_requires_mask(reconstruction_inputs):
    kwargs, _, _, model, otsu = reconstruction_inputs
    with pytest.raises(ValueError, match="requires mask_file"):
        reconstruction.reconstruct_dti(**kwargs, use_mask_as_otsu_control=True)
    otsu.assert_not_called()
    model.fit.assert_not_called()


@pytest.mark.parametrize("dilation", [0, -1])
def test_control_mode_requires_positive_dilation(reconstruction_inputs, dilation):
    kwargs, _, _, model, otsu = reconstruction_inputs
    with pytest.raises(ValueError, match="must be positive"):
        reconstruction.reconstruct_dti(
            **kwargs,
            mask_file=Path("control.nii.gz"),
            use_mask_as_otsu_control=True,
            otsu_mask_dilation=dilation,
        )
    otsu.assert_not_called()
    model.fit.assert_not_called()


@pytest.mark.parametrize("empty", [True, False])
def test_unreachable_control_fails_instead_of_looping(
    reconstruction_inputs, tmp_path, empty
):
    kwargs, data, affine, model, otsu = reconstruction_inputs
    control = np.ones(data.shape[:3], dtype=np.uint8)
    mask_file = tmp_path / "control.nii.gz"
    nib.save(nib.Nifti1Image(control, affine), mask_file)
    mask = np.zeros(control.shape, dtype=bool)
    if not empty:
        mask[2, 2, 2] = True
    otsu.side_effect = None
    otsu.return_value = (np.where(mask[..., np.newaxis], data, 0.0), mask)

    with pytest.raises(ValueError, match="cannot cover the control mask"):
        reconstruction.reconstruct_dti(
            **kwargs,
            mask_file=mask_file,
            use_mask_as_otsu_control=True,
        )

    assert otsu.call_count == (1 if empty else 2)
    model.fit.assert_not_called()
    assert not kwargs["otsu_mask_save_path"].exists()


def test_control_mode_with_real_median_otsu(
    reconstruction_inputs, tmp_path, monkeypatch
):
    from dipy.segment.mask import median_otsu

    kwargs, _, affine, model, _ = reconstruction_inputs
    data = np.zeros((25, 25, 25, 7))
    data[3:-3, 3:-3, 3:-3] = 100
    nib.save(nib.Nifti1Image(data, affine), kwargs["img_file"])
    control = np.zeros(data.shape[:3], dtype=np.uint8)
    control[24, 12, 12] = 1
    mask_file = tmp_path / "control.nii.gz"
    nib.save(nib.Nifti1Image(control, affine), mask_file)
    model.fit.return_value.lower_triangular.return_value = np.ones((25, 25, 25, 6))
    otsu = Mock(wraps=median_otsu)
    monkeypatch.setattr(reconstruction, "median_otsu", otsu)

    reconstruction.reconstruct_dti(
        **kwargs, mask_file=mask_file, use_mask_as_otsu_control=True
    )

    dilations = [call.kwargs["dilate"] for call in otsu.call_args_list]
    assert len(dilations) > 1
    assert dilations == list(range(2, dilations[-1] + 1))
    saved = nib.load(kwargs["otsu_mask_save_path"]).get_fdata().astype(bool)
    assert not np.any((control != 0) & ~saved)
    _, previous = median_otsu(
        data,
        vol_idx=range(1, 7),
        median_radius=4,
        numpass=4,
        autocrop=False,
        dilate=dilations[-1] - 1,
    )
    assert np.any((control != 0) & ~previous)
    np.testing.assert_array_equal(
        model.fit.call_args.args[0], np.where(saved[..., np.newaxis], data, 0.0)
    )


def test_cli_forwards_otsu_options(monkeypatch, tmp_path):
    typer = pytest.importorskip("typer")
    from typer.testing import CliRunner

    reconstruct = Mock()
    monkeypatch.setattr(reconstruction, "reconstruct_dti", reconstruct)

    def run_cli(function):
        app = typer.Typer()
        app.command()(function)
        result = CliRunner().invoke(
            app,
            [
                "dwi.nii.gz",
                "dwi.bvec",
                "dwi.bval",
                "tensor.nii.gz",
                "--mask-file",
                "control.nii.gz",
                "--use-mask-as-otsu-control",
                "--otsu-mask-dilation",
                "3",
                "--otsu-mask-save-path",
                str(tmp_path / "otsu.nii.gz"),
            ],
        )
        assert result.exit_code == 0, result.output

    monkeypatch.setattr(typer, "run", run_cli)
    reconstruction.main()
    assert reconstruct.call_args.kwargs["use_mask_as_otsu_control"] is True
    assert reconstruct.call_args.kwargs["otsu_mask_dilation"] == 3
    assert reconstruct.call_args.kwargs["mask_file"] == Path("control.nii.gz")
    assert (
        reconstruct.call_args.kwargs["otsu_mask_save_path"] == tmp_path / "otsu.nii.gz"
    )
