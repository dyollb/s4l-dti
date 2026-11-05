import numpy as np
import SimpleITK as sitk

from s4l_dti.register import (
    RegistrationMetric,
    Transform,
    extract_channel,
    ones_like,
    register,
)


def test_register_dwi_to_t1(download_data):
    dwi_path = download_data["dwi"]
    t1_path = download_data["t1"]

    dwi_image = sitk.ReadImage(dwi_path, sitk.sitkVectorFloat32)
    assert dwi_image.GetDimension() == 4
    t1_image = sitk.ReadImage(t1_path, sitk.sitkFloat32)

    # Extract (3D) component 0 from (4D) DWI image
    # - registration requires fixed image to have dimension as moving image
    dwi_image0 = extract_channel(dwi_image)
    assert dwi_image0.GetDimension() == 3

    tx = register(
        moving_image=dwi_image0,
        fixed_image=t1_image,
        moving_mask=ones_like(dwi_image0),
        dof=Transform.euler,
        metric=RegistrationMetric.mattes,
        sampling_percentage=1.0,
    )
    assert isinstance(tx, sitk.Euler3DTransform)

    np.testing.assert_allclose(
        tx.GetTranslation(), [-2.233617, -1.625291, -0.538782], atol=0.1
    )


def test_apply_transform_header_cli_uses_function(monkeypatch):
    """The apply_transform_header CLI must call apply_transform_header, not itself."""
    from pathlib import Path
    from unittest.mock import Mock

    import typer

    import s4l_dti.register as registration

    # register.main() imports typer locally and ends by invoking the app, so
    # swap in a fake Typer that only records the registered command functions.
    registered: dict = {}

    class FakeTyper:
        def __init__(self, *args, **kwargs):
            pass

        def command(self, *args, **kwargs):
            name = kwargs.get("name")

            def decorator(fn):
                registered[name or fn.__name__] = fn
                return fn

            return decorator

        def __call__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(typer, "Typer", FakeTyper)
    registration.main()
    cli = registered["apply_transform_header"]

    sentinel_moving = object()
    sentinel_transform = object()
    sentinel_result = object()
    read_image = Mock(return_value=sentinel_moving)
    read_transform = Mock(return_value=sentinel_transform)
    write_image = Mock()
    apply_header = Mock(return_value=sentinel_result)
    monkeypatch.setattr(registration.sitk, "ReadImage", read_image)
    monkeypatch.setattr(registration.sitk, "ReadTransform", read_transform)
    monkeypatch.setattr(registration.sitk, "WriteImage", write_image)
    monkeypatch.setattr(registration, "apply_transform_header", apply_header)

    moving = Path("moving.nii.gz")
    transform = Path("transform.hdf")
    output = Path("out.nii.gz")
    cli(moving, transform, output)

    read_image.assert_called_once_with(moving)
    read_transform.assert_called_once_with(transform)
    apply_header.assert_called_once_with(sentinel_moving, sentinel_transform)
    write_image.assert_called_once_with(sentinel_result, output)
