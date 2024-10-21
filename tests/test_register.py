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
