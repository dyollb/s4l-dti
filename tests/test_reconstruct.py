from pathlib import Path

from s4l_dti.reconstruct import reconstruct_dti


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
