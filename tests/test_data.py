from s4l_dti.data import download_ixi_025


def test_download_ixi_025(tmp_path):
    files = download_ixi_025(tmp_path)
    assert files["dwi"].exists()
    assert files["bvec"].exists()
    assert files["bval"].exists()
    assert files["t1"].exists()
