def test_download_ixi_025(download_data):
    assert download_data["dwi"].exists()
    assert download_data["bvec"].exists()
    assert download_data["bval"].exists()
    assert download_data["t1"].exists()
