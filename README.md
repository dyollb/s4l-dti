# DTI Pre-Processing for Sim4Life

[![Build Actions Status](https://github.com/dyollb/s4l-dti/workflows/CI/badge.svg)](https://github.com/dyollb/s4l-dti/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://https://opensource.org/licenses/MIT)

This Python package provides pre-processing functions to help [Sim4Life](https://sim4life.swiss/) users prepare diffusion tensor images (DTI) for use in low-frequency electro-magnetic simulations. The primary goal is to convert DTI into a format supported by Sim4Life, enabling the assignment of anisotropic inhomogeneous conductivity maps in tissue models.

### Features

- Align diffusion weighted images (DWI) to structural images, e.g. T1-weighted MRI (using SimpleITK)
- Noise removal for DWI (using dipy)
- DTI reconstruction from DWI (using dipy)
- Save DTI in Sim4Life compatible ordering (XX, YY, ZZ, XY, YZ, ZX)
- Example data is automatically downloaded

### Installation

```sh
pip install s4l-dti
```

or

```
pip install git+https://github.com/dyollb/s4l-dti.git#egg=s4l-dti
```

### Usage

Download IXI025 head dataset:

```py
from s4l_dti.data import download_ixi_025

download_dir = Path.home() / "Models" / "IXI025"
download_files = download_ixi_025(download_dir)

for key in download_files:
    print(f"Downloaded {key}: {download_files[key].relative_to(download_dir)}")
```

Reconstruction and saving as Sim4Life compatible Nifti file:

```py
from s4l_dti.reconstruct import reconstruct_dti

dwi_image_file = download_files["dwi"]
bvec_file = download_files["bvec"]
bval_file = download_files["bval"]
mask_file = download_files["labels"]
s4l_dti_file = download_dir / "DTI-s4l.nii.gz"

reconstruct_dti(
    img_file=dwi_aligned_denoised_image_file,
    bvec_file=bvec_file,
    bval_file=bval_file,
    mask_file=mask_file
    s4l_dti_file=s4l_dti_file,
)
```

For more examples, see the [Jupyter notebook](notebooks/tutorial_dti_conductivity.ipynb).
