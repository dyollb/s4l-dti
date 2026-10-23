# DTI Pre-Processing for Sim4Life

[![Build Actions Status](https://github.com/dyollb/s4l-dti/workflows/CI/badge.svg)](https://github.com/dyollb/s4l-dti/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/s4l-dti.svg)](https://badge.fury.io/py/s4l-dti)

This Python package provides pre-processing functions to help [Sim4Life](https://sim4life.swiss/) users prepare diffusion tensor images (DTI) for use in low-frequency electro-magnetic simulations. The primary goal is to convert DTI into a format supported by Sim4Life, enabling the assignment of anisotropic inhomogeneous conductivity maps in tissue models.

### Features

- Align diffusion weighted images (DWI) to structural images, e.g. T1-weighted MRI (using SimpleITK)
- Noise removal for DWI (using dipy)
- DTI reconstruction from DWI (using dipy)
- Save DTI in Sim4Life compatible ordering (XX, YY, ZZ, XY, YZ, ZX) (see [reconstruct.py](https://github.com/dyollb/s4l-dti/blob/3a7485f50eebac3167245fde888cdc6b2b382410/src/s4l_dti/reconstruct.py#L69))
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
    mask_file=mask_file,
    s4l_dti_file=s4l_dti_file,
)
```

To use an existing mask as a containment control rather than as the fitting
mask, set `use_mask_as_otsu_control=True` and provide `mask_file`. The control
mask is resampled onto the DWI grid with nearest-neighbor interpolation.
Reconstruction runs `median_otsu` starting at `otsu_mask_dilation=2` and increases
the dilation by one until all control voxels are inside the Otsu mask.
The final Otsu mask (not its intersection with the control) is used for fitting.
Set `otsu_mask_save_path` to save that mask. Control mode requires a positive
initial dilation and raises an error if the Otsu mask is empty or stops growing
before containment is reached.

The CLI exposes the same settings as `--use-mask-as-otsu-control`,
`--otsu-mask-dilation`, and `--otsu-mask-save-path`, together with `--mask-file`.

For more examples, see the [Jupyter notebook](notebooks/dti_preprocessing.ipynb).
