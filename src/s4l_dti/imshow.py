from __future__ import annotations

import SimpleITK as sitk


def plot(
    img: sitk.Image,
    factor: int = 1,
    title: str | None = None,
    margin: float = 0.0,
    dpi: int = 80,
):
    """Display a SimpleITK image using matplotlib.

    For 3D images a central axial slice is shown. For 4D vector images a
    central axial slice of the RGB/RGBA channels is displayed.

    Copied from:
    https://simpleitk.org/SimpleITK-Notebooks/10_matplotlib%27s_imshow.html

    Args:
        img: The SimpleITK image to display.
        factor: Integer scaling factor applied to the displayed image size.
        title: Optional title shown above the plot.
        margin: Fractional margin around the image axes (0.0 = no margin).
        dpi: Dots per inch for the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if c not in (3, 4):
            nda = nda[nda.shape[0] // 2, :, :]

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if c not in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        nda = nda[nda.shape[0] // 2, :, :, :]

    ysize = nda.shape[0] * factor
    xsize = nda.shape[1] * factor

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes((margin, margin, 1 - 2 * margin, 1 - 2 * margin))
    ax.set_axis_off()

    extent = (0, xsize * spacing[1], ysize * spacing[0], 0)
    t = ax.imshow(nda, extent=extent, interpolation=None)

    if nda.ndim == 2:
        t.set_cmap("gray")

    if title:
        plt.title(title)
