# Copyright (c) 2024 The Foundation for Research on Information Technologies in Society (IT'IS).
#
# This file is part of s4l-dti
# (see https://github.com/dyollb/s4l-dti).
#
# This software is released under the MIT License.
#  https://opensource.org/licenses/MIT

import logging
import sys
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import TypeVar

import numpy as np
import SimpleITK as sitk

TransformType = TypeVar("TransformType", bound=sitk.Transform)


class RegistrationMetric(StrEnum):
    msqr = "msqr"
    ncc = "ncc"
    ants_ncc = "ants_ncc"
    mattes = "mattes"
    mi = "mi"
    demons = "demons"


class Transform(StrEnum):
    affine = "affine"
    euler = "euler"
    translation = "translate"
    versor = "versor"


def configure_metric(
    registration_method: sitk.ImageRegistrationMethod, metric: RegistrationMetric
) -> None:
    """Configure the similarity metric on a SimpleITK registration method.

    Args:
        registration_method: The registration method to configure.
        metric: The similarity metric to use.
    """
    if metric == RegistrationMetric.msqr:
        registration_method.SetMetricAsMeanSquares()
    elif metric == RegistrationMetric.ncc:
        registration_method.SetMetricAsCorrelation()
    elif metric == RegistrationMetric.ants_ncc:
        registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=5)
    elif metric == RegistrationMetric.mattes:
        registration_method.SetMetricAsMattesMutualInformation()
    elif metric == RegistrationMetric.mi:
        registration_method.SetMetricAsJointHistogramMutualInformation()
    elif metric == RegistrationMetric.demons:
        registration_method.SetMetricAsDemons()


def get_logger() -> logging.Logger:
    return logging.getLogger("reg")


def command_iteration(method: sitk.ImageRegistrationMethod, logger: logging.Logger):
    logger.info(
        f"{method.GetOptimizerIteration() + 1:3} = {method.GetMetricValue():10.5f}"
    )


def command_linear_iteration(
    method: sitk.ImageRegistrationMethod, logger: logging.Logger
):
    logger.info(
        f"{method.GetOptimizerIteration() + 1:3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}"
    )


def command_multiresolution_iteration(
    method: sitk.ImageRegistrationMethod, logger: logging.Logger
):
    logger.info(f"\tStop Condition: {method.GetOptimizerStopConditionDescription()}")
    logger.info(f"\t Iteration: {method.GetOptimizerIteration()}")
    logger.info(f"\t Metric value: {method.GetMetricValue()}")
    logger.info("============= Resolution Change =============")


def _linear_register(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    initial_transform: TransformType,
    moving_transform: sitk.Transform | None = None,
    metric: RegistrationMetric = RegistrationMetric.mattes.value,  # type: ignore
    fixed_mask: sitk.Image | None = None,
    moving_mask: sitk.Image | None = None,
    sampling_percentage: float = 0.2,
    num_iterations: int = 200,
    shrink_factors: list[int] = [1],
    smoothing_sigmas: list[float] = [0.0],  # in physical units
    interpolator=sitk.sitkLinear,
) -> TransformType:
    """Find a transform to align the moving image to the fixed image.

    The ``initial_transform`` defines the degrees of freedom. An optional
    ``moving_transform`` can provide an initial alignment guess applied before
    optimisation.

    Args:
        fixed_image: Reference image that the moving image is aligned to.
        moving_image: Image to be registered.
        initial_transform: Transform type and initial parameters to optimise.
        moving_transform: Optional pre-transform applied to the moving image
            before registration.
        metric: Similarity metric used during optimisation.
        fixed_mask: Optional mask restricting the metric to a region of the
            fixed image.
        moving_mask: Optional mask restricting the metric to a region of the
            moving image.
        sampling_percentage: Fraction of voxels sampled per iteration
            (0.0–1.0).
        num_iterations: Maximum number of optimiser iterations.
        shrink_factors: Downsampling factors per resolution level.
        smoothing_sigmas: Gaussian smoothing sigmas in physical units per
            resolution level.
        interpolator: SimpleITK interpolator used when resampling the moving
            image.

    Returns:
        The optimised transform of the same type as ``initial_transform``.
    """
    R = sitk.ImageRegistrationMethod()

    if moving_transform:
        R.SetMovingInitialTransform(moving_transform)

    R.SetInitialTransform(initial_transform, inPlace=True)

    configure_metric(R, metric)

    R.SetMetricSamplingStrategy(
        sitk.ImageRegistrationMethod.RANDOM
        if sampling_percentage < 1.0
        else sitk.ImageRegistrationMethod.NONE
    )
    R.SetMetricSamplingPercentage(sampling_percentage)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1e-1,
        minStep=1e-3,
        numberOfIterations=num_iterations,
        gradientMagnitudeTolerance=1e-6,
        estimateLearningRate=sitk.ImageRegistrationMethod.EachIteration,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInterpolator(interpolator)

    R.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if fixed_mask:
        R.SetMetricFixedMask(fixed_mask)
    if moving_mask:
        R.SetMetricMovingMask(moving_mask)

    logger = get_logger()

    R.AddCommand(
        sitk.sitkIterationEvent,
        partial(command_linear_iteration, R, logger.getChild("linear")),
    )
    R.AddCommand(
        sitk.sitkMultiResolutionIterationEvent,
        partial(command_multiresolution_iteration, R, logger.getChild("linear")),
    )

    logger.info("-" * 30)
    logger.info(f"Starting {type(initial_transform)} registration")
    logger.info("-" * 30)
    outTx = R.Execute(fixed_image, moving_image)

    logger.info("-" * 30)
    logger.debug(outTx)
    logger.info(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    logger.info(f" Iteration: {R.GetOptimizerIteration()}")
    logger.info(f" Metric value: {R.GetMetricValue()}")
    logger.info("-" * 30)

    return outTx


def register(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    fixed_mask: sitk.Image | None = None,
    moving_mask: sitk.Image | None = None,
    dof: Transform = Transform.affine.value,  # type: ignore
    metric: RegistrationMetric = RegistrationMetric.mattes.value,  # type: ignore
    sampling_percentage: float = 0.2,
    log_file: Path | None = None,
) -> sitk.Transform:
    """Align a moving image to a fixed image using linear registration.

    Args:
        fixed_image: Reference image that the moving image is aligned to.
        moving_image: Image to be registered.
        fixed_mask: Optional mask restricting the metric to a region of the
            fixed image.
        moving_mask: Optional mask restricting the metric to a region of the
            moving image.
        dof: Degrees of freedom controlling the transform family.
        metric: Similarity metric used during optimisation.
        sampling_percentage: Fraction of voxels sampled per iteration
            (0.0–1.0).
        log_file: Optional path to write the registration log.

    Returns:
        The estimated linear transform aligning ``moving_image`` to
        ``fixed_image``.
    """
    logger = get_logger()
    if logger and log_file:
        fh = logging.FileHandler(f"{log_file}", mode="w")
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    tx_init: sitk.Transform
    if dof == Transform.affine:
        tx_init = sitk.AffineTransform(3)
    elif dof == Transform.euler:
        tx_init = sitk.Euler3DTransform()
    elif dof == Transform.translation:
        tx_init = sitk.TranslationTransform(3)
    else:
        tx_init = sitk.VersorRigid3DTransform()

    tx_linear = _linear_register(
        fixed_image=fixed_image,
        moving_image=moving_image,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        metric=metric,
        shrink_factors=[8, 4, 2],
        smoothing_sigmas=[3.0, 1.5, 1.0],
        sampling_percentage=sampling_percentage,
        initial_transform=tx_init,
    )

    return tx_linear


def apply_transform_header(
    moving_image: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    """Apply a transform to an image by updating its header, without resampling.

    The origin and direction cosines are updated to reflect the transform so
    that voxel data remains unchanged. Only translation and rigid transforms
    are supported.

    Args:
        moving_image: 3D or 4D image whose header will be updated.
        transform: Rigid or translation transform to apply.

    Returns:
        A copy of ``moving_image`` with the updated origin and direction.

    Raises:
        RuntimeError: If the image is not 3D or 4D, or if the transform type
            is not supported.
    """
    if moving_image.GetDimension() not in (3, 4):
        raise RuntimeError("Only 3D/4D-images are supported")

    dim = moving_image.GetDimension()
    origin = np.asarray(moving_image.GetOrigin())[:3]
    direction = np.asarray(moving_image.GetDirection()).reshape(dim, dim)[:3, :3]

    new_origin, new_direction = origin, direction

    if isinstance(transform, sitk.TranslationTransform):
        offset = np.asarray(transform.GetOffset())

        new_origin[:3] = origin[:3] - offset
    elif isinstance(transform, (sitk.VersorRigid3DTransform, sitk.Euler3DTransform)):
        # https://github.com/BRAINSia/BRAINSTools/blob/main/BRAINSCommonLib/itkResampleInPlaceImageFilter.hxx
        inverse = np.asarray(transform.GetInverse().GetMatrix()).reshape(3, 3)
        offset = np.asarray(transform.GetTranslation())
        center = np.asarray(transform.GetCenter())

        # new_origin = [R^-1] * ( O - C - T ) + C
        new_origin[:3] = np.dot(inverse, origin[:3] - center - offset) + center
        new_direction[:3, :3] = np.dot(inverse, direction[:3, :3])
    else:
        raise RuntimeError(f"Transform is not supported {type(transform)}")

    # convert back to 4D origin / orientation
    if dim == 4:
        direction = np.eye(4, 4)
        direction[:3, :3] = new_direction
        origin = np.zeros((4,))
        origin[:3] = new_origin
        new_origin, new_direction = origin, direction

    moving_image = sitk.Image(moving_image)
    moving_image.SetOrigin(tuple(new_origin.tolist()))
    moving_image.SetDirection(tuple(new_direction.flatten().tolist()))
    return moving_image


def extract_channel(image: sitk.Image, idx: int = 0):
    """Extract a single channel from a multi-channel image.

    Args:
        image: Multi-channel SimpleITK image.
        idx: Zero-based channel index to extract.

    Returns:
        A 3D scalar image containing the selected channel, with the same
        spatial metadata as the input.
    """
    tmp = image[..., idx]
    comp = sitk.GetImageFromArray(sitk.GetArrayFromImage(tmp))
    comp.CopyInformation(tmp)
    return comp


def ones_like(image: sitk.Image) -> sitk.Image:
    """Create an image of ones with the same geometry as the input.

    Args:
        image: Reference image whose spatial metadata is copied.

    Returns:
        A new image filled with ones sharing the same origin, spacing, and
        direction as ``image``.
    """
    ones = sitk.GetImageFromArray(np.ones_like(sitk.GetArrayViewFromImage(image)))
    ones.CopyInformation(image)
    return ones


def resample_to(
    image: sitk.Image,
    ref_image: sitk.Image,
    nearest_neighbor: bool = True,
):
    """Resample an image onto the grid of a reference image.

    Args:
        image: Image to resample.
        ref_image: Reference image defining the target grid.
        nearest_neighbor: If ``True``, use nearest-neighbour interpolation
            (suitable for label images); otherwise use linear interpolation.

    Returns:
        The resampled image on the reference grid.
    """
    return sitk.Resample(
        image,
        ref_image,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor if nearest_neighbor else sitk.sitkLinear,
    )


def main():
    import typer

    app = typer.Typer()

    @app.command(name="register", help=register.__doc__)
    def register_cli(
        fixed_image: Path,
        moving_image: Path,
        output_transform: Path,
        fixed_mask: Path | None = None,
        moving_mask: Path | None = None,
        dof: Transform = Transform.affine.value,  # type: ignore
        metric: RegistrationMetric = RegistrationMetric.mattes.value,  # type: ignore
        log: Path | None = None,
    ) -> None:
        def optional_read(file_path, pixel_type):
            return sitk.ReadImage(file_path, pixel_type) if file_path else None

        tx_linear = register(
            fixed_image=sitk.ReadImage(fixed_image, sitk.sitkFloat32),
            moving_image=sitk.ReadImage(moving_image, sitk.sitkFloat32),
            dof=dof,
            metric=metric,
            fixed_mask=optional_read(fixed_mask, sitk.sitkUInt16),
            moving_mask=optional_read(moving_mask, sitk.sitkUInt16),
            log_file=log,
        )
        sitk.WriteTransform(tx_linear, output_transform)

    @app.command(name="apply_transform_header", help=apply_transform_header.__doc__)
    def apply_transform_header_cli(
        moving_image: Path, transform: Path, transformed_moving_image: Path
    ) -> None:
        result = apply_transform_header_cli(
            sitk.ReadImage(moving_image), sitk.ReadTransform(transform)
        )
        sitk.WriteImage(result, transformed_moving_image)

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s]: %(message)s", stream=sys.stdout
    )

    app()
