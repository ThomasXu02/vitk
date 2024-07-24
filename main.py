import os
import itk
import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util import numpy_support

from display import *
from segmentation import segment_tumor_threshold, apply_threshold_to_slice

# File paths
filepath1 = "Data/case6_gre1.nrrd"
filepath2 = "Data/case6_gre2.nrrd"
pixelType = itk.F
dimension = 3
ImageType = itk.Image[pixelType, dimension]


def vtk_to_itk_image(vtk_image):
    """
    Convert a VTK image to an ITK image.
    """
    vtk_array = vtk_image.GetPointData().GetScalars()
    numpy_array = numpy_support.vtk_to_numpy(vtk_array)
    shape = vtk_image.GetDimensions()[::-1]
    numpy_array = numpy_array.reshape(shape)
    itk_image = itk.GetImageFromArray(numpy_array)
    itk_image.SetSpacing(vtk_image.GetSpacing())
    itk_image.SetOrigin(vtk_image.GetOrigin())
    itk_image.SetDirection(itk.GetMatrixFromArray(vtk_image.GetDirection()))
    return itk_image


def numpy_to_itk_image(array, reference_image):
    itk_image = itk.image_from_array(array.astype(np.uint8))
    itk_image.SetSpacing(reference_image.GetSpacing())
    itk_image.SetOrigin(reference_image.GetOrigin())
    itk_image.SetDirection(reference_image.GetDirection())
    return itk_image


def itk_to_vtk_image(itk_image):
    numpy_array = itk.array_from_image(itk_image)
    vtk_image = vtk.vtkImageData()
    vtk_array = numpy_support.numpy_to_vtk(numpy_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image.GetPointData().SetScalars(vtk_array)
    vtk_image.SetDimensions(itk_image.GetLargestPossibleRegion().GetSize())
    vtk_image.SetSpacing(itk_image.GetSpacing())
    vtk_image.SetOrigin(itk_image.GetOrigin())
    return vtk_image


def get_register_images(filepath1, filepath2):
    fixed_image = itk.imread(filepath1)
    moving_image = itk.imread(filepath2)

    # Cast images to float
    FloatImageType = itk.Image[itk.F, 3]
    fixed_image_float = itk.cast_image_filter(fixed_image, ttype=(type(fixed_image), FloatImageType))
    moving_image_float = itk.cast_image_filter(moving_image, ttype=(type(moving_image), FloatImageType))

    transform_type = itk.TranslationTransform[itk.D, 3]

    registration_method = itk.ImageRegistrationMethodv4[FloatImageType, FloatImageType].New()
    registration_method.SetFixedImage(fixed_image_float)
    registration_method.SetMovingImage(moving_image_float)

    initial_transform = transform_type.New()
    registration_method.SetInitialTransform(initial_transform)

    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    optimizer.SetLearningRate(4)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)
    registration_method.SetOptimizer(optimizer)

    metric = itk.MeanSquaresImageToImageMetricv4[FloatImageType, FloatImageType].New()
    registration_method.SetMetric(metric)

    registration_method.Update()
    final_transform = registration_method.GetTransform()

    resampler = itk.ResampleImageFilter[FloatImageType, FloatImageType].New()
    resampler.SetInput(moving_image_float)
    resampler.SetTransform(final_transform)
    resampler.SetReferenceImage(fixed_image_float)
    resampler.SetUseReferenceImage(True)
    resampler.SetSize(fixed_image_float.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_image_float.GetOrigin())
    resampler.SetOutputSpacing(fixed_image_float.GetSpacing())
    resampler.SetOutputDirection(fixed_image_float.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    registered_image = resampler.GetOutput()

    # Cast back to original type if needed
    if isinstance(fixed_image, itk.Image[itk.US, 3]):
        registered_image = itk.cast_image_filter(registered_image, ttype=(FloatImageType, itk.Image[itk.US, 3]))

    return registered_image


def register_images(image1, image2):
    registered_image = get_register_images(image1, image2)
    return registered_image


if __name__ == "__main__":
    original_itk_image1 = itk.imread(filepath1)
    # visualize_slice(original_itk_image1, slice_index=85, orientation='sagittal')
    # plot_slice_histogram(original_itk_image1, slice_index=85, orientation='sagittal')

    lower_threshold = 225
    upper_threshold = 255
    # segmented_slice_85 = apply_threshold_to_slice(original_itk_image1, slice_index=85,
    # lower_threshold=lower_threshold, upper_threshold=upper_threshold) visualize_segmented_slice(segmented_slice_85,
    # slice_index=85, orientation='sagittal')

    # itk_segmented_slice_85 = numpy_to_itk_image(segmented_slice_85, original_itk_image1)
    # vtk_segmented_slice_85 = itk_to_vtk_image(itk_segmented_slice_85)
    # display_volume_from_image(vtk_segmented_slice_85)

    # segmented_itk_image1 = segment_tumor_threshold(original_itk_image1, lower_threshold, upper_threshold)
    # segmented_vtk_image1 = itk_to_vtk_image(segmented_itk_image1)

    # display_volume_from_image(segmented_vtk_image1)

    # display_two_volumes(segmented_vtk_image1, segmented_vtk_image2)

    # original_vtk_image1 = itk_to_vtk_image(original_itk_image1)
    # registered_vtk_image = itk_to_vtk_image(registered_itk_image)

    # display_two_volumes(original_vtk_image1, registered_vtk_image)
