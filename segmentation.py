import itk
from sklearn.cluster import KMeans
import numpy as np


def cast_to_supported_type(image):
    pixel_type, dimension = itk.template(image)[1]
    supported_types = [
        itk.SS, itk.UC, itk.US, itk.F, itk.D
    ]

    if pixel_type not in supported_types:
        float_image_type = itk.Image[itk.F, dimension]
        caster = itk.CastImageFilter[type(image), float_image_type].New()
        caster.SetInput(image)
        caster.Update()
        return caster.GetOutput()
    return image


def segment_tumor_threshold(image, lower_threshold, upper_threshold):
    image = cast_to_supported_type(image)

    threshold_filter = itk.BinaryThresholdImageFilter.New(image)
    threshold_filter.SetLowerThreshold(lower_threshold)
    threshold_filter.SetUpperThreshold(upper_threshold)
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)
    threshold_filter.Update()

    median_filter = itk.MedianImageFilter.New(threshold_filter.GetOutput())
    median_filter.SetRadius(2)
    median_filter.Update()

    smoothing_filter = itk.SmoothingRecursiveGaussianImageFilter.New(median_filter.GetOutput())
    smoothing_filter.SetSigma(1.0)
    smoothing_filter.Update()

    return smoothing_filter.GetOutput()


def apply_threshold_to_slice(image, slice_index, lower_threshold, upper_threshold, orientation='sagittal'):
    array = itk.array_from_image(image)
    if orientation == 'sagittal':
        slice_data = array[slice_index, :, :]
    elif orientation == 'axial':
        slice_data = array[:, slice_index, :]
    elif orientation == 'coronal':
        slice_data = array[:, :, slice_index]
    else:
        raise ValueError("Invalid orientation: choose 'axial', 'sagittal', or 'coronal'")

    thresholded_slice = np.where((slice_data >= lower_threshold) & (slice_data <= upper_threshold), 1, 0)
    return thresholded_slice
