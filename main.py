import os
import itk
import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util import numpy_support

import display
from display import *

# File paths
filepath1 = "Data/case6_gre1.nrrd"
filepath2 = "Data/case6_gre2.nrrd"
pixelType = itk.F
dimension = 3
ImageType = itk.Image[pixelType, dimension]

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

def itk_to_vtk_image(itk_image):
    numpy_array = itk.array_from_image(itk_image)
    vtk_image = vtk.vtkImageData()
    vtk_array = numpy_support.numpy_to_vtk(numpy_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image.GetPointData().SetScalars(vtk_array)
    vtk_image.SetDimensions(itk_image.GetLargestPossibleRegion().GetSize())
    vtk_image.SetSpacing(itk_image.GetSpacing())
    vtk_image.SetOrigin(itk_image.GetOrigin())
    return vtk_image
    
def register_images(filepath1, filepath2):
    registered_image = get_register_images(filepath1, filepath2)
    return itk_to_vtk_image(registered_image)

def seg_region_growing(image):
    itk_image = itk.GetImageFromArray(image.astype(np.float32))
    
    ImageType = itk.Image[itk.F, 2]
    seg = itk.ConnectedThresholdImageFilter[ImageType, ImageType].New()
    seg.SetInput(itk_image)
    
    p1 = (120, 188)
    p2 = (98, 175)

    low = 420
    high = 930

    for i in range (-4, 4):
        for j in range (-4, 4):
            a = itk_image.GetPixel((p1[0] + i, p1[1] + j))
            b = itk_image.GetPixel((p2[0] + i, p2[1] + j))
            if a > low and a < high:
                seg.AddSeed((p1[0] + i, p1[1] + j))
            if b > low and b < high:
                seg.AddSeed((p2[0] + i, p2[1] + j))


    seg.SetLower(low)
    seg.SetUpper(high)
    
    seg.SetReplaceValue(900)
    seg.Update()
    output_array = itk.GetArrayFromImage(seg.GetOutput())
    return output_array

def seg_region_growing_3d(original_image):
    FloatImageType = itk.Image[itk.F, 3]
    if not isinstance(original_image, FloatImageType):
        cast_filter = itk.CastImageFilter[type(original_image), FloatImageType].New()
        cast_filter.SetInput(original_image)
        cast_filter.Update()
        itk_image = cast_filter.GetOutput()
    else:
        itk_image = original_image
    
    np_image = itk.array_from_image(itk_image)
    image_min = np.min(np_image)
    image_max = np.max(np_image)
    
    #print("3D Image min/max:", image_min, image_max)
    
    segmented_np = np.zeros_like(np_image)
    for z in range(60, 100):
        segmented_slice = seg_region_growing(np_image[z])
        segmented_np[z] = segmented_slice

    #print("Segmented 3D Image min/max:", np.min(segmented_np), np.max(segmented_np))
    
    segmented_image = itk.GetImageFromArray(segmented_np)
    segmented_image.SetOrigin(itk_image.GetOrigin())
    segmented_image.SetSpacing(itk_image.GetSpacing())
    segmented_image.SetDirection(itk_image.GetDirection())
    
    return segmented_image

def vtk_to_numpy(vtk_image):
    dims = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    numpy_array = numpy_support.vtk_to_numpy(vtk_array)
    numpy_array = numpy_array.reshape(dims[2], dims[1], dims[0])
    return numpy_array

if __name__ == "__main__":
    original_image = itk.imread(filepath1)
    original_image_np = itk.GetArrayFromImage(original_image)
    for i in range(original_image_np.shape[0]):
        original_image_np[i] = np.flipud(original_image_np[i])
    original_image_flipped = itk.GetImageFromArray(original_image_np)
    seg1 = seg_region_growing_3d(original_image_flipped)
    seg1 = itk_to_vtk_image(seg1)


    # Register 
    reg = register_images(filepath1, filepath2)
    reg_np = vtk_to_numpy(reg)
    for i in range(reg_np.shape[0]):
        reg_np[i] = np.flipud(reg_np[i])
    reg_flipped = itk.GetImageFromArray(reg_np)
    seg2 = seg_region_growing_3d(reg_flipped)
    seg2 = itk_to_vtk_image(seg2)

    # Display the two volumes
    display_two_volumes(seg1, seg2)



