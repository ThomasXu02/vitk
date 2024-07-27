import os
import numpy as np
import itk
import vtk
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import display
from display import *

# File paths
filepath1 = "Data/case6_gre1.nrrd"
filepath2 = "Data/case6_gre2.nrrd"
pixelType = itk.F
dimension = 3
ImageType = itk.Image[pixelType, dimension]

def itk_to_vtk_image(itk_image):
    numpy_array = itk.array_from_image(itk_image)
    vtk_image = vtk.vtkImageData()
    vtk_array = numpy_to_vtk(numpy_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
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
    
    segmented_np = np.zeros_like(np_image)
    for z in range(60, 100):
        segmented_slice = seg_region_growing(np_image[z])
        segmented_np[z] = segmented_slice

    segmented_image = itk.GetImageFromArray(segmented_np)
    segmented_image.SetOrigin(itk_image.GetOrigin())
    segmented_image.SetSpacing(itk_image.GetSpacing())
    segmented_image.SetDirection(itk_image.GetDirection())
    
    return segmented_image

def interpolate_volumes(volume1, volume2, alpha):
    interpolated_array = (1 - alpha) * vtk_to_numpy(volume1.GetPointData().GetScalars()) + alpha * vtk_to_numpy(volume2.GetPointData().GetScalars())
    interpolated_vtk = vtk.vtkImageData()
    interpolated_vtk.DeepCopy(volume1)
    interpolated_vtk.GetPointData().SetScalars(numpy_to_vtk(interpolated_array))
    return interpolated_vtk

def create_volume(vtk_image, color):
    mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    mapper.SetInputData(vtk_image)

    volume_color = vtk.vtkColorTransferFunction()
    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    volume_color.AddRGBPoint(255, color[0], color[1], color[2])

    volume_scalar_opacity = vtk.vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(0, 0.0)
    volume_scalar_opacity.AddPoint(255, 1.0)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(volume_color)
    volume_property.SetScalarOpacity(volume_scalar_opacity)
    volume_property.SetInterpolationTypeToLinear()

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)

    return volume

def update_volume(interpolated_volume, volume_property, volume):
    mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    mapper.SetInputData(interpolated_volume)
    
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)

def create_slider_widget(interactor, callback):
    slider_rep = vtk.vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(0.0)
    slider_rep.SetMaximumValue(1.0)
    slider_rep.SetValue(0.0)
    slider_rep.SetTitleText("Interpolation")

    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(0.1, 0.1)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(0.4, 0.1)
    
    slider_widget = vtk.vtkSliderWidget()
    slider_widget.SetInteractor(interactor)
    slider_widget.SetRepresentation(slider_rep)
    slider_widget.AddObserver("InteractionEvent", callback)
    slider_widget.EnabledOn()
    
    return slider_widget

def slider_callback(obj, event, volume1, volume2, volume_property, volume):
    slider_rep = obj.GetRepresentation()
    alpha = slider_rep.GetValue()
    interpolated_volume = interpolate_volumes(volume1, volume2, alpha)
    update_volume(interpolated_volume, volume_property, volume)
    obj.GetInteractor().GetRenderWindow().Render()

def display_interpolated_volumes(volume1, volume2):
    initial_volume = interpolate_volumes(volume1, volume2, 0.0)
    
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(vtk.vtkColorTransferFunction())
    volume_property.SetScalarOpacity(vtk.vtkPiecewiseFunction())
    volume_property.SetInterpolationTypeToLinear()

    volume = create_volume(initial_volume, (1.0, 0.0, 0.0))

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    renderer.AddVolume(volume)
    renderer.SetBackground(0, 0, 0)
    render_window.SetSize(800, 800)
    
    slider_widget = create_slider_widget(interactor, lambda obj, event: slider_callback(obj, event, volume1, volume2, volume_property, volume))
    
    render_window.Render()
    interactor.Start()

def convert_vtk_to_itk(vtk_image):
    vtk_np = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
    vtk_np = vtk_np.reshape(vtk_image.GetDimensions()[::-1])
    itk_image = itk.GetImageFromArray(vtk_np)
    return itk_image

def convert_itk_to_vtk(itk_image):
    itk_np = itk.GetArrayFromImage(itk_image)
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(itk_image.GetLargestPossibleRegion().GetSize())
    vtk_image.AllocateScalars(vtk.VTK_FLOAT, 1)
    vtk_np = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
    vtk_np[:] = itk_np.flatten()
    return vtk_image

if __name__ == "__main__":

    # Load the original image
    original_image = itk.imread(filepath1)
    original_image_np = itk.GetArrayFromImage(original_image)
    for i in range(original_image_np.shape[0]):
        original_image_np[i] = np.flipud(original_image_np[i])
    original_image_flipped = itk.GetImageFromArray(original_image_np)
    
    # Segment the first image
    seg1 = seg_region_growing_3d(original_image_flipped)
    seg1_vtk = convert_itk_to_vtk(seg1)

    # Register the images
    reg = register_images(filepath1, filepath2)
    reg_itk = convert_vtk_to_itk(reg)

    reg_np = itk.GetArrayFromImage(reg_itk)
    for i in range(reg_np.shape[0]):
        reg_np[i] = np.flipud(reg_np[i])
    reg_flipped = itk.GetImageFromArray(reg_np)
    
    # Segment the registered image
    seg2 = seg_region_growing_3d(reg_flipped)
    seg2_vtk = convert_itk_to_vtk(seg2)

    # Display the interpolated volumes
    display_interpolated_volumes(seg1_vtk, seg2_vtk)

