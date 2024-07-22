import os
import itk
import vtk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# File paths
filepath1 = "Data/case6_gre1.nrrd"
filepath2 = "Data/case6_gre2.nrrd"

def read_and_extract_slice(filepath):
    """Reads an NRRD file and extracts the central slice."""
    image = itk.imread(filepath)
    np_image = itk.GetArrayFromImage(image)
    z_index = np_image.shape[0] // 2
    central_slice = np_image[z_index, :, :]
    return central_slice

def display_images2d(filepath1, filepath2):
    """Displays two images and their difference."""
    
    slice1 = read_and_extract_slice(filepath1)
    slice2 = read_and_extract_slice(filepath2)
    
    diff_slice = np.abs(slice1 - slice2)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display the first image slice
    axs[0].imshow(slice1, cmap='gray', origin='lower')
    axs[0].set_title('Central Slice - Image 1')
    axs[0].axis('off')
    
    # Display the second image slice
    axs[1].imshow(slice2, cmap='gray', origin='lower')
    axs[1].set_title('Central Slice - Image 2')
    axs[1].axis('off')
    
    # Display the difference slice
    axs[2].imshow(diff_slice, cmap='gray', origin='lower')
    axs[2].set_title('Difference Slice')
    axs[2].axis('off')
    
    plt.show()

def display_volume(filePath):
    # Read the NRRD file
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filePath)
    reader.Update()

    # Get the range of the data
    scalar_range = reader.GetOutput().GetScalarRange()

    # Create a volume mapper
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputConnection(reader.GetOutputPort())

    # Create a volume property
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    # Create color transfer function
    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)  # Black for low values
    colorTransferFunction.AddRGBPoint(scalar_range[1] * 0.5, 0.5, 0.0, 0.0)  # Dark red for mid values
    colorTransferFunction.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)  # Bright red for high values (likely tumor)
    volumeProperty.SetColor(colorTransferFunction)

    # Create opacity transfer function
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(scalar_range[0], 0.0)  # Transparent for low values
    opacityTransferFunction.AddPoint(scalar_range[1] * 0.3, 0.0)  # Still transparent for lower mid-range
    opacityTransferFunction.AddPoint(scalar_range[1] * 0.7, 0.2)  # Start to appear
    opacityTransferFunction.AddPoint(scalar_range[1], 1.0)  # Opaque for high values (likely tumor)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)

    # Create a volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # Create a renderer and render window
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 800)  # Larger window

    # Create an interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Set the interactor style
    interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(interactorStyle)

    # Initialize and start the interaction
    renderWindowInteractor.Initialize()
    renderer.ResetCamera()  # Reset the camera to show the whole volume
    renderWindow.Render()
    renderWindowInteractor.Start()

def register_images(filepath1, filepath2):
    """
    Performs a rigid registration of two 3D NRRD images using Euler3DTransform.
    
    Args:
    filepath1 (str): Path to the fixed image file (NRRD format)
    filepath2 (str): Path to the moving image file (NRRD format)
    
    Returns:
    np.ndarray: The registered moving image as a numpy array
    """
    # Define image type
    PixelType = itk.F
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]

    # Read the images
    fixed_image = itk.imread(filepath1, ImageType)
    moving_image = itk.imread(filepath2, ImageType)

    # Initialize the transform
    TransformType = itk.Euler3DTransform[itk.D]
    initial_transform = TransformType.New()

    # Set up the registration method
    RegistrationType = itk.ImageRegistrationMethodv4[ImageType, ImageType]
    registration_method = RegistrationType.New()
    registration_method.SetFixedImage(fixed_image)
    registration_method.SetMovingImage(moving_image)
    registration_method.SetInitialTransform(initial_transform)

    # Set the similarity metric
    MetricType = itk.CorrelationImageToImageMetricv4[ImageType, ImageType]
    metric = MetricType.New()
    registration_method.SetMetric(metric)

    # Set the optimizer
    OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
    optimizer = OptimizerType.New()
    optimizer.SetLearningRate(1.0)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)
    registration_method.SetOptimizer(optimizer)

    # Set up multi-resolution framework
    registration_method.SetNumberOfLevels(3)
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])

    # Perform the registration
    registration_method.Update()

    # Get the final transform
    final_transform = registration_method.GetTransform()

    # Resample the moving image
    ResampleFilterType = itk.ResampleImageFilter[ImageType, ImageType]
    resampler = ResampleFilterType.New()
    resampler.SetInput(moving_image)
    resampler.SetTransform(final_transform)
    resampler.SetUseReferenceImage(True)
    resampler.SetReferenceImage(fixed_image)
    resampler.SetDefaultPixelValue(100)

    # Execute the resampler
    resampler.Update()
    registered_image = resampler.GetOutput()

    # Convert to numpy array
    registered_np_image = itk.array_from_image(registered_image)

    return registered_np_image

def visualize_registration(fixed_image, registered_image, slice_axis=0):
    #TODO
    return 0





def main():
    display_volume(filepath1)
    #display_images2d(filepath1, filepath2)
    
    # fixed_image = itk.array_from_image(itk.imread(filepath1, itk.F))
    # registered_image = register_images(filepath1, filepath2)
    # visualize_registration(fixed_image, registered_image)

if __name__ == "__main__":
    main()
