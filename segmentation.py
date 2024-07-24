import itk
import vtk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# File path
filepath = "Data/case6_gre2.nrrd"

def read_image(filepath):
    """Reads an NRRD file and returns it as a numpy array."""
    image = itk.imread(filepath)
    np_image = itk.GetArrayFromImage(image)
    return np_image, image

def preprocess_image(np_image):
    """Preprocesses the image (e.g., normalization, filtering)."""
    # Example: Normalization
    np_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))
    return np_image

# Easy segmentation with a threshold
def segment_tumor(np_image, low_threshold, high_threshold):
    """Segments the tumor using a thresholding method."""
    segmented_image1 = (np_image > low_threshold) 
    segmented_image2 = (np_image < high_threshold)
    segmented_image = segmented_image1 & segmented_image2
    segment_image = segmented_image[80, 50:100, 50:80]
    return segmented_image

#def segment_tumor(np_image):
#    """
#    Segments the tumor using Otsu's thresholding method.
#    
#    Args:
#    np_image (np.ndarray): The input image as a numpy array.
#    
#    Returns:
#    np.ndarray: The segmented image as a binary numpy array.
#    """
#    # Flatten the image to compute the histogram
#    image_flat = np_image.flatten()
#    
#    # Compute Otsu's threshold
#    threshold = threshold_otsu(image_flat)
#    
#    # Apply the threshold to create a binary image
#    segmented_image = np_image > (threshold * 3)
#    
#    return segmented_image




#import sys
#import itk
#import matplotlib
#import matplotlib.pyplot as plt
#import numpy as np
#import vtk

#do not work
def tp_segment_tumor(np_image, seedX=0.5, seedY=0.5, seedZ=0.5, lower=180, upper=255):
    """
    Segments a tumor from a 3D brain image using the ConnectedThreshold filter.
    
    Args:
    np_image (np.ndarray): The input image as a numpy array.
    seedX (int): The X coordinate for the seed point.
    seedY (int): The Y coordinate for the seed point.
    lower (float): The lower threshold value.
    upper (float): The upper threshold value.
    
    Returns:
    np.ndarray: The segmented image as a binary numpy array.
    """
    # Convert numpy array to ITK image
    input_image = itk.image_view_from_array(np_image.astype(np.float32))

    # Apply anisotropic diffusion smoothing
    smoother = itk.GradientAnisotropicDiffusionImageFilter.New(Input=input_image, NumberOfIterations=20, TimeStep=0.04,
                                                               ConductanceParameter=3)
    smoother.Update()

    # Instantiate the ConnectedThreshold filter
    connected_threshold = itk.ConnectedThresholdImageFilter.New(smoother.GetOutput())

    # Configure the filter
    connected_threshold.SetReplaceValue(255)
    connected_threshold.SetLower(lower)
    connected_threshold.SetUpper(upper)
    connected_threshold.SetSeed((seedX, seedY, seedZ))

    # Execute the pipeline
    connected_threshold.Update()

    # Convert the output image to a numpy array
    segmented_np_image = itk.array_view_from_image(connected_threshold.GetOutput())

    return segmented_np_image

def display_volume(np_image, title="Volume"):
    """
    Displays a 3D volume using VTK.
    
    Args:
    np_image (np.ndarray): The input image as a numpy array.
    title (str): The title for the render window.
    """
    # Convert the numpy array to a VTK image
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(np_image.shape)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    #for z in range(np_image.shape[0]):
    #    for y in range(np_image.shape[1]):
    #        for x in range(np_image.shape[2]):
    original_slice = np_image[80, 50:100, 50:80]
    for z in range(np_image.shape[0]):
        for y in range(50, 100):#np_image.shape[1]):
            for x in range(50,80):#np_image.shape[2]):
                vtk_image.SetScalarComponentFromFloat(x, y, z, 0, np_image[z, y, x])

    # Create volume mapper
    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputData(vtk_image)

    # Create volume color transfer function
    volume_color = vtk.vtkColorTransferFunction()
    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    volume_color.AddRGBPoint(255, 1.0, 1.0, 1.0)

    # Create volume opacity transfer function
    volume_opacity = vtk.vtkPiecewiseFunction()
    volume_opacity.AddPoint(0, 0.0)
    volume_opacity.AddPoint(255, 1.0)

    # Create volume property
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(volume_color)
    volume_property.SetScalarOpacity(volume_opacity)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    # Create volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)

    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName(title)

    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Initialize and start interaction
    render_window.Render()
    interactor.Start()

#def main():
#    # Assuming you have an image as a numpy array
#    # Replace this with your actual numpy array
#    input_image = itk.imread(filepath, pixel_type=itk.F)
#    np_image = itk.array_view_from_image(input_image)
#
#    # Segment the tumor
#    segmented_image = tp_segment_tumor(np_image, seedX=0, seedY=0, seedZ=0, lower=180, upper=255)
#
#    # Display the segmentation result in 3D
#    display_volume(segmented_image, title="Segmented Tumor Volume")
#
#if __name__ == "__main__":
#    matplotlib.use('TkAgg')
#    main()






















def display_segmentation(np_image, segmented_image):
    """Displays the original image and the segmentation result."""
    # Extract central slices for visualization
    slice_index = np_image.shape[0] // 2
    original_slice = np_image[80, 50:100, 50:80]
    segmented_slice = segmented_image[80, 50:100, 60:150]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].imshow(original_slice, cmap='gray')
    axs[0].set_title('Original Image Slice')
    axs[0].axis('off')
    
    axs[1].imshow(segmented_slice, cmap='gray')
    axs[1].set_title('Segmented Image Slice')
    axs[1].axis('off')
    
    plt.show()

def visualize_segmentation_3d(np_image, segmented_image):
    """Visualizes the segmentation result in 3D using VTK."""
    # Convert numpy arrays to ITK images
    image = itk.GetImageViewFromArray(np_image)
    segmented = itk.GetImageViewFromArray(segmented_image.astype(np.uint8))
    
    # Save the images to temporary files
    image_path = "image_temp.nrrd"
    segmented_path = "segmented_temp.nrrd"
    
    itk.imwrite(image, image_path)
    itk.imwrite(segmented, segmented_path)
    
    # Read the images using VTK
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(segmented_path)
    reader.Update()
    
    # Create a volume mapper
    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputConnection(reader.GetOutputPort())
    
    # Create a volume property
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    
    # Create color transfer function
    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 0.0)  # Black for low values
    color_transfer_function.AddRGBPoint(1, 1.0, 0.0, 0.0)  # Red for high values (tumor)
    volume_property.SetColor(color_transfer_function)
    
    # Create opacity transfer function
    opacity_transfer_function = vtk.vtkPiecewiseFunction()
    opacity_transfer_function.AddPoint(0, 0.0)  # Transparent for low values
    opacity_transfer_function.AddPoint(1, 1.0)  # Opaque for high values (tumor)
    volume_property.SetScalarOpacity(opacity_transfer_function)
    
    # Create a volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    
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

def main():
    # Read and preprocess the image
    np_image, itk_image = read_image(filepath)
    np_image = preprocess_image(np_image)
    
    # (Easy threshold) Segment the tumor
    low_threshold = 0.5  # You might need to adjust this threshold
    high_threshold = 0.6  # You might need to adjust this threshold
    segmented_image = segment_tumor(np_image, low_threshold, high_threshold)
    #segmented_image = segment_tumor(np_image)
    
    # Display the segmentation result
    display_segmentation(np_image, segmented_image)
    
    # Visualize the segmentation result in 3D
    visualize_segmentation_3d(np_image, segmented_image)
    #display_volume(segmented_image, title="Volume")

if __name__ == "__main__":
    main()

