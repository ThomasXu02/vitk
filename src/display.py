import itk
import vtk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from vtk.util import numpy_support

def read_and_extract_slice(filepath, z_index=85):
    """Reads an NRRD file and extracts the central slice."""
    image = itk.imread(filepath)
    np_image = itk.GetArrayFromImage(image)
    central_slice = np_image[z_index, :, :]
    flipped_slice = np.flipud(central_slice)
    return flipped_slice

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
    
def display_volume_from_path(filePath):
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filePath)
    reader.Update()

    scalar_range = reader.GetOutput().GetScalarRange()

    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputConnection(reader.GetOutputPort())

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(scalar_range[1] * 0.5, 0.5, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)
    volumeProperty.SetColor(colorTransferFunction)

    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(scalar_range[0], 0.0)
    opacityTransferFunction.AddPoint(scalar_range[1] * 0.3, 0.0)
    opacityTransferFunction.AddPoint(scalar_range[1] * 0.7, 0.2)
    opacityTransferFunction.AddPoint(scalar_range[1], 1.0)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 800)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(interactorStyle)

    renderWindowInteractor.Initialize()
    renderer.ResetCamera()
    renderWindow.Render()
    renderWindowInteractor.Start()


def display_volume_from_image(vtk_image):
    scalar_range = vtk_image.GetScalarRange()

    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(vtk_image)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(scalar_range[1] * 0.5, 0.5, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)
    volumeProperty.SetColor(colorTransferFunction)

    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(scalar_range[0], 0.0)
    opacityTransferFunction.AddPoint(scalar_range[1] * 0.3, 0.0)
    opacityTransferFunction.AddPoint(scalar_range[1] * 0.7, 0.2)
    opacityTransferFunction.AddPoint(scalar_range[1], 1.0)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 800)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(interactorStyle)

    renderWindowInteractor.Initialize()
    renderer.ResetCamera()
    renderWindow.Render()
    renderWindowInteractor.Start()

def display_segmentation_and_original(original_image, segmented_image):
    original_vtk = itk.vtk_image_from_image(original_image)
    segmented_vtk = itk.vtk_image_from_image(segmented_image)

    originalColorTransfer = vtk.vtkColorTransferFunction()
    originalColorTransfer.AddRGBPoint(0, 0.0, 0.0, 0.0)
    originalColorTransfer.AddRGBPoint(500, 0.5, 0.5, 0.5)
    originalColorTransfer.AddRGBPoint(1000, 1.0, 1.0, 1.0)

    originalOpacityTransfer = vtk.vtkPiecewiseFunction()
    originalOpacityTransfer.AddPoint(0, 0.0)
    originalOpacityTransfer.AddPoint(500, 0.1)
    originalOpacityTransfer.AddPoint(1000, 0.2)

    originalVolumeProperty = vtk.vtkVolumeProperty()
    originalVolumeProperty.SetColor(originalColorTransfer)
    originalVolumeProperty.SetScalarOpacity(originalOpacityTransfer)
    originalVolumeProperty.ShadeOn()

    originalVolumeMapper = vtk.vtkSmartVolumeMapper()
    originalVolumeMapper.SetInputData(original_vtk)

    originalVolume = vtk.vtkVolume()
    originalVolume.SetMapper(originalVolumeMapper)
    originalVolume.SetProperty(originalVolumeProperty)

    segmentedColorTransfer = vtk.vtkColorTransferFunction()
    segmentedColorTransfer.AddRGBPoint(0, 0.0, 0.0, 0.0)
    segmentedColorTransfer.AddRGBPoint(1, 1.0, 0.0, 0.0)

    segmentedOpacityTransfer = vtk.vtkPiecewiseFunction()
    segmentedOpacityTransfer.AddPoint(0, 0.0)
    segmentedOpacityTransfer.AddPoint(1, 0.5)

    segmentedVolumeProperty = vtk.vtkVolumeProperty()
    segmentedVolumeProperty.SetColor(segmentedColorTransfer)
    segmentedVolumeProperty.SetScalarOpacity(segmentedOpacityTransfer)
    segmentedVolumeProperty.ShadeOn()

    segmentedVolumeMapper = vtk.vtkSmartVolumeMapper()
    segmentedVolumeMapper.SetInputData(segmented_vtk)

    segmentedVolume = vtk.vtkVolume()
    segmentedVolume.SetMapper(segmentedVolumeMapper)
    segmentedVolume.SetProperty(segmentedVolumeProperty)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(originalVolume)
    renderer.AddVolume(segmentedVolume)
    renderer.SetBackground(0.1, 0.1, 0.1)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 800)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(interactorStyle)

    renderWindowInteractor.Initialize()
    renderWindow.Render()
    renderWindowInteractor.Start()

def display_volume(registered_vtk_image, color):
    def create_volume(vtk_image, color, opacity_max=0.5):
        scalar_range = vtk_image.GetScalarRange()
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetInputData(vtk_image)
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()
        
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)
        colorTransferFunction.AddRGBPoint(scalar_range[1], *color)
        volumeProperty.SetColor(colorTransferFunction)
        
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(scalar_range[0], 0.0)
        opacityTransferFunction.AddPoint(scalar_range[1] * 0.3, 0.0)
        opacityTransferFunction.AddPoint(scalar_range[1] * 0.7, opacity_max * 0.5)
        opacityTransferFunction.AddPoint(scalar_range[1], opacity_max)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)
        return volume

    registered_volume = create_volume(registered_vtk_image, color, opacity_max=0.9)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(registered_volume)
    renderer.SetBackground(0.1, 0.1, 0.1)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1200, 600)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(interactorStyle)

    renderWindowInteractor.Initialize()
    renderer.ResetCamera()
    renderWindow.Render()
    renderWindowInteractor.Start()

def display_two_volumes(original_vtk_image, registered_vtk_image, skull_vtk_image):
    def create_volume(vtk_image, color, opacity_max=0.5):
        scalar_range = vtk_image.GetScalarRange()
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetInputData(vtk_image)
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()
        
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)
        colorTransferFunction.AddRGBPoint(scalar_range[1], *color)
        volumeProperty.SetColor(colorTransferFunction)
        
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(scalar_range[0], 0.0)
        opacityTransferFunction.AddPoint(scalar_range[1] * 0.3, 0.0)
        opacityTransferFunction.AddPoint(scalar_range[1] * 0.7, opacity_max * 0.5)
        opacityTransferFunction.AddPoint(scalar_range[1], opacity_max)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)
        return volume

    original_volume = create_volume(original_vtk_image, (0.0, 0.0, 1.0), opacity_max=0.3)
    registered_volume = create_volume(registered_vtk_image, (1.0, 0.0, 0.0), opacity_max=0.3)
    skull_volume = create_volume(skull_vtk_image, (1, 1, 1), opacity_max=0.2)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(original_volume)
    renderer.AddVolume(registered_volume)
    renderer.AddVolume(skull_volume)
    renderer.SetBackground(0.1, 0.1, 0.1)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1200, 600)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(interactorStyle)

    renderWindowInteractor.Initialize()
    renderer.ResetCamera()
    renderWindow.Render()
    renderWindowInteractor.Start()
       
def tmp(filepath):
    image = read_and_extract_slice(filepath)
    x = 188
    y = 120
    print("first")
    for i in range(3):
        for j in range(3):
            print(image[x+i, y+j], end=' ')
    x2 = 175
    y2 = 98
    print("\nsecond")
    for i in range(3):
        for j in range(3):
            print(image[x2+i, y2+j], end=' ')
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray', origin='lower')
    plt.title('Image with 3x3 white square')
    plt.show()