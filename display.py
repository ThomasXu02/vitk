import os
import itk
import vtk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from vtk.util import numpy_support


def read_and_extract_slice(filepath):
    image = itk.imread(filepath)
    np_image = itk.GetArrayFromImage(image)
    z_index = np_image.shape[0] // 2
    central_slice = np_image[z_index, :, :]
    return central_slice


def display_images2d(filepath1, filepath2):
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
    colorTransferFunction.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)  # Black for low values
    colorTransferFunction.AddRGBPoint(scalar_range[1] * 0.5, 0.5, 0.0, 0.0)  # Dark red for mid values
    colorTransferFunction.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)  # Bright red for high values (likely tumor)
    volumeProperty.SetColor(colorTransferFunction)

    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(scalar_range[0], 0.0)  # Transparent for low values
    opacityTransferFunction.AddPoint(scalar_range[1] * 0.3, 0.0)  # Still transparent for lower mid-range
    opacityTransferFunction.AddPoint(scalar_range[1] * 0.7, 0.2)  # Start to appear
    opacityTransferFunction.AddPoint(scalar_range[1], 1.0)  # Opaque for high values (likely tumor)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 800)  # Larger window

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


def display_two_volumes(original_vtk_image, registered_vtk_image):
    def create_volume(vtk_image, color):
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
        opacityTransferFunction.AddPoint(scalar_range[1] * 0.7, 0.5)
        opacityTransferFunction.AddPoint(scalar_range[1], 1.0)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)

        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        return volume

    original_volume = create_volume(original_vtk_image, (0.0, 0.0, 1.0))  # Blue
    registered_volume = create_volume(registered_vtk_image, (1.0, 0.0, 0.0))  # Red

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(original_volume)
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


def visualize_slice(itk_image, slice_index, orientation):
    array = itk.array_from_image(itk_image)
    if orientation == 'sagittal':
        slice_data = array[slice_index, :, :]
    elif orientation == 'axial':
        slice_data = array[:, slice_index, :]
    elif orientation == 'coronal':
        slice_data = array[:, :, slice_index]
    else:
        raise ValueError("Invalid orientation: choose 'axial', 'sagittal', or 'coronal'")

    plt.imshow(slice_data, cmap='gray')
    plt.title(f'{orientation.capitalize()} Slice {slice_index}')
    plt.axis('on')
    plt.show()


def visualize_segmented_slice(segmented_slice, slice_index, orientation):
    plt.imshow(segmented_slice, cmap='gray')
    plt.title(f'Segmented {orientation.capitalize()} Slice {slice_index}')
    plt.axis('on')
    plt.show()


def plot_slice_histogram(itk_image, slice_index, orientation):
    array = itk.array_from_image(itk_image)
    if orientation == 'sagittal':
        slice_data = array[slice_index, :, :]
    elif orientation == 'axial':
        slice_data = array[:, slice_index, :]
    elif orientation == 'coronal':
        slice_data = array[:, :, slice_index]
    else:
        raise ValueError("Invalid orientation: choose 'axial', 'sagittal', or 'coronal'")

    plt.hist(slice_data.ravel(), bins=256, range=(0, 256), fc='k', ec='k')
    plt.title(f'Slice {slice_index}')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.show()
