import itk
import numpy as np
import matplotlib.pyplot as plt
from src.display import read_and_extract_slice, display_two_volumes
from src.utils import get_register_images, itk_to_vtk_image, vtk_to_numpy


def seg_region_growing_2d(image):
    itk_image = itk.GetImageFromArray(image.astype(np.float32))

    ImageType = itk.Image[itk.F, 2]
    seg = itk.ConnectedThresholdImageFilter[ImageType, ImageType].New()
    seg.SetInput(itk_image)

    p1 = (120, 188)
    p2 = (98, 175)

    low = 420
    high = 930

    for i in range(-4, 4):
        for j in range(-4, 4):
            a = itk_image.GetPixel((p1[0] + i, p1[1] + j))
            b = itk_image.GetPixel((p2[0] + i, p2[1] + j))
            if low < a < high:
                seg.AddSeed((p1[0] + i, p1[1] + j))
            if low < b < high:
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

    segmented_np = np.zeros_like(np_image)
    for z in range(60, 100):
        segmented_slice = seg_region_growing_2d(np_image[z])
        segmented_np[z] = segmented_slice

    segmented_image = itk.GetImageFromArray(segmented_np)
    segmented_image.SetOrigin(itk_image.GetOrigin())
    segmented_image.SetSpacing(itk_image.GetSpacing())
    segmented_image.SetDirection(itk_image.GetDirection())

    return segmented_image


def display_segmented_tumor_2d(filepath1):
    temp = read_and_extract_slice(filepath1)
    segmented = seg_region_growing_2d(temp)

    plt.figure(figsize=(12, 6))
    plt.imshow(segmented, cmap='gray', origin='lower')
    plt.title('Segmented Slice 85')
    plt.axis('off')
    plt.show()


def display_segmented_tumor_3d(filepath1, filepath2):
    original_image = itk.imread(filepath1)
    original_image_np = itk.GetArrayFromImage(original_image)

    for i in range(original_image_np.shape[0]):
        original_image_np[i] = np.flipud(original_image_np[i])

    original_image_flipped = itk.GetImageFromArray(original_image_np)
    seg1 = seg_region_growing_3d(original_image_flipped)
    seg1 = itk_to_vtk_image(seg1)

    # Register
    reg = get_register_images(filepath1, filepath2)
    reg_vtk = itk_to_vtk_image(reg)
    reg_np = vtk_to_numpy(reg_vtk)

    for i in range(reg_np.shape[0]):
        reg_np[i] = np.flipud(reg_np[i])

    reg_flipped = itk.GetImageFromArray(reg_np)
    seg2 = seg_region_growing_3d(reg_flipped)
    seg2 = itk_to_vtk_image(seg2)

    # Display the two volumes
    display_two_volumes(seg1, seg2, itk_to_vtk_image(original_image_flipped))
