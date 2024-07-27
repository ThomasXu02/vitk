from src.segmentation import display_segmented_tumor_2d, display_segmented_tumor_3d

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
    filepath1 = "Data/case6_gre1.nrrd"
    filepath2 = "Data/case6_gre2.nrrd"
    #display_segmented_tumor_2d(filepath1)
    display_segmented_tumor_3d(filepath1, filepath2)
