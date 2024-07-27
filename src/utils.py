import itk
import vtk
from vtk.util import numpy_support

def get_register_images(path1, path2):
    fixed_image = itk.imread(path1)
    moving_image = itk.imread(path2)

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

def vtk_to_numpy(vtk_image):
    dims = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    numpy_array = numpy_support.vtk_to_numpy(vtk_array)
    numpy_array = numpy_array.reshape(dims[2], dims[1], dims[0])
    return numpy_array
