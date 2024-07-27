from src.segmentation import display_segmented_tumor_2d, display_segmented_tumor_3d

if __name__ == "__main__":
    filepath1 = "Data/case6_gre1.nrrd"
    filepath2 = "Data/case6_gre2.nrrd"
    #display_segmented_tumor_2d(filepath1)
    display_segmented_tumor_3d(filepath1, filepath2)
