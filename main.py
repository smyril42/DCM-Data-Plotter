from pydicom import dcmread
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes

def main(path: str, coord_slice: int):
    # calculating the coordinates for the slice
    xslice = 100 * (coord_slice % 4)
    yslice = 100 * (coord_slice // 4 + 1)

    # creating a path-list for every file in the directory path
    file_list = sorted([f.path for f in os.scandir(path)])

    # creating a list  with every first 100 times 100 pixel out of every file
    images = [np.array([i[yslice - 100:yslice] for i in dcmread(file_list[j]).pixel_array[xslice - 100:xslice]]) for j in range(len(file_list))]
    image = images[0]

    # creating binary mask based to picture 0
    mask = image > 100
    mask = binary_fill_holes(mask).astype(int)
    mask = binary_erosion(mask, iterations=3).astype(int)

    # applying mask to image
    masked_image = np.multiply(image, mask)

    # finding the average value of the image
    masked_image = masked_image.astype('float')
    masked_image[masked_image == 0] = np.nan
    mean_val = np.nanmean(masked_image)
    print(mean_val)

main('012_fmre_40Hz_SS_11sl_TR1200', 2)
