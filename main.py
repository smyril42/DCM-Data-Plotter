from pydicom import dcmread
import matplotlib.pyplot as plt
from os import scandir
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from math import sqrt

def main(path: str, coord_slice: int):
    # calculating the coordinates for the slice
    #xslice = 100 * (coord_slice % 4)
    #yslice = 100 * (coord_slice // 4 + 1)

    # creating a path-list for every file in the directory path
    file_list = sorted([f.path for f in scandir(path)])

    # creating a list  with every first 100 times 100 pixel out of every file
    images = [np.array([i[100:200] for i in dcmread(file_list[j]).pixel_array[100:200]]) for j in range(len(file_list))]
    image = images[0]

    # creating binary mask based to picture 0
    mask = image > 100
    mask = binary_fill_holes(mask).astype(int)
    mask = binary_erosion(mask, iterations=3).astype(int)

    # just like images, but with the mask applied (np.nan instead of 0)
    masked_images = [np.multiply(i, mask).astype('float') for i in images]
    for i, j in enumerate(masked_images):
        masked_images[i][masked_images[i] == 0] = np.nan

    # the mean value from every masked image
    mean_values = [np.nanmean(masked_images[i]) for i in range(len(masked_images))]
    mean_val = np.nanmean(mean_values)

    # calculating the standard deviation
    standartabweichung = sqrt(sum([(i-mean_val) ** 2 for i in mean_values]) / len(mean_values))
    print(standartabweichung)
    print(mean_val)

main('012_fmre_40Hz_SS_11sl_TR1200', 3)
