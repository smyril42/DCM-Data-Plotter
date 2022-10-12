from pydicom import dcmread
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes

def main(path: str):
    # creating a path-list for every file in the directory path
    file_list = sorted([f.path for f in os.scandir(path)])

    #creating a list  with every first 100 times 100 pixel out of every file
    images = [np.array([i[0:100] for i in dcmread(file_list[j]).pixel_array[0:100]]) for j in range(len(file_list))]
    image = images[0]

    print(image)
    mask = image > 100
    mask = binary_fill_holes(mask).astype(int)

    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=1)
    mask = np.invert(mask)

    masked_image = np.ma.masked_array(image, mask=mask)

    plt.imshow(mask)
    plt.colorbar()
    plt.show()


main(path='012_fmre_40Hz_SS_11sl_TR1200')



