import pydicom
import matplotlib.pyplot as plt
import os

def main(path: str):
    # creating a path-list for every file in the directory path
    file_list = sorted([f.path for f in os.scandir(path)])

    #creating a list  with every first 100 times 100 pixel out of every file
    images = [[i[:100] for i in pydicom.dcmread(file_list[j]).pixel_array[:100]] for j in range(len(file_list))]

    image = images[0]
    plt.imshow(image)
    plt.colorbar()
    plt.show()


main(path='012_fmre_40Hz_SS_11sl_TR1200')
