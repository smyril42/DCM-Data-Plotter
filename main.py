from pydicom import dcmread
import matplotlib.pyplot as plt
from os import scandir
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion
from math import sqrt, log10

def main(path: str, coord_slice: int):
    # calculating the coordinates for the slice
    #xslice = 100 * (coord_slice % 4)
    #yslice = 100 * (coord_slice // 4 + 1)

    # creating a path-list for every file in the directory path
    file_list = sorted([f.path for f in scandir(path)])
    image_count = len(file_list)

    # creating a list  with every first 100 times 100 pixel out of every file
    images = [np.array([i[100:200] for i in dcmread(file_list[j]).pixel_array[100:200]]) for j in range(image_count)]

    # creating binary mask based to picture 0
    mask_signal = images[0] > 100
    mask_signal = binary_fill_holes(mask_signal).astype(int)
    mask_signal = binary_erosion(mask_signal, iterations=3).astype(int)
    # same with noise
    mask_noise = np.zeros((100, 100))
    mask_noise[0:20, 0:20] = 1

    # just like images, but with the mask_signal applied (and np.nan instead of 0)
    signal_masked_images = [np.multiply(i, mask_signal).astype('float') for i in images]
    for i, j in enumerate(signal_masked_images):
        signal_masked_images[i][signal_masked_images[i] == 0] = np.nan
    # same for mask_noise
    noise_masked_images = [np.multiply(i, mask_noise).astype('float') for i in images]
    for i, j in enumerate(noise_masked_images):
        noise_masked_images[i][noise_masked_images[i] == 0] = np.nan

    # the mean value from every masked image
    signal_mean_values = [np.nanmean(signal_masked_images[i]) for i in range(image_count)]
    signal_mean = np.nanmean(signal_mean_values)
    # same for noise
    noise_mean_values = [np.nanmean(noise_masked_images[i]) for i in range(image_count)]

    # calculating the standard deviation for signal
    signal_standartdeviation = sqrt(sum([(i-signal_mean) ** 2 for i in signal_mean_values]) / image_count)

    # calculating SNR (SignalNoiseRatio)
    snr = [20 * log10(signal_mean_values[i] / noise_mean_values[i]) for i in range(image_count)]

    # plotting the data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    range_image_count = range(image_count + 1)[1:]
    ax1.plot(range_image_count, signal_mean_values, "b.-") # graph
    ax1.plot(range_image_count, [signal_mean for _ in range(image_count)], "m-") # mean
    ax1.plot(range_image_count, [signal_mean + signal_standartdeviation for _ in range(image_count)], "y-") # standart deviation upper
    ax1.plot(range_image_count, [signal_mean - signal_standartdeviation for _ in range(image_count)], "y-") # standart deviation lower
    ax2.plot(range_image_count, snr, "r.-")

    ax1.set_xlabel("scan")
    ax2.set_xlabel("scan")

    ax1.set_ylabel("mean value")
    ax2.set_ylabel("SNR")


    #ax.grid(axis="y")
    fig.suptitle('Plot')
    #plt.axis([0, image_count, min(signal_mean_values) - min(signal_mean_values) % 20, max(signal_mean_values) + (20 - max(signal_mean_values) % 20)])
    #plt.xlabel('scan')
    #plt.ylabel('mean value')
    plt.show()

main('012_fmre_40Hz_SS_11sl_TR1200', 3)
