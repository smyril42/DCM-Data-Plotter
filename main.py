from time import time
from pydicom import dcmread
from os import scandir
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from math import sqrt, log10
import matplotlib.pyplot as plt
start_time = time()

def main(path: str, image_count: int, slices: int):

    # creating a path-list for every file in the directory path
    file_list = sorted([f.path for f in scandir(path)])

    # creating a list with a list for every slice with every first 100 times 100 pixel out of every file
    images = np.array([np.array([np.array([i[100 * (k % 4):100 * (k % 4) + 100] for i in dcmread(file_list[j]).pixel_array[100 * (k // 4):100 * (k // 4) + 100]]) for j in range(image_count)]) for k in range(slices)])

    # creating binary mask based for every image
    masks_signal = [[i > 100 for i in images[j]] for j in range(slices)] # creating mask
    masks_signal = [[binary_fill_holes(i).astype(int) for i in j] for j in masks_signal] # filling holes
    masks_signal = [[binary_dilation(binary_erosion(i, iterations=3), iterations=3).astype(int) for i in j] for j in masks_signal] # removing halo

    # creating a mask for noise
    mask_noise = np.zeros((100, 100))
    mask_noise[0:20, 0:20] = 1

    # new
    signal_masked_images = np.array([None for _ in range(slices)])
    for k in range(slices):
        this_slice = np.array([np.multiply(j, masks_signal[k][i]).astype('float') for i, j in enumerate(images[k])])
        for i, j in enumerate(this_slice):
            this_slice[i][this_slice[i] == 0] = np.nan
        signal_masked_images[k] = this_slice

    # new
    noise_masked_images = np.array([None for _ in range(slices)])
    for k in range(slices):
        this_slice = np.array([np.multiply(j, mask_noise).astype('float') for i, j in enumerate(images[k])])
        for i, j in enumerate(this_slice):
            this_slice[i][this_slice[i] == 0] = np.nan
        noise_masked_images[k] = this_slice

    signal_mean_values = [[np.nanmean(signal_masked_images[k][i]) for i in range(image_count)] for k in range(slices)]

    signal_mean = [np.nanmean(i) for i in signal_mean_values]

    noise_mean_values = [[np.nanmean(noise_masked_images[k][i]) for i in range(image_count)] for k in range(slices)]

    # calculating the standard deviation for signal
    signal_standartdeviation = [standard_deviation(signal_mean_values[i], signal_mean[i]) for i in range(slices)]

    # calculating SNR (SignalNoiseRatio)
    snr = [[20 * log10(signal_mean_values[k][i] / noise_mean_values[k][i]) for i in range(image_count)] for k in range(slices)]

    # plotting the data
    range_image_count1 = range(image_count + 1)[1:] # array for the length of the y-axis
    fig, ax = plt.subplots(slices, 2)#, sharex="none")
    fig.subplots_adjust(hspace=0)
    for k in range(slices):
        ax[k, 0].plot(range_image_count1, signal_mean_values[k], "b.-")
    ax[0, 0].plot(range_image_count1, [signal_mean[0] for _ in range(image_count)], "m-") # mean
    ax[0, 0].plot(range_image_count1, [signal_mean[0] + signal_standartdeviation[0] for _ in range(image_count)], "y-") # standart deviation upper
    ax[0, 0].plot(range_image_count1, [signal_mean[0] - signal_standartdeviation[0] for _ in range(image_count)], "y-") # standart deviation lower
    for k in range(slices):
        ax[k, 1].plot(range_image_count1, snr[k], "r.-")

    ax[0, 0].set_xlabel("scan")
    #ax[1].set_xlabel("scan")
    for k in range(slices):
        ax[k, 0].set_ylabel(f"slice {k + 1}")
    for k in range(slices):
        ax[k, 0].grid(axis="y")
    fig.suptitle('Plot')
    #for k in range(slices):
    #    for i in range(2):
    #        ax[k, i].axis([0, image_count, min(signal_mean_values[k]) - min(signal_mean_values[k]) % 20, max(signal_mean_values) + (20 - max(signal_mean_values[k]) % 20)])
    figmanager = plt.get_current_fig_manager()
    figmanager.full_screen_toggle()
    plt.show()

def standard_deviation(values: list, mean=None, count=None):
    return sqrt(sum([(i-(mean if not mean else np.mean(values))) ** 2 for i in values]) / (count if count != None else len(values)))

main('012_fmre_40Hz_SS_11sl_TR1200', 155, 11)

print("Runtime: ", time() - start_time)
