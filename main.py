###
# DCM-Data-Plotter by Merlin Pritlove
###
from time import time
from pydicom import dcmread
from os import scandir
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from math import sqrt, log10, floor
import matplotlib.pyplot as plt

def main(path: str, image_count: int, slices: int):
    start_time = time()
    # defining variables for objects that are used often
    range_slices = list(range(slices))
    range_image_count = list(range(image_count))

    # creating a path-list for every file in the directory path
    file_list = sorted([f.path for f in scandir(path)])

    # creating a list with a list for every slice with every first 100 times 100 pixel out of every file
    images = np.array([np.array([np.array([i[100 * (k % 4):100 * (k % 4) + 100] for i in dcmread(file_list[j]).pixel_array[100 * (k // 4):100 * (k // 4) + 100]]) for j in range_image_count]) for k in range_slices])

    # creating binary mask based for every image
    masks_signal = [[i > 100 for i in images[j]] for j in range_slices] # creating mask
    masks_signal = [[binary_fill_holes(i) for i in j] for j in masks_signal] # filling holes
    masks_signal = [[binary_dilation(binary_erosion(i, iterations=3), iterations=3).astype(int) for i in j] for j in masks_signal] # removing halo

    # creating a mask for noise
    mask_noise = np.zeros((100, 100))
    mask_noise[:15, :15] = 1
    mask_noise[75:, 75:] = 1

    # multiplying the images with their signal mask and the noise mask
    signal_masked_images = np.array([None for _ in range_slices])
    noise_masked_images = signal_masked_images.copy()
    for k in range_slices:
        signal_masked_images[k] = np.array([np.multiply(j, masks_signal[k][i]).astype('float') for i, j in enumerate(images[k])])
        noise_masked_images[k] = np.array([np.multiply(j, mask_noise).astype('float') for i, j in enumerate(images[k])])
        for i, _ in enumerate(signal_masked_images[k]):
            signal_masked_images[k][i][signal_masked_images[k][i] == 0] = np.nan
            noise_masked_images[k][i][noise_masked_images[k][i] == 0] = np.nan

    # calculating the mean values for all the images for noise and signal
    signal_mean_values = [[np.nanmean(signal_masked_images[k][i]) for i in range_image_count] for k in range_slices]
    noise_mean_values = [[np.nanmean(noise_masked_images[k][i]) for i in range_image_count] for k in range_slices]

    # calculating the mean value for every mean value
    signal_mean = [np.nanmean(i) for i in signal_mean_values]

    # calculating the standard deviation for signal
    signal_standard_deviation = [get_standard_deviation(signal_mean_values[i], signal_mean[i]) for i in range_slices]

    # calculating SNR (SignalNoiseRatio)
    snr = [[get_snr(signal_mean_values[k][i], noise_mean_values[k][i]) for i in range_image_count] for k in range_slices]

    # plotting the data
    fig, ax = plt.subplots(slices + 1, 2, sharex='col', sharey='col')
    for k in range_slices:
        mean = signal_mean[k]
        ax[k, 0].plot(range_image_count, signal_mean_values[k], 'b.-')
        ax[k, 0].plot(range_image_count, [mean for _ in range_image_count], 'm-') # mean
        ax[k, 0].plot(range_image_count, [mean + signal_standard_deviation[k] for _ in range_image_count], 'y-') # standard deviation upper
        ax[k, 0].plot(range_image_count, [mean - signal_standard_deviation[k] for _ in range_image_count], 'y-') # standard deviation lower

    for k in range_slices:
        ax[k, 1].plot(range_image_count, snr[k], 'r.-')

    merge_count = [2, 4]
    signal_merged = [[] for i in range(len(merge_count))]
    noise_merged = [[] for i in range(len(merge_count))]
    for k in range(len(merge_count)):
        for i in range(0, floor(image_count / merge_count[k]) * merge_count[k], merge_count[k]):
            signal_merged[k].append(sum(signal_mean_values[1][i+j] for j in range(merge_count[k])) / merge_count[k])
            noise_merged[k].append(sum(signal_mean_values[1][i+j] for j in range(merge_count[k])) / merge_count[k])

    snr_merged = [[get_snr(signal_merged[j][i], noise_merged[j][i]) for i in range(floor(image_count / merge_count[j]))] for j in range(len(merge_count))]

    for j in range(len(merge_count)):
        ax[slices, 1].plot([i * merge_count[j] * 1.2 for i in range(floor(image_count / merge_count[j]))], snr_merged[j], 'g.-')

    ax[0, 0].set_xlabel('scan')
    for k in range_slices:
        ax[k, 0].set_ylabel(f'slice {k}')
        for i in (0, 1):
            ax[k, i].grid(axis='y')

    ax[slices, 1].grid(axis='y')
    ax[slices, 1].set_ylabel(f'slice 1\n{merge_count} merged')

    fig.suptitle('Plot')

    print('Runtime: ', time() - start_time)

    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

def get_standard_deviation(values: list, mean=None, count=None):
    return sqrt(sum([(i-(mean if not mean else np.mean(values))) ** 2 for i in values]) / (count if count is not None else len(values)))

def get_snr(value_signal, value_noise):
    return 20 * log10(value_signal / value_noise)

if __name__ == '__main__':
    main('012_fmre_40Hz_SS_11sl_TR1200', 155, 11)
