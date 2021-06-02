import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import cv2


def histequ(gray, nlevels=256):
    # Compute histogram
    histogram = np.bincount(gray.flatten(), minlength=nlevels)
    # print ("histogram: ", histogram)

    # Mapping function
    uniform_hist = (nlevels - 1) * (np.cumsum(histogram)/(gray.size * 1.0))
    uniform_hist = uniform_hist.astype('uint8')
    # print ("uniform hist: ", uniform_hist)

    # Set the intensity of the pixel in the raw gray to its corresponding new intensity
    height, width = gray.shape
    uniform_gray = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            uniform_gray[i,j] = uniform_hist[gray[i,j]]

    return uniform_gray


def csv_to_array(csv_set_path, img_item):
    df_for_thermal = pd.read_csv(os.path.join(csv_set_path, img_item), error_bad_lines=False, sep='\t', header=None).drop([0], axis=0)

    df_for_thermal = df_for_thermal[0].str.split(',', expand=True, ).drop([0], axis=1).astype('float64')
    data_in_array = df_for_thermal.values

    return data_in_array


def localEqualHist(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(image)

    return dst


data_array_set = np.zeros((288, 384))

filepath = 'batch_data/csv'

for item in os.listdir(filepath):
    data_array = csv_to_array(filepath, item)
    # print(data_array_set.shape)
    data_array_set = np.dstack((data_array_set, data_array))

data_array_set = data_array_set[:, :, 1::]
print(data_array_set.shape)

lowest_temper, highest_temper = max(np.min(data_array_set), 15), min(np.max(data_array_set), 40)
print(lowest_temper, highest_temper)

temperature_range = highest_temper - lowest_temper
print(temperature_range)

k = (1-0)/temperature_range


for i in range(data_array_set.shape[-1]):
    data_array = data_array_set[:, :, i]

    lowest_temper, highest_temper = np.min(data_array), np.max(data_array)

    temperature_range = highest_temper - lowest_temper

    temperature_normalized = ((data_array - lowest_temper)/temperature_range * 255).astype(np.uint8)
    # img_enhanced = localEqualHist(data_array)
    # temperature_normalized = (0 + k * (data_array - lowest_temper))
    # temperature_normalized = np.floor(data_array * 255).astype(np.uint8)
    # img_enhanced = localEqualHist(temperature_normalized)
    img_scaled_with_color = cv2.applyColorMap(temperature_normalized, cv2.COLORMAP_JET)
    # img_colored = cv2.applyColorMap(temperature_normalized, cv2.COLORMAP_JET)
    cv2.imwrite('batch_data/img3/' + str(i) + '.jpg', img_scaled_with_color)
