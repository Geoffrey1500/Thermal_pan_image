import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import cv2


def csv_to_array(csv_set_path, img_item):
    # df_for_thermal = pd.read_csv(os.path.join(csv_set_path, img_item), error_bad_lines=False, sep='\t', header=None).drop([0, 385], axis=0)
    # print(item, type(item), item[:-4])
    df = pd.read_csv(os.path.join(csv_set_path, img_item), error_bad_lines=False, sep='\t', header=None).drop([0], axis=0)

    # for data from flir tools
    df = df[0].str.split(',', expand=True, ).drop([0, 385], axis=1).astype('float64')

    # df_for_thermal = df_for_thermal[0].str.split(',', expand=True, ).drop([0], axis=1).astype('float64')
    data_in_array = df.values

    return data_in_array


def localEqualHist(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(image)

    return dst


data_array_set = np.zeros((288, 384))

filepath = 'csv_data'

for item in os.listdir(filepath):
    data_array = csv_to_array(filepath, item)
    # print(data_array_set.shape)
    data_array_set = np.dstack((data_array_set, data_array))

data_array_set = data_array_set[:, :, 1::]
print(data_array_set.shape, data_array_set.flatten().shape)
data_for_plot = data_array_set.flatten()

data_for_plot = pd.Series(data_for_plot)
u = data_for_plot.mean()
std = data_for_plot.std()
data_c = data_for_plot[np.abs(data_for_plot - u) < 3*std]

print(data_for_plot.skew(), data_for_plot.kurt(), "计算偏度与峰度")
print(min(data_c), max(data_c), "3倍标准差方法")

# lowest_temper, highest_temper = max(np.min(data_array_set), 23), min(np.max(data_array_set), 35)
lowest_temper, highest_temper = min(data_c)+data_for_plot.skew(), max(data_c)+data_for_plot.skew()
print(lowest_temper, highest_temper)

temperature_range = highest_temper - lowest_temper
print(temperature_range)

k = (1-0)/temperature_range


for i in range(data_array_set.shape[-1]):
    data_array = data_array_set[:, :, i]

    data_for_plot = data_array.flatten()
    data_for_plot = pd.Series(data_for_plot)
    u = data_for_plot.mean()
    std = data_for_plot.std()
    data_c = data_for_plot[np.abs(data_for_plot - u) < 3 * std]

    print(data_for_plot.skew(), data_for_plot.kurt(), "计算偏度与峰度")
    print(min(data_c), max(data_c), "3倍标准差方法")

    # lowest_temper, highest_temper = max(np.min(data_array_set), 23), min(np.max(data_array_set), 35)
    lowest_temper, highest_temper = 23, 30
    print(lowest_temper, highest_temper)

    temperature_range = highest_temper - lowest_temper
    print(temperature_range)

    data_array = (data_array - lowest_temper)/temperature_range * 255
    data_array[data_array < 0] = 0
    data_array[data_array > 255] = 255
    # data_array = abs(data_array - 255)
    # data_array[data_array > 230] = 255
    # data_array[data_array < 230] = 0

    temperature_normalized = data_array.astype(np.uint8)

    # img_enhanced = localEqualHist(data_array)
    # temperature_normalized = (0 + k * (data_array - lowest_temper))
    # temperature_normalized = np.floor(data_array * 255).astype(np.uint8)
    # img_enhanced = localEqualHist(temperature_normalized)

    # img_enhanced = cv2.adaptiveThreshold(img_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    # img_enhanced = cv2.adaptiveThreshold(img_enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
    # img_enhanced = temperature_normalized
    # img_scaled_with_color = cv2.applyColorMap(temperature_normalized, cv2.COLORMAP_JET)
    # img_enhanced = cv2.applyColorMap(img_enhanced, cv2.COLORMAP_JET)
    cv2.imwrite('imgs/gray_2/' + str(i) + '.jpg', temperature_normalized)
