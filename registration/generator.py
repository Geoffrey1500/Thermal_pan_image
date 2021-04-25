import numpy as np
import os
import pandas as pd
import cv2


filepath = 'csv_data'
i_count = 0


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


for item in os.listdir(filepath):
    # print(item, type(item), item[:-4])
    df = pd.read_csv(os.path.join(filepath, item), error_bad_lines=False, sep='\t', header=None).drop([0], axis=0)

    df = df[0].str.split(',', expand=True, ).drop([0, 385], axis=1).astype('float64')
    data_array = df.values

    # temperature_range = np.max(data_array) - np.min(data_array)
    lowest_temper, highest_temper = 23, 30
    temperature_range = highest_temper - lowest_temper
    temperature_normalized = np.floor((data_array - lowest_temper)/temperature_range * 255).astype(np.uint8)
    # temperature_normalized = np.floor(((data_array - np.min(data_array)) * 255) / temperature_range).astype(np.uint8)
    img_color_hist = histequ(np.floor((data_array - np.min(data_array))/(np.max(data_array)-np.min(data_array)) * 255).astype(np.int64))
    # img_color_hist = cv2.equalizeHist(temperature_normalized)

    temperature_normalized_2 = np.floor((data_array - np.min(data_array))/(np.max(data_array)-np.min(data_array)) * 255).astype(np.uint8)

    img_colored_hist = cv2.applyColorMap(img_color_hist, cv2.COLORMAP_JET)
    img_colored = cv2.applyColorMap(temperature_normalized, cv2.COLORMAP_JET)

    img_scaled_with_color = cv2.applyColorMap(temperature_normalized_2, cv2.COLORMAP_JET)

    cv2.imwrite('imgs/colored/' + item[:-4] + '.jpg', img_colored)
    cv2.imwrite('imgs/gray/' + item[:-4] + '.jpg', temperature_normalized)
    # cv2.imwrite('thermal_test_img/normalized/' + item[:-4] + '.jpg', img_colored_hist)
    # cv2.imwrite('imgs/scaled_gray/' + item[:-4] + '.jpg', temperature_normalized_2)
    # cv2.imwrite('imgs/scaled_with_color/' + item[:-4] + '.jpg', img_scaled_with_color)

    i_count += 1