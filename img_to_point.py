import cv2 as cv
import numpy as np
# import open3d as o3d
from math import cos, sin, ceil, floor
import glob
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def rotation_orla(angle_set):
    rx, ry, rz = angle_set
    rx = np.pi * rx / 180
    ry = np.pi * ry / 180
    rz = np.pi * rz / 180

    r11, r12, r13 = cos(rz) * cos(ry), cos(rz) * sin(ry) * sin(rx) - sin(rz) * cos(rx), cos(rz) * sin(ry) * cos(rx) + sin(
        rz) * sin(rx)
    r21, r22, r23 = sin(rz) * cos(ry), sin(rz) * sin(ry) * sin(rx) + cos(rz) * cos(rx), sin(rz) * sin(ry) * cos(rx) - cos(
        rz) * sin(rx)
    r31, r32, r33 = -sin(ry), cos(ry) * sin(rx), cos(ry) * cos(rx)

    rotation_final = np.array([[r11, r12, r13],
                               [r21, r22, r23],
                               [r31, r32, r33]])

    return rotation_final


def img_to_point(img_cor_set, cam_para, rotation_angle_set, ang_res):
    '''
    :param img_path: 图像路径
    :param camera_path: 相机内参路径，为npz文件
    :return: 返回值为以赤道上照片的第一张读取的图片的中心点所确立的坐标, 以及色彩信息的集合
    '''
    # img_cor_ = np.dstack((x_, y_[np.lexsort(-y_.T)], np.ones_like(gray_))).reshape(-1, 3)
    img_cor_ = np.dot(cam_para, img_cor_set.T).T
    img_cor_[:, [0, 1, 2]] = img_cor_[:, [2, 0, 1]]

    y_adj = (np.max(img_cor_[:, 1]) + np.min(img_cor_[:, 1])) / 2
    z_adj = (np.max(img_cor_[:, 2]) + np.min(img_cor_[:, 2])) / 2
    img_cor_ = img_cor_ + np.array([[0, -y_adj, -z_adj]])

    img_cor_ = np.dot(rotation_orla(rotation_angle_set), img_cor_.T).T

    return img_cor_


mtx = np.array([[886.380806550638, 0, 207.48733413556],
                [0, 889.810660752285, 149.098984879267],
                [0, 0, 1]])

mtx_inv = np.linalg.inv(mtx)
angular_resolution = np.max([np.arctan(mtx[0, -1]/mtx[0, 0])/mtx[0, -1], np.arctan(mtx[1, -1]/mtx[1, 1])/mtx[1, -1]])*1.0

thermal_cor_set_1 = np.array([[128, 237, 1],
                              [53, 143, 1],
                              [171, 180, 1]])

thermal_cor_set_2 = np.array([[84, 165, 1],
                              [141, 252, 1],
                              [82, 253, 1]])

thermal_cor_set_3 = np.array([[108, 41, 1],
                              [149, 112, 1]])

cor_set1 = img_to_point(thermal_cor_set_1, mtx_inv, [0, 15, -40], angular_resolution)
print(cor_set1)

cor_set2 = img_to_point(thermal_cor_set_2, mtx_inv, [0, 0, 0], angular_resolution)
print(cor_set2)

cor_set3 = img_to_point(thermal_cor_set_3, mtx_inv, [0, 15, 180], angular_resolution)
print(cor_set3)

print(np.vstack((cor_set1, cor_set2, cor_set3)))

other = np.array([
    [1.534557, 2.441133, -0.675784],
    [1.568079, 2.406529, -0.717129],
    [1.554582, 2.485515, -0.547432],
    [1.441575, 2.451687, -0.763547],
    [1.525272, 2.534856, -0.465872],
    [1.627288, 2.430431, -0.590892]
])

new_cor = np.array([
    [1.00000000e+00,  5.64091637e-04,  3.14673685e-02],
    [1.,          0.01748684,  0.04832489],
    [1.,         0.00169227, -0.01685752],
    [1.,         -0.03440959,  0.04944872],
    [1.,         -0.01410229, -0.04944872],
    [1.,          0.03440959, -0.00112383]
])


print(0.128/np.sqrt((new_cor[0, 1] - new_cor[3, 1])**2 + (new_cor[0, 2] - new_cor[3, 2])**2), "1-4")
print(0.063/np.sqrt((new_cor[0, 1] - new_cor[1, 1])**2 + (new_cor[0, 2] - new_cor[1, 2])**2), "1-2")
print(0.126/np.sqrt((new_cor[0, 1] - new_cor[-1, 1])**2 + (new_cor[0, 2] - new_cor[-1, 2])**2), "1-6")
print(0.137/np.sqrt((new_cor[0, 1] - new_cor[2, 1])**2 + (new_cor[0, 2] - new_cor[2, 2])**2), "1-3")
print(0.230/np.sqrt((new_cor[0, 1] - new_cor[4, 1])**2 + (new_cor[0, 2] - new_cor[4, 2])**2), "1-5")

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(new_cor[:, 0], new_cor[:, 1], new_cor[:, 2])
# fig2 = plt.figure()
# ax = Axes3D(fig2)
# ax.scatter(thermal_cor_set[:, 0], thermal_cor_set[:, 1], thermal_cor_set[:, 2])
# # plt.scatter(thermal_cor_set[:, 0], thermal_cor_set[:, 1])
# plt.show()
