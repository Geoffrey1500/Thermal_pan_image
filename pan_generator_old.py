import cv2 as cv
import numpy as np
import open3d as o3d
from math import cos, sin, ceil, floor
import glob
from PIL import Image
import time


def seven_params_transform(params, slam_camera_position):
    T = np.array([[params[0], params[1], params[2]]])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(params[3]), -np.sin(params[3])],
                   [0, np.sin(params[3]), np.cos(params[3])]])

    Ry = np.array([[np.cos(params[4]), 0, np.sin(params[4])],
                   [0, 1, 0],
                   [-np.sin(params[4]), 0, np.cos(params[4])]])

    Rz = np.array([[np.cos(params[5]), -np.sin(params[5]), 0],
                   [np.sin(params[5]), np.cos(params[5]), 0],
                   [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))

    S = params[6]

    print(R.shape)

    reviteye = T.T + np.dot(R, slam_camera_position) * S

    return reviteye


def rotation_matrix(angle_set):
    """ Constructs 4D homogeneous rotation matrix given the rotation angles
        (degrees) around the x, y and z-axis. Rotation is implemented in
        XYZ order.
    :param rx: Rotation around the x-axis in degrees.
    :param ry: Rotation around the y-axis in degrees.
    :param rz: Rotation around the z-axis in degrees.
    :return:   4x4 matrix rotation matrix.
    """
    # Convert from degrees to radians.
    rx, ry, rz = angle_set
    rx = np.pi * rx / 180
    ry = np.pi * ry / 180
    rz = np.pi * rz / 180

    # Pre-compute sine and cosine of angles.
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])

    # Set up euler rotations.
    Rx = np.array([[1, 0,  0],
                   [0, cx, -sx],
                   [0, sx, cx]])

    Ry = np.array([[cy,  0, sy],
                   [0,   1, 0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz, cz,  0],
                   [0,  0,   1]])

    return Rz.dot(Ry.dot(Rx))


def point_to_panorama(cor_data, color_data, ang_res):
    pixel_x, pixel_y = 2*np.pi/ang_res, np.pi/ang_res
    print(pixel_x, pixel_y)
    r = np.zeros((int(pixel_x) + 1, int(pixel_y) + 1))
    # g = np.zeros((int(pixel_x) + 1, int(pixel_y) + 1))
    # b = np.zeros((int(pixel_x) + 1, int(pixel_y) + 1))

    r_ = np.sqrt(np.sum(cor_data ** 2, axis=1))
    lon_ = np.arctan2(cor_data[:, 1], cor_data[:, 0])
    lat_ = np.arcsin(cor_data[:, 2] / r_)

    x_new_ = np.rint(lon_ / ang_res).astype(np.int32) + int(np.pi/ang_res)
    y_new_ = -np.rint(lat_ / ang_res).astype(np.int32) + int(np.pi*0.5/ang_res)

    r[x_new_, y_new_] = color_data[:, 0]*255
    # g[x_new_, y_new_] = color_data[:, 1]*255
    # b[x_new_, y_new_] = color_data[:, 2]*255

    # base_img = np.dstack((np.flipud(b.T), np.flipud(g.T), np.flipud(r.T)))
    # base_img = np.dstack((np.rot90(b, -1), np.rot90(g, -1)*0, np.rot90(r, -1)*0))
    # _range = np.max(r_) - np.min(r_)
    # depth = (r_ - np.min(r_)) / _range
    # print(depth.shape, "深度形状")
    # r[x_new_, y_new_] = depth * 255
    base_img = np.rot90(r, -1)
    # base_img = np.dstack((np.rot90(b, -1), np.rot90(g, -1), np.rot90(r, -1)))

    kernel = np.ones((2, 2), dtype=np.uint8)
    base_img = cv.morphologyEx(base_img, cv.MORPH_CLOSE, kernel, iterations=1)

    return base_img.astype(np.uint8)


def ordinationConvert(x1, y1, z1, args):
    x2 = args[0] + (1 + args[6]) * (x1 + args[5] * y1 - args[4] * z1)
    y2 = args[1] + (1 + args[6]) * (-args[5] * x1 + y1 + args[3] * z1)
    z2 = args[2] + (1 + args[6]) * (args[4] * x1 - args[3] * y1 + z1)
    return np.vstack((x2, y2, z2)).T


angular_resolution = 0.036/180*np.pi
start = time.time()
pcd = o3d.io.read_point_cloud("0.018_intensity.pcd")
cor_set = np.asarray(pcd.points)

# Args = np.array([0, 0, 0, 0, 0, 0, 0.60214721])
Args = np.array([-0.67,  0, 0.3, 0, 0, 0, 1.10642548])
# Args = np.array([-1.38602001, -1.09937816,  0.11424165,  1.27321186, -0.38213514, -0.3901401, 1.10642548])

cor_set_after = seven_params_transform(Args, cor_set.T).T
# cor_set_after = ordinationConvert(cor_set[:, 0], cor_set[:, 1], cor_set[:, 2], Args)
# cor_set_new = cor_set + np.array([500, 0, 0])
color_set = np.asarray(pcd.colors)
end = time.time()
print(end-start, "文件读取时间")
print(cor_set.shape, len(cor_set))
print(color_set.shape, len(color_set))

print("hI")

start = time.time()
pan_img = point_to_panorama(cor_set_after, color_set, angular_resolution)
end = time.time()
print(end-start, "全景图转换时间")
cv.imwrite('laser_pan_0.018_intensity.png', pan_img)
cv.namedWindow("img", 0)
cv.resizeWindow("img", 1080, 540)
# cv.resizeWindow("img", 270, 540)
# pixel_x, pixel_y = 2*np.pi/angular_resolution, np.pi/angular_resolution
#
# ptStart = (0, int(pixel_y/2))
# ptEnd = (int(pixel_x), int(pixel_y/2))
# point_color = (0, 0, 255) # BGR
# thickness = 10
# lineType = 4
# cv.line(pan_img, ptStart, ptEnd, point_color, thickness, lineType)
#
# ptStart = (int(pixel_x/2), 0)
# ptEnd = (int(pixel_x/2), int(pixel_y))
# point_color = (0, 0, 255) # BGR
# thickness = 10
# lineType = 4
# cv.line(pan_img, ptStart, ptEnd, point_color, thickness, lineType)

cv.imshow('img', pan_img)
cv.waitKey(0)
cv.destroyAllWindows()
#
# im = Image.fromarray(pan_img) # numpy 转 image类
# im.show()
