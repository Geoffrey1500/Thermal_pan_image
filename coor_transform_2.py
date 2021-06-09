import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

np.set_printoptions(suppress=True)


# transformation function
def seven_params_func(params, x):
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
    reviteye = T.T + np.dot(R, x) * S
    return reviteye


# residual function
def residual(params, func, x, y):
    args = (params, x)
    func_x = func(*args)
    return np.squeeze(np.asarray(func_x.T.reshape((1, -1)) - y.T))


# compute 7 parameters by using non-Linear least-squares minimization
def seven_params_calc(slam_coords, revit_coords):
    seven_params_init = [0, 0, 0, 0, 0, 0, 0.65]
    lb = [-np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -np.inf]
    ub = [np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf]

    solution = least_squares(fun=residual,
                             x0=seven_params_init,
                             args=(seven_params_func, slam_coords, revit_coords),
                             ftol=1e-10, xtol=1e-08, gtol=1e-08,
                             max_nfev=1000000,
                             bounds=(lb, ub))
    return solution.x


# transform slam coordinates to revit eye by using 7 parameters
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

    reviteye = T + np.dot(R, slam_camera_position) * S

    return reviteye


def seven_params_transform_2(params, slam_camera_position):
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


# compute slam rotation to revit rotation by using 7 parameters
def seven_params_rotation(params, orientation):
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

    revitorientation = np.dot(R, orientation)

    return revitorientation


# transform camera pos of slam to revit coordinates and orientations
def transform_slam_to_revit(seven_params, slam_camera_pose):
    identify_matrix = np.array([[0, -1, 0],
                                [0, 0, 1],
                                [-1, 0, 0]])

    # slam_camera_pose: Tcw, Cw = -Tcw * Oc, -Tcw has 2 parts: Rcw.T, -Rcw.T*tcw
    # tcw
    slam_translation_vector = slam_camera_pose[0:3, 3]
    # Rcw
    slam_rotation_matrix = slam_camera_pose[0:3, 0:3]

    slam_camera_position = np.dot(-slam_rotation_matrix.T, slam_translation_vector)
    print("World coordinate of camera in SLAM: {}".format(slam_camera_position))

    eye = seven_params_transform(seven_params, slam_camera_position)
    up = seven_params_rotation(seven_params, np.dot(slam_rotation_matrix.T, identify_matrix[0,].T)).T
    forward = seven_params_rotation(seven_params, np.dot(slam_rotation_matrix.T, identify_matrix[1,].T)).T
    other = seven_params_rotation(seven_params, np.dot(slam_rotation_matrix.T, identify_matrix[2,].T)).T

    print("<eye>{}, {}, {}</eye>".format(eye[0, :3][0], eye[0, :3][1], eye[0, :3][2]))
    print("<up>{}, {}, {}</up>".format(up[0], up[1], up[2]))
    print("<forward>{}, {}, {}</forward>".format(forward[0], forward[1], forward[2]))

    print("other={}".format(repr(other)))


if __name__ == "__main__":
    # seven parameters
    # seven_params = [-1.147717074415504e+04,
    #             -1.595581113535468e+03,
    #             1.125902935313602e+03,
    #             -1.618746608287555,
    #             0.022131667557669,
    #             -1.700854855250953,
    #             1.039798561303257e+03]

    x1 = np.array([[0.600464, 1.761392, -0.705922]])
    x2 = np.array([[0.809235, 1.721604, -0.789856]])
    x3 = np.array([[1.519644, 2.520145, -0.507609]])
    x4 = np.array([[1.644352, 2.351504, -0.756491]])
    x5 = np.array([[1.488210, 2.446344, -0.719156]])

    # X: 3X5
    X = np.hstack([x1.T, x2.T, x3.T, x4.T, x5.T])

    coor_2d = np.array([[2023, 1631],
                        [2151, 1670],
                        [2687, 1432],
                        [2747, 1524],
                        [2685, 1504]])

    coor_2d_2 = (coor_2d + np.array([[-5590/2, -2795/2]]))*np.array([[1, -1]])
    scale_x, scale_y = np.pi/5590, (np.pi/2)/2795
    theta_a = coor_2d_2[:, 0] * scale_x
    alpha_a = coor_2d_2[:, 1] * scale_y
    for_r_copy = np.vstack([x1, x2, x3, x4, x5])
    r = np.sqrt(np.sum(for_r_copy**2, axis=1))

    x_new = r*np.cos(theta_a)*np.cos(alpha_a).T
    y_new = r*np.cos(theta_a)*np.sin(alpha_a).T
    z_new = r*np.sin(theta_a).T

    Y_stacked = np.vstack((x_new, y_new, z_new)).T.reshape((-1, 1))


    seven_params = seven_params_calc(X, Y_stacked)
    print(seven_params)
    print(seven_params[3:6]/np.pi*180)
    test_point = np.array([0.718384, 1.701509, -0.876935]).T
    location_new = seven_params_transform(seven_params, test_point)
    print(location_new)

    cor_test = np.vstack((x_new, y_new, z_new)).T
    print(cor_test.shape, "bbb", cor_test.T)
    location_new = seven_params_transform_2(seven_params, X)
    print(location_new, "aaa")

    print(cor_test.T - location_new)
