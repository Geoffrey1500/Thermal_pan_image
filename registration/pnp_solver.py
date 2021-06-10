import numpy as np
import cv2 as cv
import glob


# Load previously saved data
with np.load('IR_low_RS.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
print(mtx, dist)

# dist = dist*0

objp = np.array([[1.519644, 2.520145, -0.507609],
                 [1.644352, 2.351504, -0.756491],
                 [1.488210, 2.446344, -0.719156],
                 [1.521883, 2.411682, -0.760125],
                 [1.475462, 2.417302, -0.803064],
                 [1.590530, 2.449100, -0.590551],
                 [1.455, 2.481, -0.678],
                 [1.615, 2.401, -0.675]])
objp = objp.reshape((-1, 1, 3))
print(objp.shape)

corners2 = np.array([[83, 179],
                     [142, 268],
                     [81, 253],
                     [96, 268],
                     [80, 282],
                     [112, 208],
                     [67, 238],
                     [127, 238]])
corners2 = corners2.reshape((-1, 1, 2)).astype(np.float32)
print(corners2.shape)

ret, rvecs, tvecs, line = cv.solvePnPRansac(objp, corners2, mtx, dist)
imgpts, jac = cv.projectPoints(np.array([1.615, 2.401, -0.675]), rvecs, tvecs, mtx, dist)
print(imgpts)

R, J = cv.Rodrigues(rvecs)
print(R)
print(tvecs)
big_mat = np.hstack((R, tvecs))
big_mat = np.vstack((big_mat, np.array([[0, 0, 0, 1]])))
print(big_mat)

objp = np.array([[1.519644, 2.520145, -0.507609],
                 [1.644352, 2.351504, -0.756491],
                 [1.488210, 2.446344, -0.719156],
                 [1.521883, 2.411682, -0.760125],
                 [1.475462, 2.417302, -0.803064],
                 [1.590530, 2.449100, -0.590551],
                 [1.455, 2.481, -0.678],
                 [1.615, 2.401, -0.675]])

cor_after = np.hstack((objp, np.ones((objp.shape[0], 1))))
print(cor_after)

