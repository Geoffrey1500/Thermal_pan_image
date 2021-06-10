import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((11*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)*45
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('imgs/gray_good/*.jpg')
images_2 = glob.glob('imgs/gray_good_copy/*.jpg')
i = 0
for fname in images:
    img = cv.imread(fname)
    str_list = list(fname)
    # print(str_list)
    str_list.insert(14, "_copy")
    a_b = ''.join(str_list)
    img2 = cv.imread(a_b)
    print(a_b)
    print(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(img, (11,8), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img2, (11, 8), corners2, ret)
        # cv.namedWindow("img", 0)
        # cv.resizeWindow("img", 1080, 720)
        cv.imwrite('imgs/gray_for_paper/' + str(i) + '.jpg', img2)
        cv.imshow('img', img2)
        cv.waitKey(0)
        cv.destroyAllWindows()
        i += 1
        # print('good', fname)
    else:
        print('bad', fname)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx)
print(dist)
# print(rvecs)
# print(tvecs)

# np.savez("IR_low_RS.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
