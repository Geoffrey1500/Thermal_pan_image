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
images = glob.glob('imgs/good/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(img, (11,8), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (11, 8), corners2, ret)
        cv.namedWindow("img", 0)
        cv.resizeWindow("img", 1080, 720)
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print('bad')

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img, None, None)

print(mtx)
print(dist)
# print(rvecs)
# print(tvecs)

# np.savez("sony_16mm.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
