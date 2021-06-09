import numpy as np
import cv2 as cv
import glob


# Load previously saved data
with np.load('IR_low_RS.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
print(mtx, dist)


def draw(img, corners, imgpts):
    corners = corners.astype(np.int64)
    imgpts = imgpts.astype(np.int64)
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 1)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 1)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 1)
    return img


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((11*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)*45
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)*10

for fname in glob.glob('imgs/gray_good/*.jpg'):
    print("hi")
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (11,8), None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        print(objp.shape)
        print(corners2.shape)
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # print(rvecs)
        R, J = cv.Rodrigues(rvecs)
        # print(R)
        # print(tvecs)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        cv.namedWindow("img", 0)
        cv.resizeWindow("img", 1080, 720)
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:6]+'.png', img)
cv.destroyAllWindows()

