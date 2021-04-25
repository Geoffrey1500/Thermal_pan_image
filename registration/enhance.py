import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt


# filename = 'imgs/good/10.jpg'
images = glob.glob('imgs/good/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ret, th3 = cv.threshold(gray,50,255,cv.THRESH_BINARY)
    th3 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv.THRESH_BINARY, 3, 2)

    # gray = np.float32(gray)
    # dst = cv.cornerHarris(gray,2,29,0.23)
    # #result is dilated for marking the corners, not important
    # dst = cv.dilate(dst, None)
    # # Threshold for an optimal value, it may vary depending on the image.
    # img[dst>0.01*dst.max()]=[0,0,255]
    # cv.namedWindow("dst", 0)
    # cv.resizeWindow("dst", 1080, 720)
    # cv.imshow('dst',img)
    # if cv.waitKey(300) & 0xff == 27:
    #     cv.destroyAllWindows()

    corners = cv.goodFeaturesToTrack(th3, 88, 0.08, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv.circle(th3, (x, y), 3, 255, -1)
    # plt.imshow(th3), plt.show()
    #
    # ret2, corners = cv.findChessboardCorners(th3, (8, 11), None)
    # print(ret2)
    #
    cv.namedWindow("img", 0)
    cv.resizeWindow("img", 1080, 720)
    cv.imshow('img', th3)
    cv.waitKey(500)
    cv.destroyAllWindows()
