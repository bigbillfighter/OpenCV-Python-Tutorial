import numpy as np
import cv2 as cv

filename = '../img/chessboard2.jpeg'
img = cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst = cv.dilate(dst, None)
ret, dst = cv.threshold(dst, 0.01*dst.max(), 255, 0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

res = np.hstack((centroids, corners))
res = np.int0(res)

for i in range(res.shape[0]):

    img[res[i, 1]-3:res[i, 1]+1, res[i, 0]-3:res[i, 0]+1] = [0, 0, 255]
    img[res[i, 3]-2:res[i, 3]+2, res[i, 2]-2:res[i, 2]+2] = [0, 255, 0]

cv.imshow('subpixel', img)
cv.waitKey(0)
cv.destroyAllWindows()