import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/water_coins.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# kernel here means we use a 3x3 window to do erode and dilate
# we can get other shapes of window by getStructuringElement()
kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv.dilate(opening, kernel, iterations=3)

# get the distance from one pixel to the nearest value-0 pixel
# so pixels with value as 0 will be 0, pixels with positive value will be the distance
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# mark pixels with 0-value as 0 and positive value with random positive integers
ret, markers = cv.connectedComponents(sure_fg)
# markers = dist_transform.copy().astype(np.int32)

# make unknown regions become valleys, so we can let the water grow up
markers = markers+1 # so the bg will be 1
markers[unknown==255] = 0 # so the unknown region will be 0

markers = cv.watershed(img, markers)
img[markers == -1] = [0, 0, 255]
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()