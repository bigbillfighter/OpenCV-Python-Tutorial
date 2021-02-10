import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# in opencv, findContours is like finding white objects from background.
# So just make the object white and background white

im = cv.imread('../img/football.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(imgray, 127, 255, 0)

# (src, contour retrieval mode(检索模式), contour approximation method)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# contours is the list of all contours, each is a numpy array(x, y) which
# denotes the position of one boundary point

# (src, contours(should be python list), the index of contours(if want to draw all, press -1),
# color, thickness)
cv.drawContours(im, contours, -1, (0, 255, 0), 2)
# cv.drawContours(im, contours, 3, (0, 255, 0), 2)
# cnt = contours[4]
# cv.drawContours(im, [cnt], 0, (0, 255, 0), 2)
cv.imshow('Contours', im)
cv.waitKey(0)
cv.destroyAllWindows()