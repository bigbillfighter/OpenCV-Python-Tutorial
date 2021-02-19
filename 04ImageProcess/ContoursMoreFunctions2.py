import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/star.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, threshold = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
C = contours[6]
A = contours[7]
B = contours[8]
ret1 = cv.matchShapes(A, A, 1, 0.0)
ret2 = cv.matchShapes(A, B, 1, 0.0) 
ret3 = cv.matchShapes(A, C, 1, 0.0)
print('Match A and itself is {:.3f}'.format(ret1))
print('Match A and B is {:.3f}'.format(ret2))
print('Match A and C is {:.3f}'.format(ret3))
# cv.drawContours(img, contours, 8, (0, 255, 0), 2)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()