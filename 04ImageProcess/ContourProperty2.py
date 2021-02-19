import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/hand2.png', 0)
img_gray = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
_, threshold = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[1]

mask = np.zeros(img.shape, np.uint8)
cv.drawContours(mask, [cnt], 0, 255, -1)
pixelpoints = np.transpose(np.nonzero(mask)) # the result is (row, column)
# pixelpoints = cv.findNonZero(mask) # the result is (x, y)
print("Pixel points shape: {}".format(pixelpoints.shape))

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(img, mask=mask)
# the minimum value, maximum value and their locations, after doing bitwise and with the mask
print('Maximum value of the object: {}, Minumum value of the object: {}'.format(max_val, min_val))

mean_val = cv.mean(img, mask=mask)

print("Mean value of the object: {:.4f}".format(mean_val[0]))

cv.drawContours(img_gray, [cnt], 0, (0, 255, 0), 2)
cv.imshow('img', img_gray)
cv.waitKey(0)
cv.destroyAllWindows()