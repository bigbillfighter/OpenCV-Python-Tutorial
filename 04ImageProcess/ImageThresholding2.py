import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("../img/sudoku.png", 0)

# the second is the kernel size, which must be odd
# the result piexl is the median in 5*5 window
img = cv.medianBlur(img, 5)

ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# (origional image, max value, adaptive method, threshold method, kernel size, C)
# if adaptive method is cv.ADAPTIVE_THRESH_MEAN_C, that means use the mean of
# the neighbours of (x, y) in kernel*kernel window minus C as the threshold
# in this case, use the mean of 11*11-1 pixels and minus the result with 2
# use the threshold do cv.THRESH_BINARY threshold, the threshold is used on
# just this one pixel.
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 11, 2)
# if use cv.ADAPTIVE_THRESH_GAUSSIAN_C, the threshold is the gaussian mean of the
# kernel*kernel neighbor pixel value minus C
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv.THRESH_BINARY, 11, 2)
titles = ['ORIGINAL', 'GLOBAL(v=127)', 'ADAPTIVE MEAN', 'ADAPTIVE GAUSSIAN']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()