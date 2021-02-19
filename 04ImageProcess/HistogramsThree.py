import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/home.jpg')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# another method using numpy
hist_np, xbins, ybins = np.histogram2d(hsv[:, :, 0].flatten(), hsv[:, :, 1].flatten(), [180, 256], [[0, 180], [0, 256]])

plt.imshow(hist, interpolation='nearest')
plt.show()