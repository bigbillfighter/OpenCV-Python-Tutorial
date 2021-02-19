import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/rubberwhale.png', 0)
hist = cv.calcHist([img], [0], None, [256], [0, 256])
x = np.arange(256)
plt.subplot(221), plt.plot(x, hist, 'r-')

hist, bins = np.histogram(img.ravel(), 256, [0, 256])
plt.subplot(222), plt.plot(bins.astype(np.int32)[:-1], hist, 'b-')

plt.subplot(223), plt.hist(img.ravel(), 256, [0, 256])
# bins is from 0 to 256, which is total 257 numbers

img_color = cv.imread('../img/rubberwhale.png')
color = ['b', 'g', 'r']
plt.subplot(224)
for i, col in enumerate(color):
    histr = cv.calcHist([img_color], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])


plt.show()
