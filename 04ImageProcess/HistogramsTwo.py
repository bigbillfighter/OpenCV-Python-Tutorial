import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/wiki.jpg', 0)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.subplot(221)
plt.imshow(img, 'gray')

plt.subplot(222)
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img2 = cdf[img]
plt.subplot(223)
plt.imshow(img2, 'gray')

plt.subplot(224)
plt.plot(cdf, color='b')
plt.hist(img2.ravel(), 256, [0, 256], color='r')
plt.xlim([0, 256])

plt.show()