import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/rubberwhale.png', 0)
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:-100, 100:-100] = 255
masked_img = cv.bitwise_and(img, img, mask=mask)

hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])

plt.show()