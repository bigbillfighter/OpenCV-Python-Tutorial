import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/icon.png')

# average filter (blur), 5*5 kernels
# blur = cv.blur(img, (5, 5))
blur1 = cv.boxFilter(img, -1, (5, 5))

# Gaussian filter, to remove gaussian noise
# the third means sigmaX. We don't use here, so we set it 0
blur2 = cv.GaussianBlur(img, (5, 5), 0)

# median blur, use the median of the kernel
# median blur can remove salt-pepper noise, the second must be odd
blur3 = cv.medianBlur(img, 5)

# Bilateral filter
# this uses 2 gaussian filters, one as GaussianBlur, the other to make
# sure only calculate the pixels whose values are close
# thus making sure not blur the edge
# (img, kernel size, sigmaColor, sigmaSpace)
blur4 = cv.bilateralFilter(img, 7, 75, 75)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur1), plt.title('Blurred')
plt.xticks([]), plt.yticks([])

plt.show()