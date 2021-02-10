import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = np.zeros((500, 500), dtype=np.uint8)
img[120:500-120, 150:500-150] = 255

sobel8u = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)

sobel64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
abs_sobel64f = np.absolute(sobel64f)
sobel8u2 = np.uint8(abs_sobel64f)

plt.subplot(1, 3, 1), plt.imshow(img, 'gray'), plt.title('Raw')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(sobel8u, 'gray'), plt.title('Sobel 8U')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(sobel8u2, 'gray'), plt.title('Sobel 64F to 8U')
plt.xticks([]), plt.yticks([])

plt.show()