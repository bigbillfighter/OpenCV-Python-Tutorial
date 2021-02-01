import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/icon.png')
# average kernel
kernel = np.ones((5, 5), np.float32)/25
# use cv.filter2D do convolution
dst = cv.filter2D(img, -1, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])

plt.show()