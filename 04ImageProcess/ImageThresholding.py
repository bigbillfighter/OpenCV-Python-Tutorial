import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# For every pixel, the same threshold value is applied. If the pixel value is smaller
# than the threshold, it is set to 0, otherwise it is set to a maximum value.
# The function cv.threshold is used to apply the thresholding.
# The first argument is the source image, which should be a grayscale image.
# The second argument is the threshold value which is used to classify the pixel values.
# The third argument is the maximum value which is assigned to pixel values exceeding the threshold.

# img = np.zeros((256*2, 256*2), dtype=np.uint8)
# for i in range(256):
#     img[:, 2*i: 2*i+2] = i
# cv.imwrite("../img/gradient.png", img)
# exit()

img = cv.imread("../img/gradient.png", 0)
_, thresh1 = cv .threshold(img, 127, 255, cv. THRESH_BINARY)
_, thresh2 = cv .threshold(img, 127, 255, cv. THRESH_BINARY_INV)
_, thresh3 = cv .threshold(img, 127, 255, cv. THRESH_TRUNC)
_, thresh4 = cv .threshold(img, 127, 255, cv. THRESH_TOZERO)
_, thresh5 = cv .threshold(img, 127, 255, cv. THRESH_TOZERO_INV)

titles = ['Origional', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()