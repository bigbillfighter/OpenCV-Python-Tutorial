import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/j.png', 0)


# erasion
# just select the minimum pixel value within the kernel window
kernel = np.ones((5, 5), np.uint8)
erosion = cv.erode(img, kernel, iterations=1)

# dilation
# select the maximum piexl value within the kernel window
dilation = cv.dilate(img, kernel, iterations=1)

# noise white
white_noised = img.copy()
noise = np.random.randint(110, size=(20, 2))
for i in range(20):
    white_noised[noise[i, 0]: (noise[i, 0]+2), noise[i, 1]:(noise[i, 1]+2)] = 255

# opening
# opening is just first use erosion and then dilation to remove white noise
opening = cv.morphologyEx(white_noised, cv.MORPH_OPEN, kernel)

# noise black
black_noised = img.copy()
noise = np.random.randint(110, size=(200, 2))
for i in range(200):
    black_noised[noise[i, 0]: (noise[i, 0]+2), noise[i, 1]:(noise[i, 1]+2)] = 0

# closing
# closing is just first use dilation and then erosion to remove black noise
closing = cv.morphologyEx(black_noised, cv.MORPH_CLOSE, kernel)

# gradient
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

# Top Hat
# output the difference between the input image and the opening image
tophat = cv.morphologyEx(white_noised, cv.MORPH_TOPHAT, kernel)

# black hat
# the difference between the input image and the closing image
blackhat = cv.morphologyEx(black_noised, cv.MORPH_BLACKHAT, kernel)


plt.subplot(4, 3, 1), plt.imshow(img, 'gray'), plt.title('raw')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 2), plt.imshow(erosion, 'gray'), plt.title('erosion')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 3), plt.imshow(dilation, 'gray'), plt.title('dilation')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 4), plt.imshow(white_noised, 'gray'), plt.title('white_noised')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 5), plt.imshow(opening, 'gray'), plt.title('opening')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 6), plt.imshow(black_noised, 'gray'), plt.title('black_noised')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 7), plt.imshow(closing, 'gray'), plt.title('closing')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 8), plt.imshow(gradient, 'gray'), plt.title('gradient')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 9), plt.imshow(tophat, 'gray'), plt.title('tophat')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 10), plt.imshow(blackhat, 'gray'), plt.title('blackhat')
plt.xticks([]), plt.yticks([])

plt.show()
