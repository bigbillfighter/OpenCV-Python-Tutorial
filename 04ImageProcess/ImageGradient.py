import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# low-pass filters remove low frequency signals, while high-pass remove high frequency signals
# low-pass means the high frequency signals won't pass, such as these parts change very quickly
# so the edges, corners will be blurred
# high-pass means to preserve high frequency signals and remove loss frequency
# so only the edges and corners will be preserved
# Low-pass to blur, High-pass to sharp

img = cv.imread('../img/sudoku.png', 0)

### don't understand these two functions very well ###
# use laplacian function to calculate the value in the kernel box
laplacian = cv.Laplacian(img, cv.CV_64F)
# the definition of soble operator is different version
# commonly, the definition follows the Pascal Triangle
# so 3*3 detecting horizontal edge is
# [-1 -2 -1
#   0  0  0
#   1  2  1]
# 5*5 is
# [-1 -4 -6 -4 -1
#  -2 -8 -12 -8 -2
#   0  0  0  0  0
#   2  8  12  8  2
#   1  4  6  4  1]

soblex = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobley = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('raw'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(laplacian, cmap='gray')
plt.title('laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(soblex, cmap='gray')
plt.title('sobel x'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(sobley, cmap='gray')
plt.title('sobel y'), plt.xticks([]), plt.yticks([])

plt.show()