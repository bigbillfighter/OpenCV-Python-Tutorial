import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("../img/pizza.jpg")
rows, cols, chs = img.shape

# they are all width and height
pts1 = np.float32([[76, 65], [345, 52], [28, 327], [389, 290]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv. warpPerspective(img, M, (300, 300))

# draw the four corners that of the part that to be Transformed
# we can use it to make a picture regular from distorted
for i in range(len(pts1)):
    cv.circle(img, (pts1[i, 0], pts1[i, 1]), 5, [0, 0, 255], -1)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)

plt.subplot(121), plt.imshow(img), plt.title("input")
plt.subplot(122), plt.imshow(dst), plt.title('output')
plt.show()
