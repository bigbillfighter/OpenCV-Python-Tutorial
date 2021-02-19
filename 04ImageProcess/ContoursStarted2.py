import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

a = np.zeros((500, 500, 3), dtype=np.uint8)
a[150:-150, 150:-150] = 255

a_copy = a.copy()
a_gray = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
ret, threshold = cv.threshold(a_gray, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(a, contours, 0, (0, 255, 0), 2)

# plt.imshow(a, cmap='gray'), plt.xticks([]), plt.yticks([])
# plt.show()

a_gray = cv.cvtColor(a_copy, cv.COLOR_BGR2GRAY)
ret, threshold = cv.threshold(a_gray, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cnt = contours[0]
cv.drawContours(a_copy, [cnt], 0, (0, 255, 0), 2)
# cv.circle(a_copy, ())

plt.subplot(121), plt.imshow(a)
plt.title('SIMPLE'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(a_copy)
plt.title('NONE'), plt.xticks([]), plt.yticks([])

plt.show()