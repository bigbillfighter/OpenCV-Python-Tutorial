import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

x = np.random.randint(25, 100, 25)
y = np.random.randint(175, 255, 25)
z = np.hstack((x, y))
z = z.reshape((50, 1))
z = np.float32(z)
# plt.hist(z, 256, [0, 256]), plt.show()

criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 10, 1.0)
flags = cv.KMEANS_RANDOM_CENTERS
# The fifth argument means the times doing the algorithm.
# And this function will return the best result with best compactness.
compactness, labels, centers = cv.kmeans(z, 2, None, criteria, 10, flags)
A = z[labels==0]
B = z[labels==1]

plt.hist(A, 256, [0, 256], color='r')
plt.hist(B, 256, [0, 256], color='b')
plt.hist(centers, 32, [0, 256], color='y')
plt.show()