import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("../img/home_raw.jpg")
Z = img.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

for i in range(2, 10, 2):
    K = i

    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    plt.subplot(2, 2, int(i/2))
    plt.imshow(cv.cvtColor(res2, cv.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.title("K = {}".format(K))

plt.show()