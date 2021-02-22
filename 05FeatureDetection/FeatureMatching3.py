import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# FLANN (Fast Library Approximate Nearest Neighbors)
# The algorithm runs faster than BF algorithm
# But there are several paramters we need to set

# This is the SIFT, SURF params
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

# This is ORB params
# FLANN_INDEX_LSH=6
# index_params = dict(algorithm=FLANN_INDEX_LSH,
#                     table_number = 6,
#                     key_size = 12,
#                     multi_probe_level=1)

img1 = cv.imread('../img/box.png', 0)
img2 = cv.imread('../img/box_in_scene.png', 0)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parametes
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) # or just empty the dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# make mask
matchesMask = [[0, 0] for i in range(len(matches))]
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv.DrawMatchesFlags_DEFAULT)

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3), plt.show()

