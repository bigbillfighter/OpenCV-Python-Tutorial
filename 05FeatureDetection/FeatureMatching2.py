import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('../img/box.png', 0)
img2 = cv.imread('../img/box_in_scene.png', 0)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher()
# knnMatch returns k best matches, sorted by the distance.
# That means for one keypoint in the queryImg, there are 2 keypoints
# in the trainImg to match.
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3), plt.show()