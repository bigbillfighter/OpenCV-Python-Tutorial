import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Now we try to match the keypoints in queryImg and trainImg, so we have to
# calculte the distance between different keypoints
# In SIFT, SURF, it is suitable use L2-norm or L1-norm to calculate
# In OBR, BRIEF, BRISK, we shold use Hamming Distance, because their descriptors are binary.

# Brute-Force Matching algorithm is very simple.
# For every keypoints in queryImg, we just try to traverse every keypoint in the trainimg, and
# return the closed one.

img1 = cv.imread('../img/box.png', 0) # queryImage(pattern)
img2 = cv.imread('../img/box_in_scene.png', 0) # trainImage

orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# The first argument is the distance calculation method.
# It should be cv.NORM_L2 or cv.NORM_L1 when using SIFT, SURF
# It should be cv.NORM_HAMMING when using ORB, BRIEF, BRISK.
# And when using ORB and WTA_K==3 or 4, we should use cv.NORM_HAMMING2
# "crossCheck" means if using cross check. If it is true, we will make sure that when p1 in
# queryImg is closed to p2 in trainImg and vice versa.
# By default, it is false.
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
# The function return a list of DMatch objects
# There are some attributes in DMatch.
# 1. DMatch.distance - distance between descriptors. The lower, the better it is.
# 2. DMatch.trainIdx - index of the descriptor in train descriptors
# 3. DMatch.queryIdx - index of the descriptor in query descriptors
# 4. DMatch.imgIdx - Index of the train image
print(len(matches))

matches = sorted(matches, key=lambda x: x.distance)
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3), plt.show()