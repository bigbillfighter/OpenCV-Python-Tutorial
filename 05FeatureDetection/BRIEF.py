import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# In SIFT, each vector is 128-dimension and In SURF is still 64-dimension. The element in
# the vector is a float number which is memory consuming and hard to match. So a better method
# is to find a binary string to describe the keypoints, then we can use logical operation
# like XOR to match, which is very fast in modern CPU with SSE instructions.

# So in BRIEF(Binary Robust Independent Element Features), we find a set of n location pairs
# in a unique way which is detailed illustrated
# in the paper. For each pair, if I(p) < I(q), we get a 1, other wise 0. So the result is a
# binary string to describe the keypoint.

img = cv.imread('../img/blox.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# star detector
star = cv.xfeatures2d.StarDetector_create()

# BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with star
kp = star.detect(img_gray, None)
kp, des = brief.compute(img_gray, kp)

print(brief.descriptorSize())
print(des.shape)

img2 = cv.drawKeypoints(img, kp, None, (0, 255, 0))
cv.imshow('img', img2)
cv.waitKey(0)
cv.destroyAllWindows()