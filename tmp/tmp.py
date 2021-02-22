import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../img/blox.jpg', 0)
fast = cv.FastFeatureDetector_create()
kp = fast.detect(img, None)

# the third parameter is the draw function
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

print("Threshold: {}".format(fast.getThreshold())) # t
print("NonMaxSuppression: {}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType())) # neighbor type
print("Total keypoints with nonmaxSuppression: {}".format(len(kp)))

# disable nonmaxSuppression
fast.setNonmaxSuppression(False)
kp  = fast.detect(img, None)
print("Total keypoints without nonmaxSuppression: {}".format(len(kp)))

img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

rst=np.hstack((img2, img3))
cv.imshow('result', rst)
cv.waitKey(0)
cv.destroyAllWindows()
