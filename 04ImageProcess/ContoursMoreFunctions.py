import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_gray = cv.imread('../img/contour.jpg', 0)
img = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
_, threshold = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cnt = contours[1]

hull = cv.convexHull(cnt, returnPoints=False)
defects = cv.convexityDefects(cnt, hull)

cv.drawContours(img, [cnt], 0, (0, 255, 0), 2)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s, 0])
    end = tuple(cnt[e, 0])
    far = tuple(cnt[f, 0])
    cv.line(img, start, end, [0, 0, 255], 2)
    cv.circle(img, far, 5, [255, 0, 0], -1)

# Test the distance of one point to a contour, positive inside, negative outside
# the second is the point coordinate, the third is if calculate distance
# if False, just judge the position, return -1, 0, or 1
dist = cv.pointPolygonTest(cnt, (50, 50), True)
print('The distance of (50, 50) from contour: {:.3f}'.format(dist))

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()


