import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# convex hull
img = cv.imread('../img/hand.png', 0)
_, threshold = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[1]
hull = cv.convexHull(cnt)
k = cv.isContourConvex(cnt)
print(k)
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
img_cp = img.copy()
cv.drawContours(img, [hull], 0, (0, 255, 0), 2)
cv.imshow('img', img)

x, y, w, h = cv.boundingRect(cnt)
cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 255, 0), 2)

rect = cv.minAreaRect(cnt) # the contour of min-area rectangle
box = cv.boxPoints(rect) # the four corners of the rectangle
box = np.int0(box) # box = box.astype(np.int32)
cv.drawContours(img_cp, [box], 0, (0, 0, 255), 2)

(x, y), radius = cv.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv.circle(img_cp, center, radius, (255, 0, 0), 2)

ellipse = cv.fitEllipse(cnt)
cv.ellipse(img_cp, ellipse, (0, 255, 255), 2)

rows, cols = img.shape[:2]
# (points, distType, C(if the distType needs), accuracy of radius, accuracy of angle)
# return the line r = vx * x0 + vy * y0, we return [vx, vy, x0, y0]
[vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x*vy/vx)+y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(img_cp, (cols-1, righty), (0, lefty),(255, 0, 255), 2)


cv.imshow('img2', img_cp)
cv.waitKey(0)
cv.destroyAllWindows()