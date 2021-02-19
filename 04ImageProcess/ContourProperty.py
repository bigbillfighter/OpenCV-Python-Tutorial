import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/hand2.png', 0)
img_gray = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

ret, threshold = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cnt = contours[1]
x, y, w, h = cv.boundingRect(cnt)
aspect_ratio = float(w)/h
print('Aspect Ratio: {:.4f}'.format(aspect_ratio))
cv.rectangle(img_gray, (x, y), (x+w, y+h), (255, 0, 0), 2)

area = cv.contourArea(cnt)
rect_area = w*h
extent = float(area) / rect_area
print('Extent: {:.4f}'.format(extent))

hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area) / hull_area
print('Solidity: {:.4f}'.format(solidity))

equi_diameter = np.sqrt(4*area/np.pi)
print('Equivalent Diameter: {:.2f}'.format(equi_diameter))

(x, y), (MinorAxis, MajorAxis), angle = cv.fitEllipse(cnt)
print('Fitted Ellipse Orientation: {:.2f}'.format(angle))

cv.ellipse(img_gray, (int(x), int(y)), (int(MinorAxis), int(MajorAxis)), int(angle), 0, 360, (255, 255, 0), 2)

cv.drawContours(img_gray, [cnt], 0, (0, 255, 0), 2)
cv.drawContours(img_gray, [hull], 0, (0, 0, 255), 2)
cv.imshow('img', img_gray)
cv.waitKey(0)
cv.destroyAllWindows()