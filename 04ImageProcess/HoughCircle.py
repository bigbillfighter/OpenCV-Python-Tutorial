import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/opencv-logo-white.png', 0)
img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# second: detection method
# third: dp, the inverse scale coefficient. So if dp is 2, the output will be (rows/2, cols/2)
# fourth: the minimum distance of circle centers
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint32(np.around(circles))
for i in circles[0, :]:
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imshow('detected circles', cimg)
cv.waitKey(0)
cv.destroyAllWindows()
