import cv2 as cv
import numpy as np

img = cv.imread("../img/food.jpg")

# if the second is None, means it resizes based on scale factors fx and fy,
# fx is the factor of horizontal axis -- width(columns)
# fy is the factor of vertical axis -- height(rows)
# if the second is not None, the size is defined by the second param, which is a tuple (width, height)
# the last is the interpolation method
res = cv.resize(img, None, fx=0.5, fy=1, interpolation=cv.INTER_CUBIC)
# res = cv.resize(img, (640, 720))


cv.imshow("res", res)
cv.waitKey(0)
cv.destroyAllWindows()
