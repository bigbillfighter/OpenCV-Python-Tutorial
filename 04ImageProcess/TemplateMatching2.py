import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_bgr = cv.imread('../img/mario.png')
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
template = cv.imread('../img/mario_coin.png', 0)
w, h = template.shape[::-1]

res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res>threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_bgr, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 2)

cv.imwrite('../img/res.png', img_bgr)