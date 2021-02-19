import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/wiki.jpg', 0)
equ = cv.equalizeHist(img)
res = np.hstack((img, equ))
cv.imwrite('../img/wiki_res.png', res)