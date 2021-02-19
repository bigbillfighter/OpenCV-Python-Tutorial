import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/tsukuba.jpg', 0)

# CLAHE :
# use global equalization to avoid dark bg and bright fg
# In this case, if use global equalization, the constrast of fg will be decreased
# CLAHE uses local equalization to avoid this
# Also, to avoid noise, we set a constrast limitation to clip the value if contrast is above it

# (limitation, kernel size)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
plt.subplot(221), plt.imshow(img, 'gray'), plt.title('raw')
global_equalization = cv.equalizeHist(img)
plt.subplot(222), plt.imshow(global_equalization, 'gray'), plt.title('Global Equalization')
plt.subplot(223), plt.imshow(cl1, 'gray'), plt.title('After clipped')

plt.show()