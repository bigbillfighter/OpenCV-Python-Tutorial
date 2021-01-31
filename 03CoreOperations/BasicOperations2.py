import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

BLUE = [255, 0, 0]
img1 = cv.imread("../img/camera.jpg")

# (image, top pixels, bottom pixels, left pixels, right pixels, borderType, value # if bordertype is cv.BORDER_COMSTANT)
replicate = cv.copyMakeBorder(img1, 50, 50, 50, 50, cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1, 50, 50, 50, 50, cv.BORDER_REFLECT)
reflect01 = cv.copyMakeBorder(img1, 50, 50, 50, 50, cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1, 50, 50, 50, 50, cv.BORDER_WRAP)
constant = cv.copyMakeBorder(img1, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=BLUE)
# actually, the BLUE here is RED, because in plt, the color is RGB, but in cv is BGR by default

plt.imshow(img1), plt.title('ORIGIONAL')
plt.figure()
plt.imshow(replicate), plt.title('REPLICATE')
plt.figure()
plt.imshow(reflect), plt.title('REFLECT')
plt.figure()
plt.imshow(reflect01), plt.title('REFLECT_01')
plt.figure()
plt.imshow(wrap), plt.title('WRAP')
plt.figure()
plt.imshow(constant), plt.title('CONSTANT')

plt.show()