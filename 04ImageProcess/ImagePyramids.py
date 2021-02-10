import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# image pyramids mean that put the images with same content but different resolution together,
# when put high resolutions on bottom and low top, the shap looks like a pyramid

# Gaussian Pyramid
# once done of Gaussian pyramid ,you will make the figure become (W/2, H/2) or (2*W, 2*H)
# when do pyrDown, it just remove some rows and columns of the raw image.
img = cv.imread('../img/messi.jpeg')
lower_reso = cv.pyrDown(img)
higher_reso = cv.pyrUp(lower_reso)
# the higher_reso is not equal to img, because when you did pyrDown, some information lost.

print(img.shape, 'origional')
print(lower_reso.shape, 'after lower reso')
print(higher_reso.shape, 'after high reso')

plt.imshow(img)
plt.title('raw'), plt.xticks([]), plt.yticks([])

plt.figure()
plt.imshow(lower_reso)
plt.title('lower_reso'), plt.xticks([]), plt.yticks([])

plt.figure()
plt.imshow(higher_reso)
plt.title('higher_reso'), plt.xticks([]), plt.yticks([])

plt.show()

