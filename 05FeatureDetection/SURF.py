import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# SURF(Speeded-Up Robust Features) speeds up compared with SIFT
img = cv.imread('../img/butterfly.jpg', 0)

# Here set Hessian Threshold to 400. It's better set as 300-500.
# SURF is now none-free, so we can't use the function
surf = cv.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(img, None)
print(len(kp))
print(surf.getHessianThreshold())

surf.setHessianThreshold(500)
kp, des = surf.detectAndCompute(img, None)

print(len(kp))

img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
plt.imshow(img2), plt.show()

print(surf.getUpright())
# set all the orientation to be up
surf.setUpright(True)
kp = surf.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
plt.figure(), plt.imshow(img2), plt.show()

# change the descriptor size
print(surf.descriptorSize())
print(surf.getExtended())
surf.setExtended(True)
kp, des = surf.detectAndCompute(img, None)
print(surf.descriptorSize())
print(des.shape)
