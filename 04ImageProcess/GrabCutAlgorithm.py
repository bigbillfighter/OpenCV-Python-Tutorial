import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/messi5.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgbModel = np.zeros((1, 65), np.float64)
fgbModel = np.zeros((1, 65), np.float64)

rect = (50, 50, 450, 290)
cv.grabCut(img, mask, rect, bgbModel, fgbModel, 5, cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img = img*mask2[:, :, np.newaxis]
plt.imshow(img), plt.colorbar(), plt.show()