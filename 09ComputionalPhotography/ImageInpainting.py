import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../img/messi_2.jpg')
mask = cv.imread('../img/mask2.png', 0)

# cv.INPAINT_TELEA use the neighbours with weights to fix,
# cv.INPAINT_NS use fluid dynamic methods
# the third argument means the size of fix circle
dst1 = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
dst2 = cv.inpaint(img, mask, 3, cv.INPAINT_NS)

mask_3d = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
result_1 = np.hstack((img, mask_3d))
result_2 = np.hstack((dst1, dst2))
result = np.vstack((result_1, result_2))

cv.imshow('img', result)
cv.waitKey(0)
cv.destroyAllWindows()