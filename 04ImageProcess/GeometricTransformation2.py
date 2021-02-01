import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("../img/football.jpg", 0) # 0 means grayscale
rows, cols = img.shape
M = np.float32([[1, 0, -100], [0, 1, 50]]) # 100 means the translation of width, 50 means height
# dst(x, y) = (M11x+M12y+M13, M21x+M22y+M23)
translation = cv.warpAffine(img, M, (cols, rows)) # the third is (width, height)==(cols, rows)

# the first is the width and height of center, second is the degree in anti-clockwise,
# third is scale factor
M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 45, 1)
rotation = cv.warpAffine(img, M, (cols, rows))

# the lines which were parallel in original image are still parallel in the affine image
# so we can just input 3 points in original and transformed images respectively,
# and output the transformation matrix
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv.getAffineTransform(pts1, pts2)
Affined = cv.warpAffine(img, M, (cols, rows))
plt.subplot(121), plt.imshow(img,'gray'), plt.title('Input')
plt.subplot(122), plt.imshow(Affined,'gray'), plt.title('Affined')
plt.show()

cv.imshow('translation', translation)
cv.imshow('rotation', rotation)
cv.waitKey(0)
cv.destroyAllWindows()
