import cv2 as cv
import numpy as np

img1 = cv.imread('../img/football.jpg')
img2 = cv.imread('../img/icon.png')

rows, columns, channels = img2.shape
roi = img1[0:rows, 0:columns]

# create a mask of logo and create its inverse mask
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# cv.THRESH_BINARY
# if img2gray > threshold, img2gray=MAXVAL
# if img2gray <= threshold, img2gray=0
# cv.THRESH_BINARY_INV is the opposite
# if img2gray > threshold, img2gray=0
# if img2gray <= threshold, img2gray=MAXVAL
ret, mask = cv.threshold(img2gray, 150, 255, cv.THRESH_BINARY_INV)
mask_inv = cv.bitwise_not(mask)

img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv) # result = src1 and src2 if mask is not 0
img2_fg = cv.bitwise_and(img2, img2, mask=mask)

dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:columns] = dst

cv.imshow("img", img1)
cv.waitKey(0)
cv.destroyAllWindows()