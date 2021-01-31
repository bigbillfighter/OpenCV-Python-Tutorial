import cv2 as cv
import numpy as np

x = np.uint8([250])
y = np.uint8([10])

print(cv.add(x, y)) # this one output max(min(calue, 255), 0)
print(x+y) # thie one output (x+y) % 256

img1 = cv.imread('../img/food.jpg')
img2 = cv.imread('../img/football.jpg')
dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)
# dst = 0.7*img1 + 0.3*img2 + 0

cv.imshow("dst", dst)
cv.waitKey(0)
cv.destroyAllWindows()