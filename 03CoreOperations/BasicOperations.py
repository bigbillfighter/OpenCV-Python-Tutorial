import numpy as np
import cv2 as cv

img = cv.imread("../img/camera.jpg") # is a numpy array
px = img[100, 100]
print(px)

blue = img[100, 100, 0]
print(blue)

img[100, 100] = [255, 255, 255]
print(img[100, 100])

# actually, use array.item() and array.itemset() are more suitable to change a single pixel
# the above are more suitable to change a whole region
red = img.item(10, 10, 2)
print(red)

img.itemset((10, 10, 2), 100)
print(img.item(10, 10, 2))

print(img.shape)
print(img.size) # total number of pixels
print(img.dtype) # the datatype of pixel

part = img[180:240, 330:390]
img[173:233, 100:160] = part

# split and merge
b, g, r = cv.split(img)
# b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

b = np.zeros_like(b, dtype=np.uint8)
g = np.zeros_like(g, dtype=np.uint8)
# img[:, :, 0], img[:, :, 1] = 0, 0

img = cv.merge((b, g, r))

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

