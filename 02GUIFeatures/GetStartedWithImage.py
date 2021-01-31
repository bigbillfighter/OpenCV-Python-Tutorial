import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("img/camera.jpg"), cv.IMREAD_UNCHANGED)
# the second parameter has multi values
# the most useful are cv.IMREAD_COLOR, cv.IMREAD_UMCHANGED, cv.GRAYSCALE

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)
k = cv.waitKey(0)
# the value means time waiting for a key to be pressed, measured in milliseconds
# 0 means to wait forever
# k is the index of the key.

if k == ord('s'):
    cv.imwrite("img/camera.png", img)