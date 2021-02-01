import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# there more than 250 color-space conversion methods
flags = [i for i in dir(cv) if i.startswith("COLOR_")]
print(len(flags))

img = cv.imread("../img/camera.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
# Different software use different scales. So if you are comparing OpenCV values with them,
# you need to normalize these ranges.

# how to find the threshold of the blue color in hsv
blue = np.uint8([[[255, 0, 0]]])
hsv_blue = cv.cvtColor(blue, cv.COLOR_BGR2HSV)
print(hsv_blue)
# and we can set the threshold of blue as between [h-10, 100, 100] and [h+10, 255, 255]

# track the blue objects
cap = cv.VideoCapture(0)
while 1:
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # threshold the hsv image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    res = cv.bitwise_and(frame, frame, mask=mask)
    cv. imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(1) & 0xFF
    if k==27:
        break

cap.release()
cv.destroyAllWindows()

