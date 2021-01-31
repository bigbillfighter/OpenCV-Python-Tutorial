import cv2 as cv
import numpy as np

# list all the attributes in cv which start with 'EVENT'
events = [i for i in dir(cv) if 'EVENT' in i]

# the format that the callback function is
# if a event happens, the system will transfer the arguments
def draw_circle(event, x, y, flags, param):
    if event == cv. EVENT_LBUTTONDBLCLK:
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)

img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
# set the function as the mouse callback function and the window that the function works
cv.setMouseCallback('image', draw_circle)

while(1):
    cv.imshow('image', img)
    # use 0xFF to make and operation to make the result is 8 bit
    if cv.waitKey(20) & 0xFF == 27: # 27 is the 'esc' bottom
        break
cv.destroyAllWindows()