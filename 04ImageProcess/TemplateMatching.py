import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# In opencv, template matching is applying convolution with the template and the image needed to
# test. And return a matrix each value depicts the probability that the top-left point of
# the template at the place.
img = cv.imread('../img/messi5.jpg', 0)
img2 = img.copy()
template = cv.imread('../img/messi_face.jpg', 0)
w, h = template.shape[::-1]

# All the method for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0]+w, top_left[1]+h)
    cv.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    plt.show()