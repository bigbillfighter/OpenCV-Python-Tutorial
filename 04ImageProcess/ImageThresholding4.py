'''
The Otsu function to select threshold can be implementated as below.
Src: https://docs.opencv.org/4.5.1/d7/d4d/tutorial_py_thresholding.html
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/noise.png', 0)
blur = cv.GaussianBlur(img, (5, 5), 0)

# find normalized_histogram, and its cumulative distribution function
hist = cv.calcHist([blur], [0], None, [256], [0, 256])
hist_norm = hist.ravel() / hist.sum()
Q = hist_norm.cumsum()

bins = np.arange(256)
fn_min = np.inf
thresh = -1

for i in range(1, 256):
    p1, p2 = np.hsplit(hist_norm, [i]) # probabilities
    q1, q2 = Q[i], Q[255] - Q[i] # cumulate sum of classes
    if q1 < 1e-6 or q2 <1e-6:
        continue
    b1, b2 = np.hsplit(bins, [i]) # weights

    # finding means and variance
    m1, m2 = np.sum(b1*p1)/q1, np.sum(b2*p2)/q2
    v1, v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2

    # calculate the minimization function
    fn = v1*q1+v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i

# find otsu's threshold value with OpenCV functin
ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
print("{} {}".format(thresh, ret))


