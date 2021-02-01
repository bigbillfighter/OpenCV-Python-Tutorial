import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# structure an matrix in different shape
# rectangle
print(cv.getStructuringElement(cv.MORPH_RECT, (7, 7)))
# ellipse
print(cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
# cross
print(cv.getStructuringElement(cv.MORPH_CROSS, (7, 7)))