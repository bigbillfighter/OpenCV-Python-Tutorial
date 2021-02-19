import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time

img = cv.imread('../img/messi5.jpg', 0)
rows, cols = img.shape
print("Rows: {}, Cols: {}".format(rows, cols))

# the size of array influence the speed of fft, best size is the order of 2, and the products of
# several 2, 3 and 5 are also good choices
# So better method is use the function below to dialect the size, and fill the border with 0
nrows, ncols = cv.getOptimalDFTSize(rows), cv.getOptimalDFTSize(cols)
print("Optimal Rows: {}, Optimal Cols: {}".format(nrows, ncols))

nimg = np.zeros((nrows, ncols), np.uint8)
nimg[:rows, :cols]=img

# the second to the fifth is (top border, down border, left border, right border)
# the sixth is the fill method, fill the border with constant value 0.
# nimg = cv.copyMakeBorder(img, 0, nrows-rows, 0, ncols-cols, cv.BORDER_CONSTANT, value=0)

start = time.time()
for _ in range(10):
    fft1 = np.fft.fft2(img)
end = time.time()
print("Time(Numpy, raw size): {:.6f}s".format(end-start))

start = time.time()
for _ in range(10):
    fft2 = np.fft.fft2(nimg)
end = time.time()
print("Time(Numpy, optimized size): {:.6f}s".format(end-start))

start = time.time()
for _ in range(10):
    dft1 = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
end = time.time()
print("Time(OpenCV, raw size): {:.6f}s".format(end-start))

start = time.time()
for _ in range(10):
    dft2 = cv.dft(np.float32(nimg), flags=cv.DFT_COMPLEX_OUTPUT)
end = time.time()
print("Time(OpenCV, optimized size): {:.6f}s".format(end-start))
