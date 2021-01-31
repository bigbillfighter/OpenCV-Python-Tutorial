import cv2 as cv
import numpy as np
import time

img1 = cv.imread("../img/camera.jpg")
e1 = cv.getTickCount()
for i in range(5, 49, 2):
    img1 = cv.medianBlur(img1, i)
e2 = cv.getTickCount()

t = (e2-e1)/cv.getTickFrequency()
print("passed {} seconds.".format(t))

print(e1)
print(e2)
print(cv.getTickFrequency())

# use the optimized code
print(cv.useOptimized())

start = time.time()
for _ in range(10):
    cv.medianBlur(img1, 49)
end = time.time()
print("passed {:.6f} seconds".format(end-start))

cv.setUseOptimized(False)
print(cv.useOptimized())

start = time.time()
for _ in range(10):
    cv.medianBlur(img1, 49)
end = time.time()
print("passed {:.6f} seconds".format(end-start))


cv.setUseOptimized(True)
print(cv.useOptimized())

x = 5
start = time.time()
for _ in range(1000000):
    y = x**2
end = time.time()
print("passed {:.6f} seconds".format(end-start))

start = time.time()
for _ in range(1000000):
    y = x*x
end = time.time()
print("passed {:.6f} seconds".format(end-start))

z = np.uint8([5])

start = time.time()
for _ in range(1000000):
    y = z*z
end = time.time()
print("passed {:.6f} seconds".format(end-start))

start = time.time()
for _ in range(1000000):
    y = z**2
end = time.time()
print("passed {:.6f} seconds".format(end-start))

start = time.time()
for _ in range(1000000):
    y = np.square(z)
end = time.time()
print("passed {:.6f} seconds".format(end-start))

img1 = cv.imread("../img/camera.jpg")
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY )

# normally, opencv functions work faster than numpy function
start = time.time()
for _ in range(1000):
    z = cv.countNonZero(img1)
end = time.time()
print("passed {:.6f} seconds".format(end-start))

start = time.time()
for _ in range(1000):
    z = np.count_nonzero(img1)
end = time.time()
print("passed {:.6f} seconds".format(end-start))

# 1.Avoid using loops in Python as much as possible, especially double/triple loops etc.
#   They are inherently slow.
# 2.Vectorize the algorithm/code to the maximum extent possible,
#   because Numpy and OpenCV are optimized for vector operations.
# 3.Exploit the cache coherence.
# 4.Never make copies of an array unless it is necessary. Try to use views instead.
#   Array copying is a costly operation.


