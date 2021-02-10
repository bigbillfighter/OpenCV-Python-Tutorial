import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

A = cv.imread('../img/apple.jpg')
B = cv.imread('../img/orange.jpg')

A_cp = cv.resize(A.copy()[3:-4, 3:-3, :], (256, 256))
print(A_cp.shape)

B_cp = cv.resize(B.copy()[1:, 2:-3, :], (256, 256))
print(B_cp.shape)


# calculate Gaussian Pyramids
G = A_cp.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)

G = B_cp.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)

# calculate Laplacian Pyramids
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i-1], GE)
    lpA.append(L)

lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv.pyrUp(gpB[i])
    L = cv.subtract(gpB[i-1], GE)
    lpB.append(L)

LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols//2], lb[:, cols//2:]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.add(ls_, LS[i])

real = np.hstack((A_cp[:, :cols//2], B_cp[:, cols//2:]))

cv.imwrite('../img/Pyramid_blending.png',ls_)
cv.imwrite('../img/Direct_blending.png', real)