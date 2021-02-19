import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# opencv operation is faster than numpy but not as friendly
img = cv.imread('../img/messi5.jpg', 0)
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
print("Gray image shape: {}, Spectrum shape: {}, Spectrum shifted shape:{}".format(
    img.shape, dft.shape, dft_shift.shape))

magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
print("Magnitude Spectrum: {}".format(magnitude_spectrum.shape))

rows, cols = img.shape
crow, ccol = rows//2, cols//2

mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30: crow+30, ccol-30:ccol+30]=1

fshift = dft_shift * mask
fishift = np.fft.ifftshift(fshift)
img_back = cv.idft(fishift)
img_back = cv.magnitude(img_back[:, : ,0], img_back[:, :, 1])

plt.subplot(131), plt.imshow(img, 'gray')
plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_back, 'gray')
plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back)
plt.title("Result in JET"), plt.xticks([]), plt.yticks([])

plt.show()

