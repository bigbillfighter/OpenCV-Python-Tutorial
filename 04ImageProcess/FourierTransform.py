import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/messi5.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

magnitude_spectrum_notshifted = 20*np.log(np.abs(f))

plt.subplot(221), plt.imshow(img, 'gray')
plt.title('Input image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(magnitude_spectrum_notshifted, cmap='gray')
plt.title('Magnitude Spectrum(not shifted)'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()