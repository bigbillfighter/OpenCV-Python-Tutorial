import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/messi5.jpg', 0)
f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = rows//2, cols//2
print("Gray image shape: {}, Spectrum shape: {}, Spectrum shifted shape:{}".format(
    img.shape, f.shape, f_shift.shape))

# remove low frequency part
f_shift[crow-30:crow+30, ccol-30:ccol+30]=0
f_ishift = np.fft.ifftshift(f_shift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)

# we can see some waves in the inverse images. They are called "ringing effects"
# They are caused by the rectangle window we used to remove low frequency part.
# So rectangle window doesn't support Fourier Transform, better choice is Gaussian Window.

plt.subplot(131), plt.imshow(img, 'gray')
plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_back, 'gray')
plt.title("Image after HPF"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back)
plt.title("Result in JET"), plt.xticks([]), plt.yticks([])


plt.show()