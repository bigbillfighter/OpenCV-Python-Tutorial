import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# img = np.random.randint(low=0, high=137, size=(256*2, 256*2), dtype=np.uint8)
# center = np.random.randint(low=117, high=256, size=(256, 256), dtype=np.uint8)
#
# img[256//2: 256//2+256, 256//2: 256//2+256] = center
# cv.imwrite("../img/noise.png", img)

img = cv.imread('../img/noise.png', 0)

# Consider an image with only two distinct image values (bimodal image),
# where the histogram would only consist of two peaks. A good threshold
# would be in the middle of those two values. Similarly, Otsu's method determines
# an optimal global threshold value from the image histogram.

# ret is the threshold
# global thershold
ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Otsu's threshold
ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# Otsu's threshold after Gaussion filtering
blur = cv.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original', 'Hist', 'Global',
          'Original', 'Hist', 'Otsu',
          'Gaussion blurred', 'Hist', 'Otus']

for i in range(3):
   plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
   plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

   plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
   plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

   plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
   plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()

print(ret1, ret2, ret3)