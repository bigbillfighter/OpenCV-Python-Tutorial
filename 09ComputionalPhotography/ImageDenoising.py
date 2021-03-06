import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../img/die.png')
# the third is h and the forth is h-color. For colored images, they are the same.
# In gray-scale images, we should only use h. The higher h is, the efficiency of denoising
# is better but the image will be more blurred.
# the fifth is template window size, which is the size of sampling to patch
# the sixth is search window size, which is the stride of finding template window size.
dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

plt.subplot(121), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.subplot(122), plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
plt.show()