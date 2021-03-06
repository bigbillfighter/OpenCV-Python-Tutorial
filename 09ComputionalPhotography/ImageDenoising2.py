import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture("../img/vtest.avi")

# a list of first 5 frames
img = [cap.read()[1][:200, :200] for _ in range(5)]
grayed = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in img]
gray = [np.float64(i) for i in grayed]
noise = np.random.randn(*gray[1].shape)*10 # gray[1].shape=(576, 768), *gray[1].shape=576, 768
noisy = [i+noise for i in gray]
noisy = [np.uint8(np.clip(i, 0, 255)) for i in noisy]

# the second argument means which frame we need to denoise
# the third argument means how many nearby images we use to help denoise
dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 6, 11, 35)

result = np.vstack((grayed[2], noisy[2], dst))
plt.imshow(result, cmap='gray')
plt.show()
plt.xticks([]), plt.yticks([])

cap.release()