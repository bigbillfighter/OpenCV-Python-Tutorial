import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

# HDR(high dynamic range) can make up the drawbacks of exposure time. In most images, we
# use 8-bit to represent colors, which can represent far less hues and brightness that
# human eyes can identify.
# Somestimes, the images can be overexposured when at noon and underexposured in dark when
# we just set the exposure time a static value.
# HDR uses 32-bit float values to represent channels, so it can represent far more colors.
# We can just take the same sene with different exposure times. So we get images with
# different brightness. And we just consider them generally and we can generalize an image
# with proprate bightness of each objects taken by camera.
# At last, we just represent it in 8-bit channel, and save it as a normal image.

fold_path = '../img/exposure'
filenames = ["1tl.jpg", "2tr.jpg", "3bl.jpg", "4br.jpg"]
img_list = [cv.imread(os.path.join(fold_path, i)) for i in filenames]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times)
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times)

tonemap1 = cv.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())
tonemap2 = cv.createTonemap(gamma=1.3)
res_robertson = tonemap2.process(hdr_robertson.copy())

merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

cv.imshow("debevec", res_debevec_8bit)
cv.imshow("robertson", res_robertson_8bit)
cv.imshow("mertens", res_mertens_8bit)

cal_debvec = cv.createCalibrateDebevec()
crf_debvec = cal_debvec.process(img_list, times=exposure_times) # CRF
hdr_debvec = merge_debevec.process(img_list, times=exposure_times.copy(), response=crf_debvec.copy())

cal_robertson = cv.createCalibrateRobertson()
crf_robertson = cal_robertson.process(img_list, times=exposure_times) # CRF
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy(), response=crf_robertson.copy())

tonemap1 = cv.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())
tonemap2 = cv.createTonemap(gamma=1.3)
res_robertson = tonemap2.process(hdr_robertson.copy())

res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')

cv.imshow("debevec2", res_debevec_8bit)
cv.imshow("robertson2", res_robertson_8bit)

cv.waitKey(0)
cv.destroyAllWindows()