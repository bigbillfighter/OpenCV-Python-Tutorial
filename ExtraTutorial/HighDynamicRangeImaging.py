# Most images use 8-bit to represent one channel, whose range is 0-255
# But humans' eyes can identify much more colors and brightness.
# So we need high dynamic range image techniques. One is HDR, which
# use 32-bit float number to represent per channel
# one common way to get HDR images is to use the same image with difference
# level exposure.

from __future__ import print_function, division
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import argparse

def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv.imread(os.path.join(path, tokens[0])))
        times.append(1/float(tokens[1]))

    return images, np.asarray(times, dtype=np.float32)

parser = argparse.ArgumentParser(description='Code for High Dynamic Range Image tutorial.')
parser.add_argument('--input', default="./data", type=str, help='Path to the directory that contains images and exposure times.')
args = parser.parse_args()

if not args.input:
    parser.print_help()
    exit(0)

images, times = loadExposureSeq(args.input)

# images = [cv.imread('../img/cathedral.png')]
# times = np.asarray([4.], dtype=np.float32)

calibrate = cv.createCalibrateDebevec()
response = calibrate.process(images, times)

merge_debevec = cv.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)

tonemap = cv.createTonemap(2.2)
ldr = tonemap.process(hdr)

merge_mertens = cv.createMergeMertens()
fusion = merge_mertens.process(images)

cv.imwrite('writeImage/fusion.png', fusion*255)
cv.imwrite('writeImage/ldr.png', ldr*255)
cv.imwrite('writeImage/hdr.hdr', hdr)

# cv.waitKey(0)
# cv.destroyAllWindows()