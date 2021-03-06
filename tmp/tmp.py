import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

SZ = 20
bin_n = 16
affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR

def deskew(img):
    '''
    This function can make skew image be straight still.
    :param img: input image
    :return: deskewed image
    '''
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img

def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    # calculate the magnitude and angle, they are the same shape as the image

    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = (bins[:10, :10], bins[10:, 10:], bins[:10, 10:], bins[10:, 10:])
    mag_cells = (mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:])

    # mag_cells well be the weight
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists) # hist(64, )
    return hist

img = cv.imread("../img/digits.png", 0)
if img is None:
    raise Exception("Image not found!")

cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

train_cells = [i[:50] for i in cells]
test_cells = [i[50:] for i in cells]

deskewed = [list(map(deskew, row)) for row in train_cells] # deskewed(50, 50, 20, 20)
hogdata = [list(map(hog, row)) for row in deskewed] # the feature to train
trainData = np.float32(hogdata).reshape(-1, 64)
responses = np.repeat(np.arange(10), 250)[:, np.newaxis]

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC) # this type gives penalty to misclassified data
svm.setC(2.67) # the hyper parameter C
svm.setGamma(5.383)
# RBF函数作为kernel后， one hyper parameter. The bigger of gamma, the less support vectors.

svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
# svm.save('../data/svm_data.dat')

deskewed = [list(map(deskew, row)) for row in test_cells]
hogdata = [list(map(hog, row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1, bin_n*4)
result = svm.predict(testData)[1]

mask = result==responses
correct = np.count_nonzero(mask)*100.0 / result.size

print(correct)
