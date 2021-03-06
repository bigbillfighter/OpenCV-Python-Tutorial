import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

data = np.loadtxt('../data/letter-recognition.data', dtype = 'float32', delimiter=',',
                  converters={0:lambda ch:ord(ch)-ord('A')}) # data(20000, 17)
# the first column is the class which has been converted to integer, and the next 16 columns are
# features.

train, test = np.vsplit(data, 2) # they are all (10000, 17)

responses, trainData = np.hsplit(train, [1])
labels, testData = np.hsplit(test, [1])

knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, result, _, _ = knn.findNearest(testData, 5)

correct = np.count_nonzero(result==labels)
accuracy = correct * 100.0 / test.shape[0]
print(accuracy)