import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../img/digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# split hte image to 5000 cells, each image is 20x20
# np.vsplit will split the rows and np.hsplit will split the columns
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells) # x(50, 100, 20, 20)

train = x[:, :50].reshape(-1, 20*20).astype(np.float32) # flatten the image
test = x[:, 50:].reshape(-1, 20*20).astype(np.float32)

k = np.arange(10) # from 0 to 9
train_labels = np.repeat(k, 250)[:, np.newaxis] # train_labels(2500, 1)
# train_labels = [[0], [0], ..., [0], [1], [1], ..., ..., [9]]
test_labels = train_labels.copy()

knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=5)

matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.shape[0]

print(accuracy)

# np.savez('../data/npzfiles/knn_data.npz', train=train, train_labels=train_labels) # binary file

with np.load('../data/npzfiles/knn_data.npz') as data:
    print(data.files)
    train = data['train']
    train_labels = data['train_labels']
