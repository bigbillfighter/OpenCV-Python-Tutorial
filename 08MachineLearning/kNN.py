import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

### <----- k-Nearest Neighbour Method -----> ###

# This is a simple algorithm. Usually, we set an number as k, and consider the nearest k points in
# dataset away from the object point in feature space. If one class has the most votes, we just
# classify the object point as this class.
# Sometimes, there are more than 1 most-vote class. So we just consider the distances in them, and
# then choose the closest class as the final.
# Sometimes, we set the distance with weight, because sometimes it will happen that a few points of
# one class are very close to the object, and more points of another class are far from the object.
# If we set k big enough, the class with more points will win. But it seems not justified. So
# we can give weight to distance, and make the result more reasonable.

trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

red = trainData[responses.ravel()==0]
blue = trainData[responses.ravel()==1]

plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

newcomer = np.random.randint(0, 100, (3, 2)).astype(np.float32)
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')

knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
# the second return argument is the class
# the third is the class of k neighbours
# the fourth is the distances of the k neighbours
ret, results, neighbours, dist = knn.findNearest(newcomer, k=3)
print("result: {}".format(results))
print("neighbours: {}".format(neighbours))
print("distance: {}".format(dist))

plt.show()