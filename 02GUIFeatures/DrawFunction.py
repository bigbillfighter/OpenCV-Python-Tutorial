import cv2 as cv
import numpy as np

# create a black image
img = np.zeros((512, 512, 3), np.uint8)

# draw a diagonal blue line with thickness of 5 px
cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
# the second and third parameter is the coordinate of top-left and bottom-right corner
cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
# the second parameter is the coordinate of center and the third is the radius
# -1 means to fill the shape
cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
# the first argument is the center coordinate, second the major axis length and the minor axis length
# the third the rotation in anti-clockwise
# the forth and the fifth is the start degree and the end degree of the arc measured in clockwise
cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)

# the coordinates of vertices, the shape should be rows*1*2
# rows is the number of vertices and should be of type int32
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
# the third argument is if the polyline is closed, if False, the line connects the first
# and the last vertice won't exist.
# the second should be a list
cv.polylines(img, [pts], True, (0, 255, 255)) # yellow color

font = cv.FONT_HERSHEY_SIMPLEX
# the second is the content of the text
# the third is the bottom-left coordinate of the text box
# the fourth is the font
# the fifth is the font-scale
# the seventh is the color
# the eighth is the line thickness
# the ninth is the line type
cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()