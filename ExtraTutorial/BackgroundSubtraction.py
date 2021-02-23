from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='../img/vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')

args = parser.parse_args()

# two subtraction methods
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(args.input)
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

cv.namedWindow('Frame'), cv.moveWindow('Frame', 0, 0)
cv.namedWindow('FG Mask'), cv.moveWindow('FG Mask', int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), 0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # every frame is used to get the foreground and update the background, we can change the learning rate here.
    fgMask = backSub.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break