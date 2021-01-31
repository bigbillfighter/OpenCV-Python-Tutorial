import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
# the parameter can the index of camera devices
# or the path of video file

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv.CAP_PROP_FRAME_WIDTH))

ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 640)
print(ret)

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    # if frame is read correctly, ret is True
    if not ret:
        print('Cannot receive frame (stream end?). Exiting...')
        break

    # our opreations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
    # wait for the shortest time

# when everything is done, release the capture
cap.release()
cv.destroyAllWindows()

