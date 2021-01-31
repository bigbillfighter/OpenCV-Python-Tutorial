import numpy as np
import cv2 as cv

cap = cv.VideoCapture(r"F:\Movies\video\Closer.flv")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed.")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(25) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()