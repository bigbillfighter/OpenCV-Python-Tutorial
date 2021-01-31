import cv2 as cv

cap = cv.VideoCapture(0)

# define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
# the fourcc is the definition of the video format, like the defition of mp4, flv, etc.

out = cv.VideoWriter(r'F:\Movies\video\out.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed.")
        break
    frame = cv.flip(frame, 1)
    # 0 means rotate around x-axis, 1 means y-axis, -1 means both

    # write the flipped frame
    out.write(frame)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
