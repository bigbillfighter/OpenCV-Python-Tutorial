import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img/messi5.jpg')
cv.namedWindow('img')
cv.namedWindow('result')


rect = []
drawing = False
mask = np.zeros(img.shape[:2], np.uint8)
newmask = mask.copy()
img_cp = img.copy()
if_running = False


# Temporary array to detect fg and bg
bgbModel = np.zeros((1, 65), np.float64)
fgbModel = np.zeros((1, 65), np.float64)

def mouse_call1(event, x, y, flags, params):
    global rect, img_cp
    if event==cv.EVENT_LBUTTONUP:
        rect.append(x), rect.append(y)
        cv.circle(img_cp, (x, y), 2, (0, 0, 255), 2)
        cv.imshow('img', img_cp)

def mouse_call2(event, x, y, flags, params):
    global drawing, mask, img_cp, if_running
    if event==cv.EVENT_LBUTTONDOWN:
        drawing = True
    if event==cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.circle(newmask, (x, y), 2, 255, -1)
            cv.circle(img_cp, (x, y), 2, (255, 255, 255), -1)
            cv.imshow('img', img_cp)
    if event==cv.EVENT_LBUTTONUP:
        drawing= False
        if_running = True

cv.imshow('img', img_cp)
cv.moveWindow('img', 0, 0)
result = img.copy()
cv.imshow('result', result)
cv.moveWindow('result', img.shape[1], 0)


while len(rect)<4:
    cv.setMouseCallback('img', mouse_call1)
    k = cv.waitKey(10)&0xff
    if k==27 or k==ord('q'):
        exit(0)

cv.rectangle(img_cp, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (255, 0, 0), 2)
cv.imshow('img', img_cp)

mask, bgbModel, fgbModel=cv.grabCut(result, mask, (rect[0], rect[1], rect[2], rect[3]), bgbModel, fgbModel, 5, cv.GC_INIT_WITH_RECT)
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
result = result * mask[:, :, np.newaxis]
cv.imshow('result', result)
while 1:
    k = cv.waitKey(10) & 0xff
    if k==27 or k==ord('q'):
        break
    cv.setMouseCallback('img', mouse_call2)
    if if_running:
        mask[newmask == 255] = 1
        result = img.copy()
        mask, bgbModel, fgbModel = cv.grabCut(result, mask, None, bgbModel, fgbModel, 5, cv.GC_INIT_WITH_MASK)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = result * mask[:, :, np.newaxis]
        if_running = False
        cv.imshow('result', result)

cv.destroyAllWindows()