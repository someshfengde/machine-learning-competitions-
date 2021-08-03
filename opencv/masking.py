import cv2 as cv
import numpy as np

img = cv.imread('/home/som/Pictures/baby.png')
cv.imshow("image", img)

blank = np.zeros(img.shape[:2], dtype='uint8')


# drawing circle over the blank image and calling it as mask
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

cv.imshow('mask', mask)
bitwise_and = cv.bitwise_and(img, img, mask=mask)
cv.imshow('masked_image ', bitwise_and)


rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), (255, -1))


cv.imshow('weird shape ', rectangle)

cv.waitKey(0)
