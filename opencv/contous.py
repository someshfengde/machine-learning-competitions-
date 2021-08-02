# contous : line or curve which are the edges of the images, they are image object detection and they have the
# other outer shape of image

import cv2 as cv
import numpy as np
from numpy.core.defchararray import count


img = cv.imread('/home/som/Pictures/baby.png')
cv.imshow('image', img)
blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('blank_image ', blank)

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imshow('gray ', gray)


blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('blur', blur)

canny = cv.Canny(blur, 125, 175)
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

contours, hierarchies = cv.findContours(
    canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cv.imshow('canny', canny)
print(f'{len(contours)} this many countours found ')
# RETR_LIST : all the contours of the image
# RETR_EXTERNAL  : external edges only
# RETR_TREE : all the hierarchial counters

# contour approximation method
# chain_approx_none : does nothing returns as python list only
# chain_approx_simmple : takes all of those points and compresses it in 2 endpoints only
# cv.threshold : takes in the threshold values if it's below the threshold 1 set's pixel value to 0 it it's above _threshhold_2 then sets to max possilble i.e 255
cv.imshow('thresh', thresh)

# drawing the countours on the draw images
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow('Counters_drawn ', blank)

# thresh hold is lowly preferred canny is more prefered for finding the countours


cv.waitKey(0)
