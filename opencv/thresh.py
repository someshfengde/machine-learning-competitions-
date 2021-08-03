import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

img = cv.imread('/home/som/Pictures/baby.png')
cv.imshow("image", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray_color ', gray)


# simple threshholding
threshold, thresh = cv.threshold(gray, 150, 255,
                                 cv.THRESH_BINARY)

# displaying thresholded iamge
cv.imshow('simple threshholded', thresh)

# simple threshholding
threshold, thresh_inv = cv.threshold(gray, 150, 255,
                                     cv.THRESH_BINARY_INV)

cv.imshow('inversed thresh', thresh_inv)
# displaying thresholded iamge
cv.imshow('simple threshholded', thresh)

# adaptive thresholding : differnet images when we provided different threshholded value s
# we can manually specifies the inverse values
# we could essentially let computer find the acutal thresholdin valuyes itself
# and let it binarize itself
adaptive_thresh = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 9)

cv.imshow('adaptive thresholding ', adaptive_thresh)


cv.waitKey(0)
