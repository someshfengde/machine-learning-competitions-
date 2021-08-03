import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

img = cv.imread('/home/som/Pictures/baby.png')
cv.imshow("image", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray_color ', gray)

# laplacing L edges drawn over the image and
# they are little bit smudged

lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))

cv.imshow('laplcian', lap)

# sobel
sobel_x = cv.Sobel(gray, cv.CV_64F, 1,  0)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1)
commbined_sobel = cv.bitwise_or(sobel_x, sobel_y)

cv.imshow('sobelx', sobel_x)
cv.imshow('sobel_y', sobel_y)
cv.imshow('combined sobel', commbined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow('canny', canny)

cv.waitKey(0)
