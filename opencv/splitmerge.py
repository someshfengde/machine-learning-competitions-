import cv2 as cv
import numpy as np

img = cv.imread('/home/som/Pictures/baby.png')

blank = np.zeros(img.shape[:2], 'uint8')


cv.imshow("image", img)

b, g, r = cv.split(img)


blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])
cv.imshow('blue', blue)
cv.imshow('green', green)
cv.imshow('red', red)
print(img.shape)
print(f'blue {b.shape},red {r.shape} , green {g.shape}')


merged = cv.merge([b, g, r])
cv.imshow('merged image', merged)
cv.waitKey(0)
