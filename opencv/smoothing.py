import cv2 as cv

img = cv.imread('/home/som/Pictures/baby.png')
cv.imshow('image', img)

# averaging : the first method of blurring

average = cv.blur(img, (3, 3))
cv.imshow('average blur', average)

gauss = cv.GaussianBlur(img, (3, 3), 0)
cv.imshow('gaussian blur', gauss)

# medium blur
medium = cv.medianBlur(img, 3)
cv.imshow('median blur', medium)

# bilateral blurring
# most effective and used in most advance computer vision project
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('bilateral', bilateral)


cv.waitKey(0)
