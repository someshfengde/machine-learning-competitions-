import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('/home/som/Pictures/baby.png')
cv.imshow('image', img)

# bgr to the grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray ', gray)


# image to the hsv format bgr to hsv hue saturatoin value
# based on how human thinks and convince the color
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('hsv', hsv)

# BGR to L * a * b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('lab', lab)

# bgr to rfb
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('rgb', rgb)
# plt.imshow(rgb)
# plt.show()

# hsv to bgr
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('hsv_to_bgr', hsv_bgr)


cv.waitKey(0)
