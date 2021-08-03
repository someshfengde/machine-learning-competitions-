import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

img = cv.imread('/home/som/Pictures/baby.png')
cv.imshow("image", img)

# masking and polotting
blank = np.zeros(img.shape[:2], dtype='uint8')

mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
cv.imshow('mask ', mask)
# plotting histogram of grayscale image
# converting image to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('masked image ', masked)

# cv.imshow('grayscale image ', gray)

# gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])


# plt.figure()
# plt.title('grayscale histogram')
# plt.xlabel('Bins')
# plt.ylabel('no of pixels')

# # plotting the hist
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.show()


plt.figure()
plt.title('colored histogram')
plt.xlabel('Bins')
plt.ylabel('no of pixels')
# color historgram

colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()

cv.waitKey(0)
