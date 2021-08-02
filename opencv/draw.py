import cv2 as cv
import numpy as np
from numpy.core.fromnumeric import shape

# img = cv.imread('/home/som/Downloads/background.jpg')
# cv.imshow('image', img)

blank = np.zeros((500, 500, 3), dtype='uint8')
# cv.imshow('blank', blank)
# cv.waitKey(0)

# drawing the point on the image

blank[200:300, 300:400] = 0, 255, 0  # painting the entire image green
# cv.imshow('Green', blank)


# drawing the rectangle using cv.rectangle
cv.rectangle(blank, (0, 0),
             (blank.shape[0]//2, blank.shape[0]//2), (0, 255, 0), thickness=-1)
cv.imshow('Rectangle', blank)

# drawing circle
cv.circle(blank, (blank.shape[1]//2,
          blank.shape[0]//2), 40, (0, 0, 255), thickness=2)
cv.imshow('circle', blank)

cv.line(blank, (0, 0),
        (300, 400), (255, 255, 255), thickness=3)

cv.imshow('line', blank)

# writing the text on image
cv.putText(blank, "Hello we are here ", (0, 225),
           fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)
cv.imshow('textimage ', blank)
cv.waitKey(0)
