import cv2 as cv
import numpy as np

img = cv.imread('/home/som/Pictures/baby.png')
cv.imshow('image', img)

# translation shifting image from x and y axis


def translate(img, x, y):
    trans_mat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, trans_mat, dimensions)


translated = translate(img, -100, 100)
cv.imshow('translated', translated)


# rotating the image
def rotate(img, angle, rot_point=None):
    (height, width) = img.shape[:2]

    if rot_point is None:
        rot_point = (width//2, height//2)
    rotMat = cv.getRotationMatrix2D(rot_point, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)


rotated_image = rotate(img, -45)
cv.imshow('rotated', rotated_image)

rotated_rotated = rotate(rotated_image, 45)
cv.imshow('again rotated ', rotated_rotated)
# resizing
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('resized', resized)

# flipping the image
# 0 for flipping vertically 1 for flipping horiozntly , -1 for flipping both horizontally and vertically
flip = cv.flip(img, -1)
cv.imshow('flip', flip)


# cropping the image
cropped = img[200:400, 300:400]
cv.imshow('cropped image ', cropped)


cv.waitKey(0)
