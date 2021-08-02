import cv2 as cv

img = cv.imread('/home/som/Pictures/baby.png')

# cv.imshow('image', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('grayscale', gray)


# blurring effect there are some effects maybe because of bad lightning or some issues at snesor
# using gaussian image
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
# cv.imshow('blur', blur)

# creating the edge cascades : finding the edges in the images
canny = cv.Canny(img, 125, 175)
cv.imshow('canny edges', canny)


# blurry canny images
canny_blur = cv.Canny(blur, 125, 175)
cv.imshow('canny blurred image', canny_blur)


# dilating the image using the specific structuring elements (canny images )

dilated = cv.dilate(canny_blur, (7, 7), iterations=3)
cv.imshow('dilated image based on blurry canny', dilated)

# eroding images
eroded = cv.erode(dilated, (3, 3), iterations=3)
cv.imshow('eroded image ', eroded)

# resizing the image
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('resized', resized)

# cropping the image
cropped = img[50:200, 200:400]
cv.imshow('cropped', cropped)


cv.waitKey(0)
