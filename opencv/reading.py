import cv2 as cv

import matplotlib.pyplot as plt
image = cv.imread('/home/som/Downloads/brain.png')
cv.imshow('image', image)

print(image.shape)
image = plt.imread('/home/som/Downloads/brain.png')
print('plt image ', image.shape)
cv.waitKey(0)
