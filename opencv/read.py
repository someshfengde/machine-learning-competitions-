import cv2 as cv
img = cv.imread('/home/som/Downloads/background.jpg')
cv.imshow('cat', img)
cv.waitKey(0)

# reading fhte videos using open cv
capture = cv.VideoCapture('/home/som/Downloads/video.mp4')

capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    cv.imshow('video', frame)
    if cv.waitKey(20) & 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)

# how to resize and rescale images and videos in opencv

img = cv.imread('/home/som/Downloads/background.jpg')


def rescaleFrame(frame, scale=0.2):
    # images video and live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


cv.imshow('Image', rescaleFrame(img))

capture = cv.VideoCapture('/home/som/Downloads/video.mp4')
while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame=frame)
    cv.imshow('video', frame)
    cv.imshow('video_resized', frame_resized)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()

capture = cv.VideoCapture(0)
# changing resolution of live video


def change_res(width, height):
    capture.set(3, width)
    capture.set(4, height)

    # only working on the live video
change_res(1080, 1020)

while True:
    isTrue, frame = capture.read()
    cv.imshow('video', frame)
    if cv.waitKey(20) and 0xff == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
