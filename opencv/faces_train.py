import os
import cv2 as cv
import numpy as np
from numpy.core.numerictypes import ScalarType
people = ['Ben Afflek', 'Elton John',
          'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# p = []

# for i in os.listdir(r'/home/som/Desktop/code/machine-learning-competitions-/opencv/train'):
# p.append(i)

DIR = '/home/som/Desktop/code/machine-learning-competitions-/opencv/train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# print(p
features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                       minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_train()

print(
    f'Length of hte features list = {len(features)} and length of lables are {len(labels)}')


# training recognizer on features and labels list
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

print('training doneeeeee')
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

# pylint:disable=no-member

# import os
# import cv2 as cv
# import numpy as np

# people = ['Ben Afflek', 'Elton John',
#           'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# DIR = './train/'

# haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = []
# labels = []


# def create_train():
#     for person in people:
#         path = os.path.join(DIR, person)
#         label = people.index(person)

#         for img in os.listdir(path):
#             img_path = os.path.join(path, img)

#             img_array = cv.imread(img_path)
#             if img_array is None:
#                 continue

#             gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

#             faces_rect = haar_cascade.detectMultiScale(
#                 gray, scaleFactor=1.1, minNeighbors=4)

#             for (x, y, w, h) in faces_rect:
#                 faces_roi = gray[y:y+h, x:x+w]
#                 features.append(faces_roi)
#                 labels.append(label)


# create_train()
# print('Training done ---------------')

# features = np.array(features, dtype='object')
# labels = np.array(labels)

# face_recognizer = cv.face.LBPHFaceRecognizer_create()

# # Train the Recognizer on the features list and the labels list
# face_recognizer.train(features, labels)

# face_recognizer.save('face_trained.yml')
# np.save('features.npy', features)
# np.save('labels.npy', labels)
