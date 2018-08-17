# -*- coding: utf-8 -*-
import cv2
import sys, os


imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar
facecascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = facecascade.detectMultiScale(
   gray,
   scaleFactor=1.1,
   minNeighbors=5,
   minSize=(30, 30),
   flags = cv2.CASCADE_SCALE_IMAGE
)

print('Found {0} faces'.format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y),(x + w, y + h),(0, 255, 0), 2)

cv2.imshow('Faces found', image)
cv2.waitKey(0)