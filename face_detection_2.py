from __future__ import print_function

import os
import pytesseract
import cv2 as cv
from builtins import input
import numpy as np
import argparse
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def adjustBrightness():
    # Read image given by user
    parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
    args = parser.parse_args()
    image = cv.imread(cv.samples.findFile(args.input))
    if image is None:
        print('Could not open or find the image: ', args.input)
        exit(0)
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.0  # Simple contrast control
    beta = 0  # Simple brightness control
    # Initialize values
    print(' Basic Linear Transforms ')
    print('-------------------------')
    try:
        alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
        beta = int(input('* Enter the beta value [0-100]: '))
    except ValueError:
        print('Error, not a number')
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
    cv.imshow('Original Image', image)
    cv.imshow('New Image', new_image)
    # Wait until user press some key
    cv.waitKey()


def detectAndDisplay(frame):
    text = pytesseract.image_to_string(frame)
    print(text)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    text2 = pytesseract.image_to_string(frame_gray)
    print(text)


    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray,scaleFactor = 1.1,
    minNeighbors = 2,
    minSize = (10, 10))

    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI,1.03,1)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        eyes = eyes_cascade_2.detectMultiScale(faceROI,1.03,1)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (0, 255, 0 ), 4)
    cv.imshow('Capture - Face detection', frame)
    cv.waitKey()


face_cascade_name = 'classifier/haarcascade_frontalface_default.xml'
eyes_cascade_name = 'classifier/haarcascade_eye_tree_eyeglasses.xml'
eyes_cascade_name_2 = 'classifier/haarcascade_eye.xml'

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
eyes_cascade_2 = cv.CascadeClassifier()
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
if not eyes_cascade_2.load(cv.samples.findFile(eyes_cascade_name_2)):
    print('--(!)Error loading eyes cascade 2 ')
    exit(0)




for filename in os.listdir('res/Movie_Poster_Dataset/2015'):
    if filename.endswith(".jpg"):
        print('res/Movie_Poster_Dataset/2015' +filename)
        img = cv.imread('res/Movie_Poster_Dataset/2015/'+filename)
        detectAndDisplay(img)
        continue
    else:
        continue


