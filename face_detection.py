import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('classifier/haarcascade_smile.xml')
# Read the input image
img = cv2.imread('res/Movie_Poster_Dataset/2015/tt3530002.jpg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.5,2)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
# Display the output
cv2.imshow('img', img)
cv2.waitKey()
