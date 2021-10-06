import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#To capture a video from webcam
webcam = cv2.VideoCapture(0)

#Iterate forever in one frame
while True:

    #Read the current frame
    successful_frame_read, frame = webcam.read()

    #Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect Face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 8)

    cv2.imshow('Aadil face detector', frame)
    key = cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release the video capture object
webcam.release()
