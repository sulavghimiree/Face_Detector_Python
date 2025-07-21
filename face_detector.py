import cv2 # for image processing
import numpy as np # for arrays
import os # for file paths
import sqlite3 # for database

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load face detector
cam = cv2.VideoCapture(0) # start camera

recognizer = cv2.face.LBPHFaceRecognizer_create() # create recognizer
recognizer.read('recognizer/trainingData.yml') # read trained data

def getProfile(id): # function to get profile from database
    conn = sqlite3.connect('database.db') # connect to database
    query = "SELECT * FROM students WHERE ID = " + str(id) # query to get profile
    cursor = conn.execute(query) # execute query
    profile = None # initialize profile
    for row in cursor: # loop through results
        profile = row # get profile
    conn.close() # close connection
    return profile # return profile

while True: # loop forever
    ret, img = cam.read() # read camera image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to grayscale
    faces = faceDetect.detectMultiScale(gray, 1.3, 5) # detect faces
    for(x, y, w, h) in faces: # loop through faces
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) # draw rectangle around face
        id, conf = recognizer.predict(gray[y:y+h, x:x+w]) # predict ID
        
        # Check if confidence is less than a threshold (e.g., 70), you can adjust this value
        if conf < 70:
            profile = getProfile(id) # get profile
            if profile != None: # if profile exists
                cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2) # write name
                cv2.putText(img, "Age: " + str(profile[2]), (x, y+h+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2) # write age
        else:
            cv2.putText(img, "Not Recognized", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # write not recognized
    
    cv2.imshow('Face', img) # show image
    if(cv2.waitKey(1) == ord('q')): # wait for 'q' key to exit
        break # exit loop

cam.release() # release camera
cv2.destroyAllWindows() # close all windows