import cv2          # OpenCV
import numpy as np     # Numpy
import sqlite3         # SQLite

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Load the face detection model
cam = cv2.VideoCapture(0)       # Open the camera

def insertOrUpdate(Id, Name, age):   # Function to insert or update the details in the database
    conn = sqlite3.connect("database.db")   # Connect to the database
    cmd = "SELECT * FROM STUDENTS WHERE ID = " + str(Id)   # Select the record with the given ID
    cursor = conn.execute(cmd)   # Execute the command
    isRecordExist = 0   # Flag to check if the record exists
    for row in cursor:   # Loop through the records
        isRecordExist = 1   # Set the flag to 1
    
    if isRecordExist == 1:   # If the record exists
        conn.execute("UPDATE STUDENTS SET Name = ?, Age = ? WHERE ID = ?",(Name, age, Id))   # Update the record
    else:   # If the record does not exist
        conn.execute("INSERT INTO STUDENTS(ID, Name, Age) VALUES(?, ?, ?)",(Id, Name, age))  # Insert the record
    
    conn.commit()   # Commit the changes
    conn.close()   # Close the connection


# Take the input from the user
Id = input('Enter the ID of the Student:')  # Take the ID of the student
Name = input('Enter the Name of the Student:')  # Take the Name of the student
age = input('Enter the Age of the Student:')  # Take the Age of the student

insertOrUpdate(Id, Name, age)

# Detect the face and save the images
sampleNum = 0   # Initialize the sample number
while(True):
    ret, img = cam.read()   # Read the image from the camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # Convert the image to grayscale
    faces = face_detect.detectMultiScale(gray, 1.3, 5)   # Detect the faces in the image
    for (x, y, w, h) in faces:   # Loop through the faces
        sampleNum = sampleNum + 1   # Increment the sample number
        cv2.imwrite("dataset/User." + str(Id)+"."+str(sampleNum)+".jpg", gray[y:y+h, x:x+w])   # Save the image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw a rectangle around the face
        cv2.waitKey(100)  # Wait for 100ms
    cv2.imshow("Face", img)   # Show the image
    cv2.waitKey(1)   # Wait for 1ms
    if(sampleNum > 20):   # If 20 images are captured
        break   # Break the loop

cam.release()   # Release the camera
cv2.destroyAllWindows()   # Close the windows