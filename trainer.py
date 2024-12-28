import os # for file paths
import cv2 # for image processing
import numpy as np  # for arrays
from PIL import Image # for image processing

recognizer = cv2.face.LBPHFaceRecognizer_create() # create recognizer
path = 'dataset' # path to dataset

def getImageswithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)] # get all image paths
    faces = [] # list to store faces
    IDs = [] # list to store IDs
    for imagePath in imagePaths: # loop through all images
        faceImg = Image.open(imagePath).convert('L') # open image and convert to grayscale
        faceNp = np.array(faceImg, 'uint8') # convert image to numpy array
        id = int(os.path.split(imagePath)[-1].split('.')[1]) # get ID from image name
        print(id) # print ID
        faces.append(faceNp) # append face to faces list
        IDs.append(id) # append ID to IDs list
        cv2.imshow('Training', faceNp) # show training image
        cv2.waitKey(10) # wait for 10ms
    
    return np.array(IDs), faces # return IDs and faces

Ids, faces = getImageswithID(path) # get IDs and faces
recognizer.train(faces, Ids) # train recognizer
recognizer.save('recognizer/trainingData.yml') # save trained data
cv2.destroyAllWindows() # close all windows