import imutils
import cv2
import os
import numpy as np


class FaceDetectorSSD():
    '''
        Single Shot Detectors "SSD" based face detection
    '''

    def __init__(self,dir="GP_Blind_Features/pretrained_models",prototxt="deploy.prototxt.txt",model="res10_300x300_ssd_iter_140000.caffemodel",probability=0.5):
        self.prototxt = os.path.join(dir,prototxt)
        self.model = os.path.join(dir,model)
        self.face_detector = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

        # the probability of faces we want to detects
        self.proba=probability
    

    def detect(self,image):
        '''
        Input:
            image: the image to detect the faces in it
        Output:
            returns "faces" which is a list of dictionaries
            each one has 'box' contians a tupe (x,y,w,h)
            and 'confidence' contains the probabiliy the model detected that this is a face
        '''

        (h, w) = image.shape[:2]

        ## preprocess the image to make it easier and faster for the model to predict
        image = imutils.resize(image, width=400)
        image=cv2.resize(image, (300, 300))
        blob = cv2.dnn.blobFromImage(image, 1.0,image.shape[:2], (104.0, 177.0, 123.0))

        ## apply the model to detect faces
        self.face_detector.setInput(blob)
        detections=self.face_detector.forward()

        faces=list()

        for i in range(0, detections.shape[2]): #loop over each face

            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence < self.proba:
                continue

            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # create a dictionary 
            # box (x,y,w,h)
            d= {'box':(startX, startY, endX-startX, endY-startY) ,'confidence':confidence}

            #save the face
            faces.append(d)
        
        return faces


