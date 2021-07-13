import dlib
import numpy as np
from enum import Enum
import os


class LandmarksDetector():

    def __init__(self, dir="GP_Blind_Features/pretrained_models",predictor="shape_predictor_5_face_landmarks.dat"):
        self.predictor = os.path.join(dir,predictor)
        self.detector = dlib.shape_predictor(self.predictor)

    def convert_to_numpy(self, landmarks):
        num_landmarks = 5
        coords = np.zeros((num_landmarks, 2), dtype=np.int)
        for i in range(num_landmarks):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
        return coords

    def detect(self, frame, rect):
        # landmarks detection accept only dlib rectangles to operate on
        if type(rect) != dlib.rectangle:
            (x,y,w,h) = rect
            rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

        # convert from dlib style to numpy style
        landmarks = self.detector(frame, rect)
        return self.convert_to_numpy(landmarks)