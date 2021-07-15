import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, SeparableConv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, Add, Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from face_detection import FaceDetectorSSD
from face_alignment import FaceAlignment
import sys
class XceptionModel():
    '''
        Base Class
        Load or Build the Xception Blocks
    '''
    def __init__(self,pretrained_path=None):
        if pretrained_path is not None:
            self.model = tf.keras.models.load_model(pretrained_path)
        else:
            self.model= None
        
        self.face_detector = FaceDetectorSSD()
        self.alignment= FaceAlignment()
    
    def __xception_block(self,input_tensor, num_kernels, l2_reg=0.01):
        #  params
        #     input_tenso: Keras tensor.
        #     num_kernels: Int. Number of convolutional kernels in block.
        #     l2_reg: Float. l2 regression.
        #  Returns
        #     output tensor for the block.
        
        residual = Conv2D(num_kernels, 1, strides=(2, 2),padding='same', use_bias=False)(input_tensor)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(num_kernels, 3, padding='same',kernel_regularizer=l2(l2_reg), use_bias=False)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(num_kernels, 3, padding='same',kernel_regularizer=l2(l2_reg), use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=(2, 2), padding='same')(x)
        x = Add()([x, residual])
        return x
    
    def __build_xception(self,input_shape, num_classes, kernels, block_kernels, l2_reg=0.01):
        #  params
        #     input_shape: List corresponding to the input shape of the model.
        #     num_classes: Integer.
        #     kernels: List of integers. Each element of the list indicates
        #         the number of kernels used as stem blocks.
        #     block_kernels: List of integers. Each element of the list Indicates
        #         the number of kernels used in the xception blocks.
        #     l2_reg. Float. L2 regularization used in the convolutional kernels.
        #  Returns
        #     Tensorflow-Keras model.
        

        x = inputs = Input(input_shape, name='image')
        for num_kernels in kernels:
            x = Conv2D(num_kernels, 3, kernel_regularizer=l2(l2_reg),use_bias=False, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        for num_kernels in block_kernels:
            x = self.__xception_block(x, num_kernels, l2_reg)

        x = Conv2D(num_classes, 3, kernel_regularizer=l2(l2_reg),padding='same')(x)
    
        x = GlobalAveragePooling2D()(x)
        output = Activation('softmax', name='label')(x)

        model_name = '-'.join(['XCEPTION',str(input_shape[0]),str(kernels[0]),str(len(block_kernels))])
        model = Model(inputs, output, name=model_name)
        return model

    def __Xception(self,input_shape, num_classes):
        #  params
        #     input_shape: List of three integers
        #     num_classes: Int.
        #  Returns
        #     Tensorflow-Keras model.

        stem_kernels = [32, 64]
        block_data = [128, 128, 256, 256, 512, 512, 1024]
        model_inputs = (input_shape, num_classes, stem_kernels, block_data)
        model = self.__build_xception(*model_inputs)
        model._name = 'MINI-XCEPTION'
        return model
    
    def build_model(self,input_shape=None, num_classes=None):
        '''
            if a pretrained model is given as input will return it
            else will build and initialize a new xception model
        '''
        if self.model is not None:
            return self.model
        else:
            return self.__Xception(input_shape,num_classes)
    
    def detect(self,image,draw=False):
        '''
            input:
                image: the image to detect the features from it
                draw: flag, if true the output is drawn on the image else image won't change
            output:
                takes the image and then
                    - First extracts the faces using FaceDetectorSSD
                    - Then, for each face will align and give it input to the model 
                    - Return 
                        -- image
                        -- faces: list of each face detected
                        -- features: list of each face's feature given from the model
        '''
        img = np.copy(image)
        labels= self.get_labels()
        faces = self.face_detector.detect(image)
        face_images=list()
        out= list()
        for face in faces:
            (x,y,w,h) = face['box']
            face = self.alignment.frontalize_face(face['box'],image)

            if type(face) ==type(None):
                continue

            # save the face
            face_images.append(face)
            

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face , (48,48))
            face = cv2.equalizeHist(face)

            ######### Apply the model
            outputs= self.model.predict(np.expand_dims(face,axis=0)/255.0)
            c = np.argmax(outputs)
            o = labels[c]

            # save the out
            out.append(o)

            if draw:
                # draw the bounding box of the face along with the associated probability
                y1 =y - 10 if y - 10 > 10 else y + 10
                img=cv2.rectangle(img, (x, y), (x+w, y+h),(0, 0, 255), 2)

                # Text
                text= "{}".format(o)
                img=cv2.putText(img, text, (x, y1), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return img, face_images,out
    
    def get_labels(self):
        '''
            returns a tuple of labels of the model 
            "depends on the model type"
        '''
        raise NotImplementedError


class EmotionRecognition(XceptionModel):
    '''
        Emotion Recognition model build based on Xception model strucutre
    '''
    def __init__(self,pretrained_path=None):
        super(EmotionRecognition,self).__init__(pretrained_path)
    
    def get_labels(self):
        return ('Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral')





class AgeGenderClassification():
    '''
        Age Classification Class based on caffee model
    '''
    def __init__(self,dir="",prototxt="",model="",margin=None):
        self.prototxt = os.path.join(dir,prototxt)
        self.model = os.path.join(dir,model)
        self.recognizer = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

        self.face_detector = FaceDetectorSSD()
        self.margin=margin
    
    def detect(self, image,draw=False):
        faces = self.face_detector.detect(image)

        img = np.copy(image)

        margin = self.margin
        img_h , img_w = image.shape[:2]
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        labels = self.get_labels()

        face_images= list()
        output= list()
        

        for face in faces:
            x1,y1,w,h= face['box']
            x2,y2 = x1+w+1, y1+h+1
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            face = image[yw1:yw2 + 1, xw1:xw2 + 1]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            self.recognizer.setInput(blob)
            predictions = self.recognizer.forward()

            o = labels[predictions[0].argmax()]
            face_images.append(face)
            output.append(o)

            if draw:
                y0 =y1 - 10 if y1 - 10 > 10 else y1 + 10
                img = cv2.rectangle(img, (xw1, yw1), (xw2+1, yw2+1), (255, 0, 0), 2)

                text= "{}".format(o)
                img=cv2.putText(img, text, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return img, face_images,output


    def get_labels(self):
        raise NotImplementedError


class GenderRecognition(AgeGenderClassification):
    def __init__(self,dir="GP_Blind_Features/pretrained_models",prototxt="deploy_gender.prototxt",model="gender_net.caffemodel",margin=0):
        super(GenderRecognition,self).__init__(dir,prototxt,model,margin)
    
    def get_labels(self):
        return ['Male', 'Female']
    


class AgeDetector():
    '''
        Age estimation class based on DEX model
    '''
    def __init__(self,pretrained_path="GP_Blind_Features/pretrained_models/DEX_age.hdf5",margin=0.26):
        self.model = tf.keras.models.load_model(pretrained_path)

        self.face_detector = FaceDetectorSSD()
        self.margin=margin
    

    
    def detect(self, image,draw=False):
        '''
            input:
                image: the image to detect the age from it
                draw: flag, if true the output is drawn on the image else image won't change
            output:
                takes the image and then
                    - First extracts the faces using FaceDetectorSSD
                    - Then, for each face detect the age
                    - Return 
                        -- image
                        -- faces: list of each face detected
                        -- features: list of each face's feature given from the model
        '''
        faces = self.face_detector.detect(image)

        img = np.copy(image)

        margin = self.margin
        img_h , img_w = image.shape[:2]

        face_images= list()
        output= list()
        

        for face in faces:
            x1,y1,w,h= face['box']
            x2,y2 = x1+w+1, y1+h+1
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            face = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (224, 224))
            


            results= self.model.predict(np.expand_dims(face,0))
            predicted_ages = np.argmax(results[1])

            o = predicted_ages
            face_images.append(face)
            output.append(o)

            if draw:
                y0 =y1 - 10 if y1 - 10 > 10 else y1 + 10
                img = cv2.rectangle(img, (xw1, yw1), (xw2+1, yw2+1), (255, 0, 0), 2)

                text= "{}".format(o)
                img=cv2.putText(img, text, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return img, face_images,output
img  = cv2.imread('images/test.jpg')
if  sys.argv[1] == "emotion":
    Emodel = EmotionRecognition(r"GP_Blind_Features/pretrained_models/xception_model_99.hdf5")
    print(Emodel.detect(img)[2][0])
elif  sys.argv[1] == "gender":
    Gmodel = GenderRecognition()
    print(Gmodel.detect(img)[2][0])
elif  sys.argv[1] == "age":
    Amodel = AgeDetector()
    print(Amodel.detect(img)[2][0])
