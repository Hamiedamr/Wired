import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import onnx
import cv2
from onnx_tf.backend import prepare
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.python.ops.distributions.uniform import Uniform
import random
import math

np.random.seed(1234)
class I3D_WLASL():
    '''
        Class of I3D Model trained of WLASL, saved & loaded using ONNX opensource packaging

        to install ONNX:
        git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow && pip install -e . 
    '''
    def __init__(self,model_path="GP_Deaf_Features/pretrained_models/i3d.onnx"):
        model = onnx.load(model_path)
        self.tf_rep = prepare(model)
        self.WEIGHTS = {'rgb_imagenet_and_kinetics_no_top':
        'GP_Deaf_Features/pretrained_models/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
        }
        self.labels=self.get_labels()
    
    def __Unit_3d(self,x,
            filters,
            num_frames,
            num_row,
            num_col,
            padding='same',
            strides=(1, 1, 1),
            use_bias=False,
            use_activation_fn=True,
            use_bn=True,
            name=None):
        """
        Utility function to apply conv3d + BN.
        :return: Output tensor after applying `Conv3D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        layer = Conv3D(
            filters, (num_frames, num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=conv_name)(x)

        if use_bn:
            layer = BatchNormalization(axis=4, scale=False, name=bn_name)(layer)

        if use_activation_fn:
            layer = Activation('relu', name=name)(layer)

        return layer
    
    def i3d_arch(self,include_top=True,
                          pretrained_weights='rgb_imagenet_and_kinetics',
                          input_shape=None,
                          dropout_prob=0.0,
                          endpoint_logit=True,
                          classes=400):
        """
        Instantiates the Inception i3D Inception v1 architecture.
        :return: Inception i3D model model.
        """
        if not include_top:
            pretrained_weights = pretrained_weights + '_no_top'

        if pretrained_weights not in self.WEIGHTS:
            raise ValueError('in valid weight name')

        input_shape = input_shape
        img_input = Input(shape=input_shape)
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 4

        # Downsampling via convolution (spatial and temporal)
        x = self.__Unit_3d(img_input, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', name='Conv3d_1a_7x7')

        # Downsampling (spatial only)
        x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)
        x = self.__Unit_3d(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv3d_2b_1x1')
        x = self.__Unit_3d(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3')

        # Downsampling (spatial only)
        x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)

        # Mixed 3b
        branch_0 = self.__Unit_3d(x, 64, 1, 1, 1, padding='same', name='Conv3d_3b_0a_1x1')

        branch_1 = self.__Unit_3d(x, 96, 1, 1, 1, padding='same', name='Conv3d_3b_1a_1x1')
        branch_1 = self.__Unit_3d(branch_1, 128, 3, 3, 3, padding='same', name='Conv3d_3b_1b_3x3')

        branch_2 = self.__Unit_3d(x, 16, 1, 1, 1, padding='same', name='Conv3d_3b_2a_1x1')
        branch_2 = self.__Unit_3d(branch_2, 32, 3, 3, 3, padding='same', name='Conv3d_3b_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3')(x)
        branch_3 = self.__Unit_3d(branch_3, 32, 1, 1, 1, padding='same', name='Conv3d_3b_3b_1x1')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name='Mixed_3b')

        # Mixed 3c
        branch_0 = self.__Unit_3d(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_0a_1x1')

        branch_1 = self.__Unit_3d(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_1a_1x1')
        branch_1 = self.__Unit_3d(branch_1, 192, 3, 3, 3, padding='same', name='Conv3d_3c_1b_3x3')

        branch_2 = self.__Unit_3d(x, 32, 1, 1, 1, padding='same', name='Conv3d_3c_2a_1x1')
        branch_2 = self.__Unit_3d(branch_2, 96, 3, 3, 3, padding='same', name='Conv3d_3c_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3c_3a_3x3')(x)
        branch_3 = self.__Unit_3d(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_3c_3b_1x1')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name='Mixed_3c')

        # Downsampling (spatial and temporal)
        x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

        # Mixed 4b
        branch_0 = self.__Unit_3d(x, 192, 1, 1, 1, padding='same', name='Conv3d_4b_0a_1x1')

        branch_1 = self.__Unit_3d(x, 96, 1, 1, 1, padding='same', name='Conv3d_4b_1a_1x1')
        branch_1 = self.__Unit_3d(branch_1, 208, 3, 3, 3, padding='same', name='Conv3d_4b_1b_3x3')

        branch_2 = self.__Unit_3d(x, 16, 1, 1, 1, padding='same', name='Conv3d_4b_2a_1x1')
        branch_2 = self.__Unit_3d(branch_2, 48, 3, 3, 3, padding='same', name='Conv3d_4b_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3')(x)
        branch_3 = self.__Unit_3d(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4b_3b_1x1')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name='Mixed_4b')

        # Mixed 4c
        branch_0 = self.__Unit_3d(x, 160, 1, 1, 1, padding='same', name='Conv3d_4c_0a_1x1')

        branch_1 = self.__Unit_3d(x, 112, 1, 1, 1, padding='same', name='Conv3d_4c_1a_1x1')
        branch_1 = self.__Unit_3d(branch_1, 224, 3, 3, 3, padding='same', name='Conv3d_4c_1b_3x3')

        branch_2 = self.__Unit_3d(x, 24, 1, 1, 1, padding='same', name='Conv3d_4c_2a_1x1')
        branch_2 = self.__Unit_3d(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4c_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4c_3a_3x3')(x)
        branch_3 = self.__Unit_3d(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4c_3b_1x1')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name='Mixed_4c')

        # Mixed 4d
        branch_0 = self.__Unit_3d(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_0a_1x1')

        branch_1 = self.__Unit_3d(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_1a_1x1')
        branch_1 = self.__Unit_3d(branch_1, 256, 3, 3, 3, padding='same', name='Conv3d_4d_1b_3x3')

        branch_2 = self.__Unit_3d(x, 24, 1, 1, 1, padding='same', name='Conv3d_4d_2a_1x1')
        branch_2 = self.__Unit_3d(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4d_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4d_3a_3x3')(x)
        branch_3 = self.__Unit_3d(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4d_3b_1x1')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name='Mixed_4d')

        # Mixed 4e
        branch_0 = self.__Unit_3d(x, 112, 1, 1, 1, padding='same', name='Conv3d_4e_0a_1x1')

        branch_1 = self.__Unit_3d(x, 144, 1, 1, 1, padding='same', name='Conv3d_4e_1a_1x1')
        branch_1 = self.__Unit_3d(branch_1, 288, 3, 3, 3, padding='same', name='Conv3d_4e_1b_3x3')

        branch_2 = self.__Unit_3d(x, 32, 1, 1, 1, padding='same', name='Conv3d_4e_2a_1x1')
        branch_2 = self.__Unit_3d(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4e_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4e_3a_3x3')(x)
        branch_3 = self.__Unit_3d(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4e_3b_1x1')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name='Mixed_4e')

        # Mixed 4f
        branch_0 = self.__Unit_3d(x, 256, 1, 1, 1, padding='same', name='Conv3d_4f_0a_1x1')

        branch_1 = self.__Unit_3d(x, 160, 1, 1, 1, padding='same', name='Conv3d_4f_1a_1x1')
        branch_1 = self.__Unit_3d(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_4f_1b_3x3')

        branch_2 = self.__Unit_3d(x, 32, 1, 1, 1, padding='same', name='Conv3d_4f_2a_1x1')
        branch_2 = self.__Unit_3d(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_4f_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4f_3a_3x3')(x)
        branch_3 = self.__Unit_3d(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_4f_3b_1x1')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name='Mixed_4f')

        # Downsampling (spatial and temporal)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

        # Mixed 5b
        branch_0 = self.__Unit_3d(x, 256, 1, 1, 1, padding='same', name='Conv3d_5b_0a_1x1')

        branch_1 = self.__Unit_3d(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1')
        branch_1 = self.__Unit_3d(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3')

        branch_2 = self.__Unit_3d(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1')
        branch_2 = self.__Unit_3d(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3')(x)
        branch_3 = self.__Unit_3d(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name='Mixed_5b')

        # Mixed 5c
        branch_0 = self.__Unit_3d(x, 384, 1, 1, 1, padding='same', name='Conv3d_5c_0a_1x1')

        branch_1 = self.__Unit_3d(x, 192, 1, 1, 1, padding='same', name='Conv3d_5c_1a_1x1')
        branch_1 = self.__Unit_3d(branch_1, 384, 3, 3, 3, padding='same', name='Conv3d_5c_1b_3x3')

        branch_2 = self.__Unit_3d(x, 48, 1, 1, 1, padding='same', name='Conv3d_5c_2a_1x1')
        branch_2 = self.__Unit_3d(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5c_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5c_3a_3x3')(x)
        branch_3 = self.__Unit_3d(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5c_3b_1x1')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name='Mixed_5c')

        if include_top:
            # Classification block
            x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
            x = Dropout(dropout_prob)(x)

            x = self.__Unit_3d(x, classes, 1, 1, 1, padding='same',
                        use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

            num_frames_remaining = int(x.shape[1])
            x = Reshape((num_frames_remaining, classes))(x)

            # logits (raw scores for each class)
            x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                    output_shape=lambda s: (s[0], s[2]))(x)

            if not endpoint_logit:
                x = Activation('softmax', name='prediction')(x)
        else:
            h = int(x.shape[2])
            w = int(x.shape[3])
            x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)

        inputs = img_input
        # create model
        model = Model(inputs, x, name='i3d_inception')

        # load weights
        model.load_weights(self.WEIGHTS[pretrained_weights])
        model.trainable = False

        return model

    def __TopLayer(self,input_shape, classes, dropout_prob):
        '''
        adding the top layer
        '''
        inputs = Input(shape=input_shape, name="input")
        x = Dropout(dropout_prob)(inputs)
        x = self.__Unit_3d(x, classes, 1, 1, 1, padding='same',
                    use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, classes))(x)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(x)
        x = Activation('softmax', name='prediction')(x)
        final_model = Model(inputs=inputs, outputs=x, name="i3d_top")
        return final_model
    
    def __add_top_layer(self,base_model: Model, classes: int, dropout_prob: float):
        '''
        applying the top layer
        '''
        top_layer = self.__TopLayer(base_model.output_shape[1:], classes, dropout_prob)
        x = base_model.output
        predictions = top_layer(x)
        new_model = Model(inputs=base_model.input, outputs=predictions, name="i3d_with_top")
        return new_model
    
    def __layers_freeze(self,model, leave_last=50):
        '''
        freezes the all the model layers except the last "leave_last" layers
        '''
        print("Freezing %d layers of %d in Model %s" % (len(model.layers)-leave_last, len(model.layers), model.name))
        for layer in model.layers[:-leave_last]:
            layer.trainable = False
        for layer in model.layers[-leave_last:]:
            layer.trainable = True
        return model
    
    def build_model(self,freeze=0):
        '''
        builds the i3d model and loads the kinetics pretrained weights in the model
        '''
        m_rgb = self.i3d_arch(include_top=False, pretrained_weights="rgb_imagenet_and_kinetics", dropout_prob=0.5,
                        input_shape=(64, 224, 224, 3), classes=400)

        if freeze:
            m_rgb = self.__layers_freeze(m_rgb,freeze)
            print("Freezing layers done")

        m_rgb = self.__add_top_layer(m_rgb, classes=100, dropout_prob=0.5)

        return m_rgb
    


    def __test_video(self,video_path):
        '''
        - takes the video path
        - loads the video
        - applies the necessary preprocessing of:
             - extracting exactly 64 frames
             - spatial cropping (randomly)
             - spatial cropping (from center)
             - padding
        '''

        # associate functions
        def adjust_gamma(image, gamma=1.0):
            # build a lookup table mapping the pixel values [0, 255] to
            # their adjusted gamma values
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            # apply gamma correction using the lookup table
            return cv2.LUT(image, table)
        def load_rgb_frames_from_video(video_path, start, num, resize=(256, 256)):
            vidcap = cv2.VideoCapture(video_path)

            frames = []

            total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

            vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for offset in range(min(num, int(total_frames - start))):
                success, img = vidcap.read()
                if success:
                    w, h, c = img.shape
                    if w < 226 or h < 226:
                        d = 226. - min(w, h)
                        sc = 1 + d / min(w, h)
                        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

                    if w > 256 or h > 256:
                        img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

                    ##
                    img= cv2.medianBlur(img,3)
                    img= adjust_gamma(img,1.5)
                    ##
                    img = (img / 255.) * 2 - 1

                    frames.append(img)
            return np.array(frames,dtype=np.float32)
        

        def padding(imgs,total_frames):
            if imgs.shape[0] < total_frames:
                    num_padding = total_frames - imgs.shape[0]

                    if num_padding:
                        prob = np.random.random_sample()
                        if prob > 0.5:
                            pad_img = imgs[0]
                            pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                            padded_imgs = np.concatenate([imgs, pad], axis=0)
                        else:
                            pad_img = imgs[-1]
                            pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                            padded_imgs = np.concatenate([imgs, pad], axis=0)
            else:
                    padded_imgs = imgs

            return padded_imgs
        
        
        def center_crop(frames,out_size=(224,224)):
            t,h,w,c = frames.shape
            th,tw = out_size
            i = int(np.round((h - th) / 2.))
            j = int(np.round((w - tw) / 2.))
            return frames[:, i:i+th, j:j+tw, :]
        


        #########################
        


        total_frames = 64
        cap = cv2.VideoCapture(video_path)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(nf)
        start_frame = 0
        try:
            start_f = random.randint(0, nf - total_frames - 1) + start_frame
        except ValueError:
            start_f = start_frame
        imgs = load_rgb_frames_from_video(video_path, start_f, total_frames)
        imgs = padding(imgs , total_frames)
        frames = center_crop(imgs)
        return frames

    def detect(self,video_path):
        '''
        takes the video path and returns 
            - the classified word
            - the index of the classified word
        '''
        video = self.__test_video(video_path)
        video = np.moveaxis(video,-1,0)
        output = self.tf_rep.run(np.expand_dims(video,0))
        result=np.argmax(np.max(np.array(output),axis=-1))
        return self.labels[result], result
    

    def get_labels(self,num_classes=100,path='GP_Deaf_Features/wlasl_class_list.txt'):
        '''
        returns list of labels 
        '''
        file = open(path,'r')
        all_lines = file.readlines()
        lines = all_lines[:num_classes]
        classes=list()
        for line in lines:
            label=line.split()[1]
            classes.append(label)
        return classes
os.system("rm videos/test.mp4")
os.system('ffmpeg -i {} -codec copy {}'.format("videos/test.MKV","videos/test.mp4"))
model = I3D_WLASL()
out = model.detect("videos/test.mp4")[0]
print(out)

