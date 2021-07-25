# Features For The Blind
We implemented several methods to support variant features to help the blind communicate better

First let's go through the features and their details:

## Emotion Recognition
As we all know how it is important to know the feelings of the person you are talking to, we implemented the Emotion Recognition model as the feelings of the person
are reflected on their faces most of the time causing a specific emotion

For this task we used the **Xception Model** mentioned in this [Paper](https://arxiv.org/pdf/1710.07557.pdf)

![Xception Model](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/pretrained_models/xception.png)

The above figure is the architecture of our used model

You can find our code in the [blind_features.py](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/blind_features.py) file in the **XceptionModel** Class

Also you can find our [Notebook](https://colab.research.google.com/drive/1r6mADG4VRlZ-jiXRqM7Hrrr_XtQTFQiL?authuser=1) which contains our training results and graphs

## Gender Classification
Because it is important to knowe the gender of the opponent for the blind user to determine what is the right way to address the other peer, we implemented the
Gender Classification Model

For this task we used the below model which mentioned in this [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W08/papers/Levi_Age_and_Gender_2015_CVPR_paper.pdf)

![Gender Classification Model](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/pretrained_models/gender_net2.png)

You can find our code in the [blind_features.py](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/blind_features.py) file in the **GenderRecognition** Class

## Age Estimation
Because it is important to know the age of the opponent who we talk to is very important to establish a bridge of communication between the peers we implemented the 
Age Estimation Model

For this task we used the **EfficentNetB3 Model** which mentioned in this [Paper](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf)

![EfficentNetB3 Model](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/pretrained_models/DEX_age2.png)

You can find our code in the [blind_features.py](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/blind_features.py) file in the **AgeDetector** Class

## Face Detection and Alignment
For face detection we used the SSD Model

![SSD Model](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/pretrained_models/res10_300x300_ssd_iter_140000.png)

For the face alignment problem we used an image processing technique for face alignment in which we first extract the eyes landmarks from the face and then we compute the angle of the line connecting the center of each eye, if the angle equal θ we apply a transformation on the face to rotate the face with angle -θ and provide a face aligned to center with no rotation. 

![Alignment](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/pretrained_models/title_image_I9k7Tog.jpeg)

We used the shape predictor provided by dlib to extract the eyes key points and used the affine transformation provided by OpenCV to apply the transformation the face. You can find the landmarks detection code in [landmarks_detection.py](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/landmarks_detection.py)

In the [face_detection.py](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/face_detection.py) file there is our code for the SSD Model usage, we used pretrained weights from caffe model

In the [face_alignment.py](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/face_alignment.py) file there is our technique explained above for alignment


