# Deaf Features
Understanding the language of the opponent we talk to is very important to make the communication more effective, unfortunately the deaf cannot be understood because basically they speak a different language, thats why we supported a Sign Language Translator for the deaf

## Sign Language Translator

There are two main approaches for Sign Language Translation task:
* holistic visual appearance-based approach
* 2D human pose-based approach

We adopted the *holistic visual appearance-based approach* because it is the one which have a faster results in real-time

The main architecture of this approach is in the figure below

![Conv3D](https://github.com/Hamiedamr/Wired/blob/master/GP_Deaf_Features/pretrained_models/conv3d.png)

This approach treats the video as Tensor of *four* dimensions (F x W x H x C) where F is the number of video frames, W & H are each frame's width and height and C is the number of frame's channels

Our Model is the **I3D** Model which is the Implementation of the [WLASL Paper](https://dxli94.github.io/WLASL/)

![I3D](https://github.com/Hamiedamr/Wired/blob/master/GP_Deaf_Features/pretrained_models/i3d.png)

Before Training the data are passed to a certain pipeline to preprocess it and get it ready for training

![Pipeline](https://github.com/Hamiedamr/Wired/blob/master/GP_Deaf_Features/pretrained_models/pipe.png)

You can find our model's implementation and training details in the [Notebook](https://colab.research.google.com/drive/1L2sC7nyvMlav0AeJA9iZoZEkE09dJKpF?usp=sharing)

The model's code which is used for prediction and testing and also the data pipline is in the [deaf_features.py](https://github.com/Hamiedamr/Wired/blob/master/GP_Deaf_Features/deaf_features.py) file in the **I3D_WLASL** Class
