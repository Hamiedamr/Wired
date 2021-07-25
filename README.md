# WIRED MEETINGS

## Idea
**Wired Meetings** is an online platform which aims to facilitate the online communication between blind and deaf specially during interviews
by adopting a video chat app implemented on a website designed by us and also deployed on Azure with our domain.
To use the app you have to signup and create a meeting then send the link of the meeting to the other person who also needs to signup.

The application developed using multiple frameworks:

* Backend implemented using
  1. NodeJs for web services
  2. Python, TensorFlow2 and OpenCV for Deep Learning models and Computer Vision Tasks
* Frontend implemented using basic CSS & React Framewrok

This Github Repo contains many directories:

* [views](https://github.com/Hamiedamr/Wired/tree/master/views) : Contains the ejs files for chat, navbar and all the website pages
* [public](https://github.com/Hamiedamr/Wired/tree/master/public) : Contains the CSS & designg files and images
* [videos](https://github.com/Hamiedamr/Wired/tree/master/videos) : Contains the recorded videos from the user during the meeting
* [audios](https://github.com/Hamiedamr/Wired/tree/master/audios) : Contains the recorded audios to and from the user during the meeting
* [images](https://github.com/Hamiedamr/Wired/tree/master/images) : Contains the captured images from the user during the meeting
* [models](https://github.com/Hamiedamr/Wired/tree/master/models) : Contains the ejs file which handles user authentication
* [onnx-tensorflow](https://github.com/Hamiedamr/Wired/tree/master/onnx-tensorflow) : Contains ONNX library which is necessary for loading some of our Deep Learning models
* [GP_Blind_Features](https://github.com/Hamiedamr/Wired/tree/master/GP_Blind_Features) : Contains the classes and trained models that supports our features for the **Blind**
* [GP_Deaf_Features](https://github.com/Hamiedamr/Wired/tree/master/GP_Deaf_Features) : Contains the classes and trained models that supports our features for the **Deaf**

### Demo Video Link
You can find a demo video [here](https://www.youtube.com/watch?v=mSMTYfLXifg)


## Team Members & Their Contribution
| Name                                   | Contribution                                            |
| ---------------------------------------| --------------------------------------------------------|
| [Ahmed Mohammed Salah Abd El-Aziz](https://github.com/Ahmed-Salah6011)               | Blind Features Task Leader               |
| [Ibrahim Atef Abd El-Halim](https://github.com/Ibrahimatef)             | Deaf Features Task Leader                |
| [Abd El-Hamid Amr Abd El-Hamid](https://github.com/Hamiedamr)         | Deployment on Azure Task Leader & Emotion Recognition  |
| [Mark Sameh Azer](https://github.com/marksameh19)                        | Website Backend Task Leader & Azure Deployment        |
| [Fatma Elzahraa Mahmoud Esmail Mahmoud](https://github.com/fatma-elzahraa99)  | Chatting system Task Leader & Website Frontend |
| [Mariam Salah Abd El-Hamid](https://github.com/mariamsalah98) | Website Frontend Task Leader & Chatting System |

## Features

### Blind Features
More details about implementation and used model is in [here](https://github.com/Hamiedamr/Wired/blob/master/GP_Blind_Features/README.md)
* Emotion Recognition
* Gender Classification
* Age Estimation
* Text to Speech

### Deaf Features
More details about implementation and used model is in [here](https://github.com/Hamiedamr/Wired/blob/master/GP_Deaf_Features/README.md)
* Sign Language Translator
* Speech to Text

### Deployment on Website
Full integrated web platform supports:
* Login System that allows the user to register an account, login with his account and reset the password if it was ever forgotten
* Chatting room that allows *two* users to join the room and have a private conversation between each other
* Video and audio calling that allows the users *within* the chatting room to communicate with each other using audio and video not only messages

### Deployment on Microsoft Azure
Deployed on **Microsoft Azure** with our domain with our reserved server so that anyone around the world can use our services

## Installation
* No need to install anything just open our link and enjoy!
* Supported Browser : Chrome, Microsoft Edge, Mozilla Firefox...etc

## Future Work ISA
In the future we aim to support the following features on our platform ISA:
* Chatbot to help the user with his experience throughout the website and answer his/her questions
* Arabic Sign Language Translator instead of our American Sign Language Translator which we used due to the lack of the ARSL dataset 
