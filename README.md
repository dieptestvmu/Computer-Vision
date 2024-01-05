Eye Tracking Mouse Control	
Introduction
This software is developed to detect the eye gaze direction and utilize that information to control the mouse on a computer. The application is capable of recognizing actions such as up, down, left, right, center, and eye blink.

Requirements
Programming language: Python
Libraries: OpenCV, Tensorflow, Mediapipe, Numpy, ...

Usage
Install the necessary libraries:
pip install opencv-python tensorflow mediapipe numpy

Create and train the models:

Prepare the dataset following the structure:
Dataset/
├── left_eye/
|   ├── up/
|   ├── down/
|   ├── left/
|   ├── right/
|   ├── center/
|   ├── blink/
├── right_eye/
    ├── blink/
    ├── noblink/
