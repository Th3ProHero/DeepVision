Project: Brain Tumor Detection with Kaggle and Convolutional Neural Networks (CNN)

Description

This project aims to develop a deep learning model to detect brain tumors in magnetic resonance imaging (MRI) using a Kaggle dataset. A convolutional neural network (CNN) model was implemented to classify the images as "with tumor" or "without tumor."

The project was developed in a Kaggle Notebook environment, leveraging its resources and peculiarities, such as predefined data paths and GPU support.

Project Features

Dataset:

The dataset used contains 10,000 MRI images divided into training and test folders.

The folder and file structure is predefined in Kaggle.

Preprocessing:

Reading images with OpenCV (cv2).

Resizing images to 320x320 pixels for uniformity.

Normalizing pixel values between 0 and 1.

CNN Model:

Designed with TensorFlow/Keras.

Simple architecture: convolutional, pooling, and dense layers.

Training:

Dataset split:

80% training

10% validation

10% testing

A GPU was used to accelerate training.

Evaluation:

Metrics such as accuracy, precision, and recall.

Differences When Working in Kaggle

Predefined Paths:

In Kaggle, datasets are automatically mounted in folders under /kaggle/input/. This simplifies data access:

python
paths = ["/kaggle/input/brain-tumor-mri-dataset/Training/",
         "/kaggle/input/brain-tumor-mri-dataset/Testing/"]
Non-native Functions and Libraries:

Kaggle provides direct access to common libraries such as TensorFlow, NumPy, and Matplotlib, avoiding the need to manually install dependencies.

GPU Usage:

Kaggle facilitates GPU resource allocation:

In the environment, select GPU in "Settings".

TensorFlow automatically detects the GPU:

python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
Storing Results:

It is common to save trained models or intermediate results in the output directory:

python
model.save("/kaggle/working/model.h5")
Requirements

Dependencies

The main libraries used include:

tensorflow

numpy

matplotlib

opencv-python

Local Installation

If you want to run the project outside of Kaggle:

Download the dataset from Kaggle.

Install the dependencies:

sh
pip install tensorflow numpy matplotlib opencv-python
Adjust the data paths according to your local environment.

Project Execution

Preprocessing

Load and preprocess the images:

python
import os
import cv2
import numpy as np

data = []
for path in paths:
    for label in os.listdir(path):
        for filename in glob.glob(os.path.join(path, label, '*.jpg')):
            img = cv2.imread(filename)
            data.append([label, cv2.resize(img, (320, 320))])
Training

Define and train the model:

python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(320, 320, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
Evaluation

Evaluate the model on the test set:

python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
Results

Average accuracy: 92%

Training time: <5 minutes (with GPU in Kaggle)

Final Notes

This project showcases the advantages of using Kaggle for rapid and effective development of deep learning models. It leverages the native functionalities of the environment to simplify the workflow, understanding the workings of a CNN, and the computational complexity it requires.
