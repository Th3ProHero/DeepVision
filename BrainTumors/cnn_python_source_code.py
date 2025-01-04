# Import required libraries / Importar librerías necesarias
paths=["/kaggle/input/brain-tumor-mri-dataset/Training/",
    "/kaggle/input/brain-tumor-mri-dataset/Testing/"]

import os 
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display an image / Función para mostrar una imagen
def imshow(img):
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    ax.imshow(img,cmap='gray')

# Get possible labels from the dataset / Obtener etiquetas posibles del dataset
possible_labels=os.listdir(paths[0])

possible_labels

# Load and preprocess images / Cargar y preprocesar las imágenes
data=[]

for i,path in enumerate(paths):
    for label_int, label_string in enumerate(possible_labels):
        for filename in glob.glob(path+label_string+'/*.jpg'):
            img=cv2.imread(filename)  # Read the image / Leer la imagen
            data.append( [label_int,cv2.resize(img,(320,320))] )  # Resize and store / Redimensionar y almacenar

len(data)

# Shuffle and split data into train, validation, and test sets / Barajar y dividir los datos en conjuntos de entrenamiento, validación y prueba
import random
random.Random(0).shuffle(data) 

x_train=[]
y_train=[]

x_val=[]
y_val=[]

x_test=[]
y_test=[]

for i, sample in enumerate(data):
    label=sample[0]
    img=sample[1]
    if i<= 0.8*len(data):  # 80% for training / 80% para entrenamiento
        x_train.append(img)
        y_train.append(label)
    elif i>0.8*len(data) and i<=0.9*len(data):  # 10% for validation / 10% para validación
        x_val.append(img)
        y_val.append(label)
    else:  # 10% for testing / 10% para prueba
        x_test.append(img)
        y_test.append(label)

# Convert lists to NumPy arrays / Convertir listas a arreglos NumPy
x_train=np.array(x_train)
x_val=np.array(x_val)
x_test=np.array(x_test)

y_train=np.array(y_train)
y_val=np.array(y_val)
y_test=np.array(y_test)

# Check the shape of the test set / Verificar la forma del conjunto de prueba
x_test.shape

# Display the first image in the training set / Mostrar la primera imagen del conjunto de entrenamiento
imshow(x_train[0])

# Display the corresponding label / Mostrar la etiqueta correspondiente
y_train[0]

