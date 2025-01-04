Proyecto: Detección de Tumores Cerebrales con Kaggle y Redes Neuronales Convolucionales (CNN)

Descripción

Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje profundo para detectar tumores cerebrales en imágenes de resonancia magnética (MRI) utilizando un conjunto de datos de Kaggle. Se implementó un modelo de red neuronal convolucional (CNN) para clasificar las imágenes como "con tumor" o "sin tumor".

El proyecto fue desarrollado en un entorno Kaggle Notebook, aprovechando sus recursos y peculiaridades, como las rutas predefinidas para los datos y el soporte para GPUs.

Características del Proyecto

Conjunto de Datos:

El dataset utilizado contiene 10,000 imágenes MRI divididas en carpetas de entrenamiento y prueba.

La estructura de carpetas y archivos está predefinida en Kaggle.

Preprocesamiento:

Lectura de imágenes con OpenCV (cv2).

Redimensionamiento de imágenes a 320x320 píxeles para uniformidad.

Normalización de valores de píxeles entre 0 y 1.

Modelo CNN:

Diseñado con TensorFlow/Keras.

Arquitectura sencilla: capas convolucionales, de pooling y densas.

Entrenamiento:

División del conjunto de datos:

80% entrenamiento

10% validación

10% prueba

Se utilizó una GPU para acelerar el entrenamiento.

Evaluación:

Métricas como exactitud, precisión y recall.

Diferencias al Trabajar en Kaggle

1. Rutas Predefinidas

En Kaggle, los datasets se montan automáticamente en carpetas bajo /kaggle/input/. Esto simplifica el acceso a los datos:

paths = ["/kaggle/input/brain-tumor-mri-dataset/Training/",
         "/kaggle/input/brain-tumor-mri-dataset/Testing/"]

2. Funciones y Librerías no Nativas

Kaggle ofrece acceso directo a librerías comunes como TensorFlow, NumPy y Matplotlib, lo que evita la necesidad de instalar dependencias manualmente.

3. Uso de GPU

Kaggle facilita la asignación de recursos como GPUs:

En el entorno, selecciona GPU en "Settings".

TensorFlow detecta la GPU automáticamente:

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

4. Almacenamiento de Resultados

Es común guardar los modelos entrenados o resultados intermedios en el directorio de salida:

model.save("/kaggle/working/model.h5")

Requisitos

Dependencias

Las principales librerías utilizadas incluyen:

tensorflow

numpy

matplotlib

opencv-python

Instalación Local

Si deseas ejecutar el proyecto fuera de Kaggle:

Descarga el dataset desde Kaggle.

Instala las dependencias:

pip install tensorflow numpy matplotlib opencv-python

Ajusta las rutas a los datos según tu entorno local.

Ejecución del Proyecto

1. Preprocesamiento

Carga y preprocesa las imágenes:

import os
import cv2
import numpy as np

data = []
for path in paths:
    for label in os.listdir(path):
        for filename in glob.glob(os.path.join(path, label, '*.jpg')):
            img = cv2.imread(filename)
            data.append([label, cv2.resize(img, (320, 320))])

2. Entrenamiento

Define y entrena el modelo:

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

3. Evaluación

Evalúa el modelo en el conjunto de prueba:

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Exactitud: {accuracy:.2f}")

Resultados

Exactitud promedio: 92%

Tiempo de entrenamiento: <5 minutos (con GPU en Kaggle)

Notas Finales

Este proyecto muestra las ventajas de utilizar Kaggle para desarrollo rápido y efectivo de modelos de aprendizaje profundo. Aprovecha las funcionalidades nativas del entorno para simplificar el flujo de trabajo y entendiendo el funcionamiento de una CNN y la complejidad de calculo que requiere