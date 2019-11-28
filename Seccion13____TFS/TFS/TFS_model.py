# -*- coding: utf-8 -*-

import os
import json
import random
import requests
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

print(tf.__version__)

## --------------------------------------------------------------
## Pre procesado de los datos
## --------------------------------------------------------------
# Cargar el dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_names = ['avión', 'coche', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

# Normalizacion de imagenes
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train.shape

## --------------------------------------------------------------
##  Definir el modelo
## --------------------------------------------------------------
# Modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='Adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['sparse_categorical_accuracy'])

## --------------------------------------------------------------
## Entrenar modelo
## --------------------------------------------------------------
model.fit(X_train, 
          y_train, 
          batch_size=128, 
          epochs=1)

## --------------------------------------------------------------
## Evaluar modelo
## --------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("La precisión del modelo es de {} %".format(test_accuracy*100.0))

## --------------------------------------------------------------
## Guardar el modelo para subir a producción
## --------------------------------------------------------------
MODEL_DIR = "model/"
version = 1

export_path = os.path.join(MODEL_DIR, str(version))
print(export_path)

# Guardar el modelo para TensorFlow Serving
tf.saved_model.simple_save(tf.keras.backend.get_session(), export_dir=export_path, 
                           inputs={"input_image":model.input}, 
                           outputs={t.name:t for t in model.outputs})















