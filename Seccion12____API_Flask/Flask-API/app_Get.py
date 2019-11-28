#Paso 1: Importar todas las dependencias del proyecto

import os
import requests
import numpy as np
import tensorflow as tf
import flask
#Hacer el downgrade a la versión 1.1.0 con pip install scipy==1.1.0
from scipy.misc import imread, imsave
from flask import Flask, request, jsonify

print(tf.__version__)

#Paso 2: Cargar el modelo pre entrenado
with open('fashion_model_flask.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

# cargar los pesos en el modelo
model.load_weights("fashion_model_flask.h5")

#Paso 3: Crear la API con Flask
#Crear una aplicación de Flask
app_get = Flask(__name__)

#Definir la función de clasificación de imágenes
@app_get.route("/api_get/v1/<string:img_name>", methods=["GET"])
def classify_image(img_name):
    #Definir donde se encuentra la carpeta de imágenes
    upload_dir = "uploads/"
    #Cargar una de las imágenes de la carpeta
    image = imread(upload_dir + img_name)
    
    #Definir la lista de posibles clases de la imagen
    classes = ["Camiseta", "Pantalón", "Sudadera", "Vestido", "Abrigo", 
               "Sandalia", "Jersey", "Zapatilla", "Bolsa", "Botas"]

    #Hacer la predicción utilizando el modelo pre entrenado
    prediction = model.predict([image.reshape(1, 28*28)])

    #Devolver la predicción al usuario
    return jsonify({"object_identified":classes[np.argmax(prediction[0])]})

#Iniciar la aplicación de Flask
app_get.run(port=5001, debug=False)
