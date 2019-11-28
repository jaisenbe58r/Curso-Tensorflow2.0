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
##  Configurar el entorno de producción
## --------------------------------------------------------------
# Exportar el MODEL_DIR a las variables de entorno
MODEL_DIR = "model/"
version = 1
os.environ['MODEL_DIR'] = os.path.abspath(MODEL_DIR)



# Examinar modelo guardado
saved_model_cli show --dir {export_path} --all

# Ejecutar la API REST de Tensorflow Serving

%%bash --bg
nohup tensorflow_model_server /
--rest_api_port=8000 --model_name=cifar10 --model_base_path="${MODEL_DIR}" >server.log 2>&1