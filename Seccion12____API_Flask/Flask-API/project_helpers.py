#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:43:12 2019

@author: JuanGabriel
"""

import numpy as np
np.__version__

import tensorflow as tf
tf.__version__

from tensorflow.keras.datasets import fashion_mnist
from scipy.misc import imsave

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imsave(name="uploads/{}.png".format(i), arr=X_test[i])
