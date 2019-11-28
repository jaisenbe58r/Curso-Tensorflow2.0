# -*- coding: utf-8 -*-
"""
//===========================================================================
// JAIME SENDRA BERENGUER
// TECH TRAININGS - MACHINE LEARNING
//-----------------------------------------------------------------------------
// Autor: JS 
// Revisado: JS 
//-----------------------------------------------------------------------------
// Library:       -
// Tested with:   CPU CORE i7 16Gb
// Engineering:   -
// Restrictions:  -
// Requirements:  Python 3.6
// Functionality: Helpers -- Reinforcement Learning
// 
//-----------------------------------------------------------------------------
// Change log table:
//
// Version Date           In charge       Changes applied
// 01.00.00 31/10/2019     JS              First released version
//
//===========================================================================
"""

import math

# Función Sigmoide para normalizar los datos

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# Formateado de los resultados
  
def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))