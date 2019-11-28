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
// Functionality: Main -- Reinforcement Learning
// 
//-----------------------------------------------------------------------------
// Change log table:
//
// Version Date           In charge       Changes applied
// 01.00.00 31/10/2019     JS              First released version
//
//===========================================================================
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data_reader

from tqdm import tqdm

from helpers import helper
from Agent import AI_Trader


#Configuración de Hiperparámetros:


def dataset_loader(stock_name):
  
  #Complete the dataset loader function
  dataset = data_reader.DataReader(stock_name, data_source="yahoo")
  
  start_date = str(dataset.index[0]).split()[0]
  end_date = str(dataset.index[-1]).split()[0]
  
  close = dataset['Close']

  return close


def state_creator(data, timestep, window_size):
  
  starting_id = timestep - window_size + 1
  
  if starting_id >= 0:
    windowed_data = data[starting_id:timestep+1]
  else:
    windowed_data = - starting_id * [data[0]] + list(data[0:timestep+1])
    
  state = []
  for i in range(window_size - 1):
    state.append(helper.sigmoid(windowed_data[i+1] - windowed_data[i]))
    
  return np.array([state])



def train(episodes, batch_size, window_size, data, trader):
    

    data_samples = len(data) - 1

    
    for episode in range(1, episodes + 1):
      
      print("Episodio: {}/{}".format(episode, episodes))
      
      state = state_creator(data, 0, window_size + 1)
      
      total_profit = 0
      trader.inventory = []
      
      for t in tqdm(range(data_samples)):
        
        action = trader.trade(state)
        
        next_state = state_creator(data, t+1, window_size + 1)
        reward = 0
        
        if action == 1: #Compra
          trader.inventory.append(data[t])
          print("AI Trader compró: ", helper.stocks_price_format(data[t]))
          
        elif action == 2 and len(trader.inventory) > 0: #Vende
          buy_price = trader.inventory.pop(0)
          
          reward = max(data[t] - buy_price, 0)
          total_profit += data[t] - buy_price
          print("AI Trader vendió: ", helper.stocks_price_format(data[t]), " Beneficio: " + helper.stocks_price_format(data[t] - buy_price) )
          
        if t == data_samples - 1:
          done = True
        else:
          done = False
          
        trader.memory.append((state, action, reward, next_state, done))
        
        state = next_state
        
        if done:
          print("########################")
          print("BENEFICIO TOTAL: {}".format(total_profit))
          print("########################")
        
        if len(trader.memory) > batch_size:
          trader.batch_train(batch_size)
          
      if episode % 1 == 0:
        trader.model.save("Checkpoints/ai_trader_{}.h5".format(episode))
        



def main():

    window_size = 10
    episodes = 20 
    
    batch_size = 32
    
    stock_name = "AAPL"
    data = dataset_loader(stock_name)
    
    trader = AI_Trader(window_size)
    trader.model.summary()
    
    train(episodes, batch_size, window_size, data, trader)

if __name__ == '__main__':
	main()