# -*- coding: utf-8 -*-
"""
Created on Thu Dec 9 11:51:25 2021

@author: Dhia
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Load the data 
from google.colab import files
uploaded = files.upload()

df = pd.read_csv('BTC_data.csv')

#Visualize the price history
plt.figure(figsize=(16,8))
plt.title('Price History')
plt.plot(df['Price'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price USD ($)', fontsize=18)
plt.show()

data = df.filter(['Price'])
dataset = data.values

training_data_len = math.ceil(len(dataset)*.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []

for i in range(224, len(train_data)):
  x_train.append(train_data[i-224:i,0])
  y_train.append(train_data[i,0])
  if i <= 225:
    print(x_train)
    print(y_train)
    print() 
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=30)

#Create a new array containing scaled values from index 1062 to 1522
test_data = scaled_data[training_data_len - 898: , :]
#Create the datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(898, len(test_data)):
  x_test.append(test_data[i-898:i, 0])
x_test = np.array(x_test)

#Reshape the data to 3dim.
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days', fontsize = 18)
plt.ylabel('Price USD ($)', fontsize = 18)
plt.plot(train['Price'])
plt.plot(valid[['Price' ,'Predictions']])
plt.legend(['Train', 'Real', 'Predictions'], loc='lower right')
plt.show()