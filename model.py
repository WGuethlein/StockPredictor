import math
from ssl import SSLSyscallError
import pandas_datareader as web
import dataframe_image as dfi
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#get stock quote
df = web.DataReader('CCL', data_source='yahoo', start='2007-01-01', end='2022-08-15')

#size x,y
#print(df.shape)
#print head of datafram
#print(df.tail())
#tail of df
#print(df.tail())


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
#plt.show()

#Create a new df with only close column
data = df. filter(['Close'])
#convert df to numpy array
dataset = data.values


#get num rows
training_data_len = math.ceil(len(dataset) * .8)


#print(training_data_len)

#scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#print scaled data
#print(scaled_data)

#create training dataset
#scaled training data set
train_data = scaled_data[0:training_data_len , :]
#split it x_train, y_train

x_train = []
y_train = []


for i in range(60, len(train_data)):
    #x trained data gets 0-59
    x_train. append(train_data[i-60:i,0])
    #y_train get #60
    y_train.append(train_data[i,0])


#convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)


model = Sequential()

model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(x_train, y_train, batch_size=1, epochs=1)

model.save('model')