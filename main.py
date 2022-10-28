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
df = web.DataReader('CCL', data_source='yahoo', start='2012-01-01', end='2022-09-18')

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
training_data_len = math.ceil(len(dataset) * .9)


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

model.add(LSTM(75,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(75,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(x_train, y_train, batch_size=1, epochs=1)

#create testing dataset
#create a new array containeing scaled values from 1543 to 2003
test_data = scaled_data[training_data_len - 60:,:]
#create data sets x_test, y_test
x_test = []
y_test = dataset[training_data_len:, :]


for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

#convert the data to numpy array
x_test = np.array(x_test)

#reshape data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

#get models predicted values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get the RMSE
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))

print(rmse)

#plot data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('CXlose Price USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val', 'Predictions'], loc = 'lower right')
#plt.show()


quote = web.DataReader('CCL', data_source='yahoo', start='2012-01-01', end='2022-08-18')

new_df = quote.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
pred_price = model.predict(X_test)

pred_price = scaler.inverse_transform(pred_price)

quote2 = web.DataReader('CCL', data_source='yahoo', start='2022-08-19', end='2022-08-19')


pct_diff = 100 * (abs(pred_price - quote2['Close'].values)/((pred_price + quote2['Close'].values)/2))

print('Predicted Value: ', pred_price, ' Actual Price: ', quote2['Close'].values)
print('Pct Diff: ', pct_diff)