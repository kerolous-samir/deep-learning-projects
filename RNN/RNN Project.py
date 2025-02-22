#Imports
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

#Functions
def scaled_test_func(scaled_test):
    X_test = []
    
    for i in range(60,80):
        X_test.append(scaled_test[i-60:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    return X_test

def scaled_train_func(scaled_train):
    X_train = []
    y_train = []
    for i in range(60,1258):
        X_train.append(scaled_train[i-60:i,0])
        y_train.append(scaled_train[i,0])
    X_train , y_train = np.array(X_train) , np.array(y_train)
    X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    return X_train , y_train

#Import Data
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train['Open'].values.reshape(-1,1)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
stock_price = dataset_test['Open'].values.reshape(-1,1)

#Concat Data
dataest_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

#Scaling Data
sc = MinMaxScaler()
X_train , y_train = scaled_train_func(sc.fit_transform(training_set))
scaled_stock_price = sc.transform(stock_price).reshape(-1,1)
X_test = scaled_test_func(sc.transform(dataest_total[len(dataest_total)-len(dataset_test)-60:].values.reshape(-1,1)))

#Model
regressor = Sequential()
regressor.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(50))
regressor.add(Dropout(0.2))
regressor.add(Dense(1))
regressor.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
regressor.fit(X_train,y_train,batch_size=32,epochs=100) 

#Save Model
regressor.save("RNN_model.h5")

#Predictions
prediction = regressor.predict(X_test)
prediction_stock_price = sc.inverse_transform(prediction)

#Evaluation
print("\nScaled Mean Absolute Error: ",mean_absolute_error(scaled_stock_price,prediction))
print("\nScaled Mean Squared Error: ",mean_squared_error(scaled_stock_price,prediction))
print("\nScaled Root Mean Squared Error: ",(mean_squared_error(scaled_stock_price,prediction))*(1/2))
print("\nReal Mean Absolute Error: ",mean_absolute_error(stock_price,prediction_stock_price))
print("\nReal Mean Squared Error: ",mean_squared_error(stock_price,prediction_stock_price))
print("\nReal Root Mean Squared Error: ",(mean_squared_error(stock_price,prediction_stock_price))*(1/2))