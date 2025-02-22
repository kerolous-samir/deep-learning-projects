#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from minisom import MiniSom
from matplotlib.pylab import bone,pcolor,colorbar,plot,show
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report,confusion_matrix,explained_variance_score

#Data
df = pd.read_csv('Credit_Card_Applications.csv')
customers = df.drop('CustomerID',axis=1).values
X = df.drop('Class',axis=1).values
y = df['Class'].values

#Scale (Normalization)
sc = MinMaxScaler()
X = sc.fit_transform(X)

#Unsupervised Model
som = MiniSom(x=10,y=10,input_len=15,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X,num_iteration=100,verbose=1)

#Discover fraudulent customers 
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i , x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,w[1] + 0.5,markers[y[i]],markeredgecolor=colors[y[i]],markeredgewidth=2,markerfacecolor='None',markersize=10)
show()


#Export Frauds
mapping = som.win_map(X)
fraud = np.concatenate((mapping[(7,3)],mapping[(4,4)]),axis=0)
frauds = sc.inverse_transform(fraud)

#Scale Customers
scaler = StandardScaler()
customers = scaler.fit_transform(customers)

#Supervised Model
ann = Sequential()
ann.add(Dense(2,activation='relu'))
ann.add(Dense(1,activation='sigmoid'))
ann.compile(optimizer='adam',loss='binary_crossentropy',metrices=['accuracy'])
ann.fit(customers,is_fraud,batch_size=1,epochs=2)

#Prediction
pred = (ann.predict(customers) > 0.5).astype('int32')

#Evaluation
print(classification_report(y,pred))
print(confusion_matrix(y,pred))
print(explained_variance_score(y,pred))