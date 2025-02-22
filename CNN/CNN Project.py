#Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

#Create Model
train_idg = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
train_import = train_idg.flow_from_directory('training_set',target_size=(64,64),batch_size=32,class_mode='binary')
test_idg = ImageDataGenerator(rescale=1./255)
test_import = train_idg.flow_from_directory('test_set',target_size=(64,64),batch_size=32,class_mode='binary')
cnn = Sequential()
callback = EarlyStopping(monitor='val_loss',mode='min',patience=5)
cnn.add(Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(MaxPool2D(pool_size=2,strides=2))
cnn.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(MaxPool2D(pool_size=2,strides=2))
cnn.add(Flatten())
cnn.add(Dense(60,activation='relu'))
cnn.add(Dropout(0.2))
cnn.add(Dense(1,activation='sigmoid'))
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn.fit(x=train_import,epochs=25,batch_size=256,callbacks=[callback],validation_data=test_import)

#Testing Machine
test = image.load_img('cat_or_dog_1.jpg',target_size=(64,64))
test = image.img_to_array(test)
test = np.expand_dims(test,axis=0)
pred = cnn.predict(test)
train_import.class_indices
if int(pred) == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)