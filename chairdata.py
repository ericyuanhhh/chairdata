# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:59:50 2017

@author: eric yuan
"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.applications.vgg16 import preprocess_input
import numpy as np  
import os
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


model = VGG16(weights='imagenet', include_top=True)
path = 'C:\\Users\\eric yuan\\Documents\\machinelearning\\Data\\chair'
listing = os.listdir(path)

num_samples = size(listing)
training_data = []
for img_path in listing:
    img = image.load_img(path+"\\"+ img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    inp = model.input
    layer = model.layers[-2]
    output = layer.output
    
    functor = K.function([inp]+[K.learning_phase()],[output])
    layer_out = functor([x,1.])
    training_data.append(layer_out[0])
training_data = np.array(training_data)
print(training_data.shape)
training_data = np.reshape(training_data,(num_samples, 4096))

print(training_data.shape)
label=np.ones((num_samples,),dtype = int)
count = 0
for i in range(0,len(label),19):
    if i+19 < len(label):
        label[i:i+19]=count
        count = count+1
    else:
        label[i:] = count

data,Label = shuffle(training_data,label, random_state=2)
train_data = [data,Label]


#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 36
# number of epochs to train
nb_epoch = 20


(X, y) = (train_data[0],train_data[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


#neural model

model2 = Sequential([
    Dense(100, input_dim=4096),
    Activation('relu'),
    Dropout(0.5),
    Dense(36),
    Activation('softmax'),
])

#define  optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model2.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# train the model
model2.fit(X_train, Y_train, epochs=30, batch_size=32)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model2.evaluate(X_test, Y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
model2.save('chairdata_2.h5')