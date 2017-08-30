# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:24:43 2017

@author: eric yuan
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import load_model
from matplotlib import pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras import backend as K
model = VGG16(weights='imagenet', include_top=True)

img_path = 'C:\\Users\\eric yuan\\Documents\\machinelearning\\Data\\chair\\images-301.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
inp = model.input
layer = model.layers[-2]
output = layer.output
    
functor = K.function([inp]+[K.learning_phase()],[output])
layer_out = functor([x,1.])
training_data = layer_out[0]
print(training_data)

model2 = load_model('chairdata_2.h5')

features = model2.predict(training_data)
features = np.reshape(features,36)
print(features)
x_dim = np.arange(36)
plt.plot(x_dim,features)
plt.show()