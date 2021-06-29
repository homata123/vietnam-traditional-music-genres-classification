# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 23:03:20 2021

@author: ADMIN
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import linear_model

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
import sklearn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
from numpy.random import permutation, randint
from keras.models import model_from_json
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from statistics import mode
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from tensorflow.keras.layers import LeakyReLU
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import math
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random
import librosa, IPython
import librosa.display as lplt
import os
seed = 12
np.random.seed(seed)
#loading dataset
data1 = np.loadtxt('D:\MGC\Data1\datatrain3.txt',delimiter=',',dtype=float)
labels = np.loadtxt('D:\MGC\Data1\labeltrain3.txt',delimiter=',',dtype=float)
labels.astype(int)

data2=data1[:,:10]
data3=data1[:,18:]
data=np.hstack((data2,data3))
#data=data1

# Randomly shuffle the index of nba.
randomize = np.arange(len(data))
np.random.shuffle(randomize)
data = data[randomize]
labels = labels[randomize]
scalar = sklearn.preprocessing.StandardScaler()
data=scalar.fit_transform(data)
# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = int(math.floor(len(data)/10))
# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
test_data = data[0:test_cutoff]

#Transform to scalar

##test_data = scalar.fit_transform(test_data)

test_label = labels[0:test_cutoff]
# Generate the train set with the rest of the data.
train_data = data[test_cutoff:]
train_label = labels[test_cutoff:]
#Transform to scalar
##train_data = scalar.fit_transform(train_data)
#model building
import tensorflow as tf
print("TF version:-", tf.__version__)
import keras as k
tf.random.set_seed(seed)

leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)

#CNN
ACCURACY_THRESHOLD = 1.1
class myCallback(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > ACCURACY_THRESHOLD):
            print("\n\nStopping training as we have reached %2.2f%% accuracy!" %(ACCURACY_THRESHOLD*100))
            self.model.stop_training = True

def trainModel(model, epochs, optimizer):
    batch_size = 32
    callback = myCallback()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy'
    )
    return model.fit(train_data, train_label,  epochs=epochs,
                     batch_size=batch_size, callbacks=[callback])


#model_1 = k.models.Sequential([
#    k.layers.Dense(512, activation='relu', input_shape=(train_data.shape[1],)),
#    k.layers.Dense(256, activation='relu'),
#    k.layers.Dense(128, activation=leaky_relu),
#    k.layers.Dense(64,activation=leaky_relu),

#    k.layers.Dense(5, activation='softmax'),
#])

model_1=Sequential()
model_1.add(layers.Dense(512,input_shape=(train_data.shape[1],),activation='relu'))
model_1.add(layers.Dense(256,activation='relu'))
model_1.add(layers.Dense(128)) # no activation here
model_1.add(layers.LeakyReLU(alpha=0.01))
model_1.add(layers.Dense(64)) # no activation here
model_1.add(layers.LeakyReLU(alpha=0.01))
model_1.add(layers.Dense(5,activation='softmax'))



print(model_1.summary())
model_1_history = trainModel(model=model_1, epochs=150, optimizer='adam')


# serialize model to JSON
#model_1.save("model5best.h5")
#CNN best score=0.90 at 1000 epochs
test_loss, test_acc  = model_1.evaluate(test_data, test_label, batch_size=128)
print("The CNN test Loss is :",test_loss)
print("\nThe Best CNN test Accuracy is :",test_acc)


#Z=np.array(test_data[i]).reshape(1,-1)


##for i in range(0,20):
    #Y=test_data[i]
    #Y=np.array(Y).reshape(1,-1)
    #print(print("Label predicted: ",model_1.predict_classes(Y)))
    #if model_1.predict_classes(Y)==1:
        #print(Y)
        #print("index: ",i)





