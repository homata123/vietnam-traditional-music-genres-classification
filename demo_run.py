from numpy import loadtxt
from keras.models import load_model


import sklearn
import tensorflow as tf

import keras as k


from tensorflow.keras.layers import LeakyReLU


#
import numpy as np
import csv
import librosa
import os
import time
SOUND_SAMPLE_LENGTH = 30000
HAMMING_SIZE = 1000
HAMMING_STRIDE = 500
DIR = 'D:\MGC\Data1\songtest'
#




#
labels = {
    'Cailuong'     :   0,
    'Catru' :   1,
    'Chauvan'   :   2,
    'Cheo'    :   3,
    'Xam'      :   4,
    'Ysample': 5
}

def getFeatures(filepath):
    # y, sr = librosa.load(filepath)
    # mfcc = librosa.feature.mfcc(y,sr)
    #featuresArray = []
    features = []
    y, sr = librosa.load(filepath)
    y=y[0:617000]
    #Add hamming window overlap code
    spectral_centroid = librosa.feature.spectral_centroid(y,sr)
    mean_spectral_centroid = np.mean(spectral_centroid, axis=1)
    std_spectral_centroid = np.std(spectral_centroid, axis=1)
    features.append(mean_spectral_centroid[0])
    features.append(std_spectral_centroid[0])

    chroma_stft=librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma_stft=np.mean(chroma_stft,axis=1)
    std_chroma_stft=np.std(chroma_stft,axis=1)
    features.append(mean_chroma_stft[0])
    features.append(std_chroma_stft[0])

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    mean_spectral_bandwidth = np.mean(spectral_bandwidth, axis=1)
    std_spectral_bandwidth = np.std(spectral_bandwidth, axis=1)
    features.append(mean_spectral_bandwidth[0])
    features.append(std_spectral_bandwidth[0])

    spectral_rolloff = librosa.feature.spectral_rolloff(y, sr)
    mean_spectral_rolloff = np.mean(spectral_rolloff, axis=1)
    std_spectral_rolloff = np.std(spectral_rolloff, axis=1)
    features.append(mean_spectral_rolloff[0])
    features.append(std_spectral_rolloff[0])

    zcr = librosa.feature.zero_crossing_rate(y, sr)
    mean_zcr = np.mean(zcr, axis=1)
    std_zcr = np.std(zcr, axis=1)
    features.append(mean_zcr[0])
    features.append(std_zcr[0])



    #

    rmse = librosa.feature.rms(y)
    mean_rmse = np.mean(rmse, axis=1)

    low_energy = len(rmse[np.where(rmse < mean_rmse)])
    features.append(mean_rmse[0])
    features.append(low_energy)

    return np.asarray(features)




X=[]
Y=[]
F=[]
print('Processing....')
for subdir, dirs, files in os.walk(DIR):
    #print subdir, files
    files.sort()

    for file in files:
        temp = str(file).split('.')
        filepath = os.path.join(DIR, temp[0], file)
        start = time.time()
        if not str(filepath).__contains__('.DS_Store'):
            mfcc = getFeatures(filepath)

            label = labels[temp[0]]
            X.append(mfcc)
            Y.append(label)
            F.append(file)
# np.savetxt('files.npy',F,delimiter=',')
np.savetxt('D:\MGC\Data1\datatrain1.txt',X,delimiter=',')
np.savetxt('D:\MGC\Data1\labeltrain1.txt', Y,delimiter=',')


# np.savetxt('files.npy',F,delimiter=',')
#loading dataset
data = np.loadtxt('D:\MGC\Data1\datatrain1.txt',delimiter=',',dtype=float)
labels = np.loadtxt('D:\MGC\Data1\labeltrain1.txt',delimiter=',',dtype=float)
labels.astype(int)
# load model
model = load_model('D:\MGC\Data1\model5best.h5')
# summarize model.
#model.summary()




#Transform to scalar
scalar = sklearn.preprocessing.StandardScaler()

# Generate the train set with the rest of the data.
train_data = data

#Transform to scalar
train_data = scalar.fit_transform(train_data)

Z = np.array(train_data[10]).reshape(1, -1)
print(np.argmax(model.predict(Z), axis=-1))
#print(model.predict(Z))

print("Probablity of Cai luong:",model.predict(Z)[0][0])
print("Probablity of Ca tru:",model.predict(Z)[0][1])
print("Probablity of Chau van:",model.predict(Z)[0][2])
print("Probablity of Cheo:",model.predict(Z)[0][3])
print("Probablity of Xam:",model.predict(Z)[0][4])



if model.predict(Z)[0][0]==model.predict(Z)[0][np.argmax(model.predict(Z), axis=-1)]:
    print('Bai hat thuoc the loai Cai luong')
if model.predict(Z)[0][1]==model.predict(Z)[0][np.argmax(model.predict(Z), axis=-1)]:
    print('Bai hat thuoc the loai Ca tru')
if model.predict(Z)[0][2]==model.predict(Z)[0][np.argmax(model.predict(Z), axis=-1)]:
    print('Bai hat thuoc the loai Chau van')
if model.predict(Z)[0][3]==model.predict(Z)[0][np.argmax(model.predict(Z), axis=-1)]:
    print('Bai hat thuoc the loai Cheo')
if model.predict(Z)[0][4]==model.predict(Z)[0][np.argmax(model.predict(Z), axis=-1)]:
    print('Bai hat thuoc the loai Xam')