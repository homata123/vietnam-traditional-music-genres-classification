'''
Compute Spectral Centroid, Spectral Roll Off, Zero crossing rate
Make Gaussian assumption and save mean and standard deviation for each feature
Save low energy as fraction of frames having rmse lower than the avg rmse of the audio file
'''
import numpy as np
import csv
import librosa
import os
import time
SOUND_SAMPLE_LENGTH = 30000
HAMMING_SIZE = 1000
HAMMING_STRIDE = 500
DIR = 'D:\MGC\Data1\data5genres'

labels = {
    'CaiLuong'     :   0,
    'Catru' :   1,
    'Chauvan'   :   2,
    'Cheo'    :   3,
    'Xam'      :   4
    #'metal'     :   6,
    #'pop'       :   7,
    #'reggae'    :   8,
    #'rock'      :   9,
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

    mfc20 = librosa.feature.mfcc(y, sr)
    mean_mfc20 = np.mean(mfc20, axis=1)
    std_mfc20 = np.std(mfc20, axis=1)
    features.append(mean_mfc20[0])
    features.append(std_mfc20[0])
    
    mfc15 = librosa.feature.mfcc(y, sr,n_mfcc=15)
    mean_mfc15 = np.mean(mfc15, axis=1)
    std_mfc15 = np.std(mfc15, axis=1)
    features.append(mean_mfc15[0])
    features.append(std_mfc15[0])
    
    mfc10 = librosa.feature.mfcc(y, sr,n_mfcc=10)
    mean_mfc10 = np.mean(mfc10, axis=1)
    std_mfc10 = np.std(mfc10, axis=1)
    features.append(mean_mfc10[0])
    features.append(std_mfc10[0])
    
    mfc5 = librosa.feature.mfcc(y, sr,n_mfcc=5)
    mean_mfc5 = np.mean(mfc5, axis=1)
    std_mfc5 = np.std(mfc5, axis=1)
    features.append(mean_mfc5[0])
    features.append(std_mfc5[0])
    

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
for subdir, dirs, files in os.walk(DIR):
    #print subdir, files
    files.sort()
    print(files)
    for file in files:
        temp = str(file).split('.')
        filepath = os.path.join(DIR, temp[0], file)
        start = time.time()
        if not str(filepath).__contains__('.DS_Store'):
            mfcc = getFeatures(filepath)
            print(file+' processed'+str(time.time()-start))
            label = labels[temp[0]]
            X.append(mfcc)
            Y.append(label)
            F.append(file)
# np.savetxt('files.npy',F,delimiter=',')
np.savetxt('D:\MGC\Data1\datatrain3.txt',X,delimiter=',')
np.savetxt('D:\MGC\Data1\labeltrain3.txt', Y,delimiter=',')