"""Created on Thu Aug 20 19:49:05 2020.

@author: Veljko
"""
from model import *
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
# 3rd party libs
from python_speech_features import mfcc
from python_speech_features import delta
# built-in libs
import scipy.io.wavfile as wav

# MFCC parameteres
winlen = 0.025  # length of analysis window in seconds
winstep = 0.01  # the step between successive windows in seconds
nfilt = 40  # the number of filters in the filterbank
lowfreq = 133.333  # lowest band edge of mel filters in Hz
highfreq = 6855.4976  # highest band edge of mel filters in Hz
ncep = 13  # the number of cepstrum coefficients
test_audio_path = 'D:\\obama_dataset\\audio_test\\nIxM8rL5GVE_croped_norm.wav'
rate, audio_sig = wav.read(test_audio_path)
mfcc_feat = mfcc(audio_sig,
                 rate,
                 winlen=winlen,
                 winstep=winstep,
                 numcep=ncep,
                 nfilt=nfilt,
                 lowfreq=lowfreq,
                 highfreq=highfreq,
                 ceplifter=0,
                 appendEnergy=True,
                 winfunc=np.hamming)
delta_feat = delta(mfcc_feat, N=2)
timestamps = []
for i in range(mfcc_feat.shape[0]):
    start_time = i * winstep
    end_time = min(len(audio_sig)*rate, start_time+winlen)
    timestamps.append((start_time+end_time)/2)
columns = []
columns.append("Energy")
for i in range(1, ncep):
    columns.append("Cepstral%d" % i)
columns.append("Energy_delta")
for i in range(1, ncep):
    columns.append("Cepstral%d_delta" % i)
X = pd.DataFrame(data=np.c_[mfcc_feat, delta_feat],
                 index=timestamps,
                 columns=columns)
X.index.name = "Timestamps"
X = X.to_numpy(np.float32)
f = open('C:\\Projects\\Python projects\\DeepFake\\save\\obama_data\\obama_data.cpkl', 'rb')
a = pickle.load(f)
f.close()
means = a['input_mean']
stds = a['input_std']
for i in range(len(X)):
    X[i] = (X[i]-means)/stds
x = np.empty((len(X)//100, 100, 26), dtype=np.float32)
seq_start_ind = 0
for i in range(len(x)):
    x[i] = np.copy(X[seq_start_ind:seq_start_ind + 100, :])
    seq_start_ind += 100
rnn_net = BidirectionalSingleLayerLSTMmodel("SL_BLSTM_90units", rnn_size=90)
y = rnn_net.test(x)
y = np.reshape(y, (len(x)*100, 54))
timestamps = timestamps[0:(len(timestamps)//100)*100]
y = pd.DataFrame(y, index=pd.to_timedelta(timestamps, unit='s'))
y_s = pd.Series(y[0], index=y.index)
y_r = y.resample('33333U').mean()
y_r.drop(y_r.tail(1).index, inplace=True)
y_r.to_csv('D:\\obama_dataset\\mouth_shapes_test\\nIxM8rL5GVE.csv',
           header=False,
           index=False)