from __future__ import print_function

import pandas as pd
import numpy as np
import os
import sys
import gzip

from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

#TIMEOUT=3600 # in sec; set this to -1 for no timeout

import candlepb.NT3.nt3 as bmk
import candlepb.common.candle_keras as candle

from deephyper.benchmark.util import numpy_dict_cache

def initialize_parameters():

    # Build benchmark object
    nt3Bmk = bmk.BenchmarkNT3(bmk.file_path, 'nt3_default_model.txt', 'keras',
    prog='nt3_baseline', desc='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')

    # Initialize parameters
    gParameters = candle.initialize_parameters(nt3Bmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters

def load_data2(train_path, test_path, gParameters):

    print('Loading data...')
    df_train = (pd.read_csv(train_path,header=None).values).astype('float32')
    df_test = (pd.read_csv(test_path,header=None).values).astype('float32')
    print('done')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    seqlen = df_train.shape[1]

    df_y_train = df_train[:,0].astype('int')
    df_y_test = df_test[:,0].astype('int')

    Y_train = np_utils.to_categorical(df_y_train,gParameters['classes'])
    Y_test = np_utils.to_categorical(df_y_test,gParameters['classes'])

    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    return X_train, Y_train, X_test, Y_test

@numpy_dict_cache('/projects/datascience/regele/data-tmp/nt3_data.npz')
def load_data1():
    gParameters = initialize_parameters()
    print ('Params:', gParameters)

    file_train = gParameters['train_data']
    file_test = gParameters['test_data']
    url = gParameters['data_url']

    train_file = candle.get_file(file_train, url+file_train, cache_subdir='Pilot1')
    test_file = candle.get_file(file_test, url+file_test, cache_subdir='Pilot1')

    X_train, Y_train, X_test, Y_test = load_data2(train_file, test_file, gParameters)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)
    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test
    }
    return data

@numpy_dict_cache('/dev/shm/nt3_data.py')
def load_data_proxy():
    return load_data1()

def load_data():
    data = load_data_proxy()
    return (data['X_train'], data['Y_train']), (data['X_test'], data['Y_test'])

if __name__ == '__main__':
    load_data1()