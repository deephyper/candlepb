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


def load_data1(gParameters):

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
    return (X_train, Y_train), (X_test, Y_test)

HERE = os.path.dirname(os.path.abspath(__file__))

def load_data_deephyper(prop=0.1):
    fnames = [f'x_train-{prop}', f'y_train-{prop}', f'x_valid-{prop}', f'y_valid-{prop}']
    dir_path = "{}/DATA".format(HERE)
    format_path = dir_path + "/data_cached_{}.npy"

    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except:
            pass

    if not os.path.exists(format_path.format(fnames[1])):
        print('-- IF --')
        gParameters = initialize_parameters()
        (x_train, y_train), (x_valid, y_valid) = load_data1(gParameters)

        cursor_train = int(len(y_train) * prop)
        cursor_valid = int(len(y_valid) * prop)

        #!! remove single dimensions for concatenation along axis 1 in preprocessing
        x_train = np.squeeze(x_train[:cursor_train])
        y_train = y_train[:cursor_train]

        x_valid = np.squeeze(x_valid[:cursor_valid])
        y_valid = y_valid[:cursor_valid]

        fdata = [x_train, y_train, x_valid, y_valid]

        for i in range(len(fnames)):
            fname = fnames[i]
            with open(format_path.format(fname), "wb") as f:
                np.save(f, fdata[i])
        # df: dataframe, pandas

    print('-- reading .npy files')
    fls = os.listdir(dir_path)
    fls.sort()
    fdata = []
    x_train = None
    x_valid = None
    y_train = None
    y_valid = None
    for i in range(len(fnames)):
        with open(format_path.format(fnames[i]), "rb") as f:
            if "val" in fnames[i]:
                if "x" in fnames[i]:
                    x_valid = np.load(f)
                else:
                    y_valid = np.load(f)
            else:
                if "x" in fnames[i]:
                    x_train = np.load(f)
                else:
                    y_train = np.load(f)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)

    print('x_valid shapes:', x_valid.shape)
    print('y_valid shape:', y_valid.shape)

    return (x_train, y_train), (x_valid, y_valid)

if __name__ == '__main__':
    load_data_deephyper()
