import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

np.random.seed(2018)

def load_data():
    filename = 'training_6.dat'
    if os.path.exists(filename):
        train_data = pd.read_csv(filename,header=None,delim_whitespace=True)
        train_data = train_data.drop(train_data.columns[[0]], axis=1) 
        print(train_data.head())
    
    filename = 'verify_6.dat'
    if os.path.exists(filename):
        test_data = pd.read_csv(filename,header=None,delim_whitespace=True)
        test_data = test_data.drop(test_data.columns[[0]], axis=1) 
        print(test_data.head())

    train_y = train_data.iloc[:,10:17]
    train_X = train_data.iloc[:,0:10]

    valid_y = test_data.iloc[:,10:17]
    valid_X = test_data.iloc[:,0:10]

    print(f'train_X shape: {np.shape(train_X)}')
    print(f'train_y shape: {np.shape(train_y)}')
    print(f'valid_X shape: {np.shape(valid_X)}')
    print(f'valid_y shape: {np.shape(valid_y)}')
    return (train_X, train_y), (valid_X, valid_y)

if __name__ == '__main__':
    load_data()
