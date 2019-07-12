import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from candlepb.NT3.nt3_baseline_keras2 import load_data1
from candlepb.NT3.problems.problem_baseline import Problem

# parser = argparse.ArgumentParser()
# parser.add_argument("-id", "--id", dest="id", help="which arch to use")
# parser.add_argument("-f", "--fname", dest="fname", help="which file to use")
# cmd_args = vars(parser.parse_args())

# archid = int(cmd_args["id"])
# file_name = cmd_args["fname"]
# with open(file_name) as json_file:
#     data = json.load(json_file)

# arch_seq = data['arch_seq'][archid]
arch_seq = []
print(arch_seq)
config = Problem.space
config['arch_seq'] = arch_seq
config['hyperparameters']['num_epochs'] = 20

data = load_data1()
x_train, y_train, x_val, y_val = data['X_train'], data['Y_train'], data['X_test'], data['Y_test']

input_shape = [np.shape(x_train)[1:]]
output_shape = np.shape(y_train)[1:]

structure = config['create_structure']['func'](input_shape, output_shape)
structure.set_ops(config['arch_seq'])
model = structure.create_model(activation='softmax')

n_params = model.count_params()


optimizer = keras.optimizers.deserialize({'class_name': 'adam', 'config': {}})

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

t1 = time.time()
history = model.fit(x_train, y_train,
                            batch_size=config['hyperparameters']['batch_size'],
                            epochs=config['hyperparameters']['num_epochs'],
                            validation_data=(x_val, y_val))
t2 = time.time()

data = history.history
data['n_parameters'] = n_params
data['training_time'] = t2 - t1

print(data)
