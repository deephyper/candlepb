#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import collections
import logging
import os
import random
import threading

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import plot_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from scipy.stats.stats import pearsonr

# For non-interactive plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import candlepb.Uno.uno as benchmark
import candlepb.common.candle_keras as candle

import candlepb.Uno.uno_data as uno_data
from candlepb.Uno.uno_data import CombinedDataLoader, CombinedDataGenerator

from deephyper.contrib.callbacks import StopIfUnfeasible


logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

session_conf = tf.ConfigProto(intra_op_parallelism_threads=62)
sess = tf.Session(config=session_conf)
K.set_session(sess)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    random.seed(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        tf.set_random_seed(seed)
        # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)

        # Uncommit when running on an optimized tensorflow where NUM_INTER_THREADS and
        # NUM_INTRA_THREADS env vars are set.
        # session_conf = tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
        #	intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)


def verify_path(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def set_up_logger(logfile, verbose):
    verify_path(logfile)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    for log in [logger, uno_data.logger]:
        log.setLevel(logging.DEBUG)
        log.addHandler(fh)
        log.addHandler(sh)


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.E={}'.format(args.epochs)
    ext += '.O={}'.format(args.optimizer)
    # ext += '.LEN={}'.format(args.maxlen)
    ext += '.LR={}'.format(args.learning_rate)
    ext += '.CF={}'.format(''.join([x[0] for x in sorted(args.cell_features)]))
    ext += '.DF={}'.format(''.join([x[0] for x in sorted(args.drug_features)]))
    if args.feature_subsample > 0:
        ext += '.FS={}'.format(args.feature_subsample)
    if args.drop > 0:
        ext += '.DR={}'.format(args.drop)
    if args.warmup_lr:
        ext += '.wu_lr'
    if args.reduce_lr:
        ext += '.re_lr'
    if args.residual:
        ext += '.res'
    if args.use_landmark_genes:
        ext += '.L1000'
    if args.no_gen:
        ext += '.ng'
    for i, n in enumerate(args.dense):
        if n > 0:
            ext += '.D{}={}'.format(i+1, n)
    if args.dense_feature_layers != args.dense:
        for i, n in enumerate(args.dense):
            if n > 0:
                ext += '.FD{}={}'.format(i+1, n)

    return ext


def discretize(y, bins=5):
    percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
    thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    return classes


def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def mae(y_true, y_pred):
    return keras.metrics.mean_absolute_error(y_true, y_pred)


def evaluate_prediction(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'corr': corr}


def log_evaluation(metric_outputs, description='Comparing y_true and y_pred:'):
    logger.info(description)
    for metric, value in metric_outputs.items():
        logger.info('  {}: {:.4f}'.format(metric, value))


def plot_history(out, history, metric='loss', title=None):
    title = title or 'model {}'.format(metric)
    val_metric = 'val_{}'.format(metric)
    plt.figure(figsize=(8, 6))
    plt.plot(history.history[metric], marker='o')
    plt.plot(history.history[val_metric], marker='d')
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train_{}'.format(metric), 'val_{}'.format(metric)], loc='upper center')
    png = '{}.plot.{}.png'.format(out, metric)
    plt.savefig(png, bbox_inches='tight')


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())))
        self.print_fcn(msg)


class PermanentDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x


class ModelRecorder(Callback):
    def __init__(self, save_all_models=False):
        Callback.__init__(self)
        self.save_all_models = save_all_models
        get_custom_objects()['PermanentDropout'] = PermanentDropout

    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.best_val_loss = np.Inf
        self.best_model = None

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_model = keras.models.clone_model(self.model)
            self.best_val_loss = val_loss


def build_feature_model(input_shape, name='', dense_layers=[1000, 1000],
                        activation='relu', residual=False,
                        dropout_rate=0, permanent_dropout=True):
    x_input = Input(shape=input_shape)
    h = x_input
    for i, layer in enumerate(dense_layers):
        x = h
        h = Dense(layer, activation=activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    model = Model(x_input, h, name=name)
    return model


def build_model(loader, args, permanent_dropout=True, silent=False):
    input_models = {}
    dropout_rate = args.drop
    for fea_type, shape in loader.feature_shapes.items():
        base_type = fea_type.split('.')[0]
        if base_type in ['cell', 'drug']:
            box = build_feature_model(input_shape=shape, name=fea_type,
                                      dense_layers=args.dense_feature_layers,
                                      dropout_rate=dropout_rate, permanent_dropout=permanent_dropout)
            if not silent:
                logger.debug('Feature encoding submodel for %s:', fea_type)
                box.summary(print_fn=logger.debug)
            input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name='input.'+fea_name)
        inputs.append(fea_input)
        if fea_type in input_models:
            input_model = input_models[fea_type]
            encoded = input_model(fea_input)
        else:
            encoded = fea_input
        encoded_inputs.append(encoded)

    merged = keras.layers.concatenate(encoded_inputs)

    h = merged
    for i, layer in enumerate(args.dense):
        x = h
        h = Dense(layer, activation=args.activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if args.residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(1)(h)

    return Model(inputs, output)


def initialize_parameters():

    # Build benchmark object
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, 'uno_default_model.txt', 'keras',
    prog='uno_baseline', desc='Build neural network based models to predict tumor response to single and paired drugs.')

    # Initialize parameters
    gParameters = candle.initialize_parameters(unoBmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

from deephyper.benchmark.util import numpy_dict_cache

def load_data1(batch_size):

    params = initialize_parameters()
    args = Struct(**params)
    set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    verify_path(args.save)
    prefix = args.save + ext
    logfile = args.logfile if args.logfile else prefix+'.log'
    set_up_logger(logfile, args.verbose)
    logger.info('Params: {}'.format(params))

    loader = CombinedDataLoader(seed=args.rng_seed)
    loader.load(cache=args.cache,
                ncols=args.feature_subsample,
                cell_features=args.cell_features,
                drug_features=args.drug_features,
                drug_median_response_min=args.drug_median_response_min,
                drug_median_response_max=args.drug_median_response_max,
                use_landmark_genes=args.use_landmark_genes,
                use_filtered_genes=args.use_filtered_genes,
                preprocess_rnaseq=args.preprocess_rnaseq,
                single=args.single,
                train_sources=args.train_sources,
                test_sources=args.test_sources,
                embed_feature_source=not args.no_feature_source,
                encode_response_source=not args.no_response_source,
                )

    val_split = args.validation_split
    train_split = 1 - val_split

    loader.partition_data(cv_folds=args.cv, train_split=train_split, val_split=val_split, cell_types=args.cell_types, by_cell=args.by_cell, by_drug=args.by_drug)
    print('-- partition data ok')
    train_gen = CombinedDataGenerator(loader, batch_size=batch_size, shuffle=args.shuffle)
    val_gen = CombinedDataGenerator(loader, partition='val', batch_size=batch_size, shuffle=args.shuffle)

    return train_gen, val_gen


from deephyper.search import util

def run_model(config):

    num_epochs = config['hyperparameters']['num_epochs']
    batch_size = config['hyperparameters']['batch_size']

    config['create_structure']['func'] = util.load_attr_from(
         config['create_structure']['func'])

    train_gen, val_gen = load_data1(batch_size=batch_size)
    x_train_list, y_train = train_gen.flow(single=False).__next__()

    input_shape = [np.shape(a)[1:] for a in x_train_list]
    print('input_shape: ', input_shape)
    output_shape = (1, )
    print('output_shape: ', output_shape)

    cs_kwargs = config['create_structure'].get('kwargs')
    if cs_kwargs is None:
        structure = config['create_structure']['func'](input_shape, output_shape)
    else:
        structure = config['create_structure']['func'](input_shape, output_shape, **cs_kwargs)

    arch_seq = config['arch_seq']

    print(f'actions list: {arch_seq}')

    structure.set_ops(arch_seq)
    structure.draw_graphviz('model_global_uno.dot')

    model = structure.create_model()

    from keras.utils import plot_model
    plot_model(model, 'model_global_combo.png', show_shapes=True)

    model.summary()


    optimizer = optimizers.deserialize({'class_name': 'adam', 'config': {}})

    model.compile(loss='mse', optimizer=optimizer, metrics=[mae, r2])

    history = model.fit_generator(train_gen.flow(single=False),
                                train_gen.steps,
                                epochs=num_epochs,
                                validation_data=val_gen.flow(single=False),
                                validation_steps=val_gen.steps)

    print(history.history)

    try:
        return history.history['val_r2'][0]
    except:
        return -1.0

if __name__ == '__main__':
    from candlepb.Uno.problems.problem_exp2 import Problem
    config = Problem.space
    config['hyperparameters']['num_epochs'] = 10
    config['hyperparameters']['batch_size'] = 256
    config['arch_seq'] = [
            0.2222222222222222,
            0.2222222222222222,
            0.05555555555555555,
            0.8333333333333334,
            0.1111111111111111,
            0.6111111111111112,
            0.2222222222222222,
            0.0,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.0,
            0.2222222222222222,
            0.2222222222222222,
            0.16666666666666666,
            0.2222222222222222,
            0.2222222222222222,
            0.0,
            0.16666666666666666,
            0.5,
            0.5555555555555556,
            0.0,
            0.1111111111111111,
            0.0,
            0.2222222222222222,
            0.1111111111111111,
            0.2222222222222222,
            0.5,
            0.2222222222222222,
            0.6111111111111112,
            0.16666666666666666,
            0.6111111111111112,
            0.2222222222222222,
            0.6666666666666666,
            0.16666666666666666,
            0.3333333333333333,
            0.6111111111111112,
            0.0,
            0.1111111111111111,
            0.6111111111111112,
            0.0,
            0.16666666666666666,
            0.0,
            0.0
        ]
    run_model(config)