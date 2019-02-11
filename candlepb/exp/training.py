import numpy as np
import traceback
from scipy import stats
from inspect import signature

#### COMBO ####
# from candlepb.Combo.problem_exp1 import Problem
# from candlepb.Combo.problem_exp2 import Problem
from candlepb.Combo.problem_exp3 import Problem

#### NT3 ####
# from candlepb.NT3.problem_exp1 import Problem
# from candlepb.NT3.problem_exp2 import Problem

# from deephyper.benchmark.nas.mnist1D.problem import Problem

from tensorflow.keras.utils import plot_model

from deephyper.core.model_utils import number_parameters
from deephyper.search import util
from deephyper.search.nas.model.trainer.regressor_train_valid import \
    TrainerRegressorTrainValid
from deephyper.search.nas.model.trainer.classifier_train_valid import \
    TrainerClassifierTrainValid

PROP = 0.1
NUM_EPOCHS = 0
ARCH_SEQ = [
            0.8,
            0.1,
            0.2,
            0.4,
            0.5,
            0.0,
            0.2,
            0.1,
            0.0,
            0.1,
            0.4,
            0.0,
            0.0,
            0.8,
            0.0,
            0.8,
            0.9,
            0.2,
            0.2,
            0.2,
            0.3,
            0.6,
            0.4,
            0.7,
            0.2,
            0.6,
            0.6,
            0.4,
            0.1,
            0.4,
            0.4,
            0.5,
            0.9,
            0.6,
            0.1,
            0.8,
            0.9,
            0.4,
            0.3,
            0.0,
            0.6,
            0.4,
            0.0,
            0.7,
            0.4
        ]

def main(config):

    num_epochs = NUM_EPOCHS

    load_data = config['load_data']['func']

    print('[PARAM] Loading data')
    # Loading data
    kwargs = config['load_data'].get('kwargs')
    sig_load_data = signature(load_data)
    if len(sig_load_data.parameters) == 0:
        data = load_data()
    else:
        if 'prop' in sig_load_data.parameters:
            if kwargs is None:
                data = load_data(prop=PROP)
            else:
                kwargs['prop'] = PROP
                data = load_data(**kwargs)
        else:
            if kwargs is None:
                data = load_data()
            else:
                data = load_data(**kwargs)
    print('[PARAM] Data loaded')

    # Set data shape
    if type(data) is tuple:
        if len(data) != 2:
            raise RuntimeError(f'Loaded data are tuple, should ((training_input, training_output), (validation_input, validation_output)) but length=={len(data)}')
        (t_X, t_y), (v_X, v_y) = data
        if type(t_X) is np.ndarray and  type(t_y) is np.ndarray and \
            type(v_X) is np.ndarray and type(v_y) is np.ndarray:
            input_shape = np.shape(t_X)[1:]
        elif type(t_X) is list and type(t_y) is np.ndarray and \
            type(v_X) is list and type(v_y) is np.ndarray:
            input_shape = [np.shape(itX)[1:] for itX in t_X] # interested in shape of data not in length
        else:
            raise RuntimeError(f'Data returned by load_data function are of a wrong type: type(t_X)=={type(t_X)},  type(t_y)=={type(t_y)}, type(v_X)=={type(v_X)}, type(v_y)=={type(v_y)}')
        output_shape = np.shape(t_y)[1:]
        config['data'] = {
            'train_X': t_X,
            'train_Y': t_y,
            'valid_X': v_X,
            'valid_Y': v_y
        }
    elif type(data) is dict:
        config['data'] = data
        input_shape = [data['shapes'][0][f'input_{i}'] for i in range(len(data['shapes'][0]))]
        output_shape = data['shapes'][1]
    else:
        raise RuntimeError(f'Data returned by load_data function are of an unsupported type: {type(data)}')

    structure = config['create_structure']['func'](input_shape, output_shape, **config['create_structure']['kwargs'])
    arch_seq = ARCH_SEQ
    structure.set_ops(arch_seq)
    try:
        structure.draw_graphviz('graph_full.dot')
    except:
        pass

    print('Model operations set.')

    if config.get('preprocessing') is not None:
        preprocessing = util.load_attr_from(config['preprocessing']['func'])
        config['preprocessing']['func'] = preprocessing
        print(f"Preprocessing set with: {config['preprocessing']}")
    else:
        print('No preprocessing...')
        config['preprocessing'] = None

    model_created = False
    if config['regression']:
        try:
            model = structure.create_model()
            model_created = True
        except:
            model_created = False
            print('Error: Model creation failed...')
            print(traceback.format_exc())
        if model_created:
            try:
                plot_model(model, to_file='model.png', show_shapes=True)
            except:
                print('can\t create model.png file...')
            try:
                model.load_weights("model_weights.h5")
                print('model weights loaded!')
            except:
                print('failed to load model weights...')
            trainer = TrainerRegressorTrainValid(config=config, model=model)
    else:
        try:
            model = structure.create_model(activation='softmax')
            model_created = True
        except Exception as err:
            model_created = False
            print('Error: Model creation failed...')
            print(traceback.format_exc())
        if model_created:
            try:
                plot_model(model, to_file='model.png', show_shapes=True)
            except:
                print('can\t create model.png file...')
            try:
                model.load_weights("model_weights.h5")
                print('model weights loaded!')
            except:
                print('failed to load model weights...')
            trainer = TrainerClassifierTrainValid(config=config, model=model)

    print('Trainer is ready.')
    print(f'Start training... num_epochs={num_epochs}')

    nparams = number_parameters()
    print('model number of parameters: ', nparams)
    trainer.train(num_epochs=num_epochs)

    # serialize weights to HDF5
    model.save_weights("model_weights.h5")
    print("Saved model weight to disk: model_weights.h5")

    if config['regression']:
        y_orig, y_pred = trainer.predict('valid')
        r_list = list()
        for dim in range(np.shape(y_orig)[1]):
            r, _ = stats.pearsonr(y_orig[:, dim], y_pred[:, dim])
            r_list.append(r)
        print('r_list: ', r_list)



if __name__ == '__main__':
    config = Problem.space
    main(config)
