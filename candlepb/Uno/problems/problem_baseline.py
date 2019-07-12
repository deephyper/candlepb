
from deephyper.benchmark import NaProblem
from candlepb.Uno.structs.uno_mlp_baseline import create_structure
from candlepb.Uno.uno_baseline_keras2 import load_data_multi_array

Problem = NaProblem()

Problem.load_data(load_data_multi_array)

# Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_structure)

Problem.hyperparameters(
    batch_size=64,
    learning_rate=0.001,
    optimizer='adam',
    num_epochs=1,
)

Problem.loss('mse')

Problem.metrics(['r2'])

Problem.objective('val_r2__last')

Problem.post_training(
    num_epochs=1000,
    metrics=['r2'],
    model_checkpoint={
        'monitor': 'val_r2',
        'mode': 'max',
        'save_best_only': True,
        'verbose': 1
    },
    early_stopping={
        'monitor': 'val_r2',
        'mode': 'max',
        'verbose': 1,
        'patience': 20
    }
)

if __name__ == '__main__':
    print(Problem)
