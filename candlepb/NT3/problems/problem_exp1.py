from candlepb.NT3.models.candle_conv_mlp_1 import create_structure
from candlepb.NT3.nt3_baseline_keras2 import load_data
from deephyper.benchmark import NaProblem

Problem = NaProblem()

Problem.load_data(load_data)

# Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_structure)

Problem.hyperparameters(
    batch_size=20,
    learning_rate=0.01,
    optimizer='adam',
    num_epochs=1,
)

Problem.loss('categorical_crossentropy')

Problem.metrics(['acc'])

Problem.objective('val_acc__last')

Problem.post_training(
    num_epochs=1000,
    metrics=['acc'],
    model_checkpoint={
        'monitor': 'val_acc',
        'mode': 'max',
        'save_best_only': True,
        'verbose': 1
    },
    early_stopping={
        'monitor': 'val_acc',
        'mode': 'max',
        'verbose': 1,
        'patience': 20
    }
)

if __name__ == '__main__':
    print(Problem)
