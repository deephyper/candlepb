from deephyper.benchmark import Problem
from candlepb.Uno.uno_baseline_keras2 import load_data_proxy
from candlepb.Uno.models.uno_mlp_1 import create_structure

# We create our Problem object with the Problem class, you don't have to name your Problem object 'Problem' it can be any name you want. You can also define different problems in the same module.
Problem = Problem()

# You define if your problem is a regression problem (the reward will be minus of the mean squared error) or a classification problem (the reward will be the accuracy of the network on the validation set).
Problem.add_dim('regression', True)

# You define the create structure function. This function will return an object following the Structure interface. You can also have kwargs arguments such as 'num_cells' for this function.
Problem.add_dim('create_structure', {
    'func': create_structure,
})

# You define the hyperparameters used to train your generated models during the search.
Problem.add_dim('hyperparameters', {
    'batch_size': 32,
    'num_epochs': 1,
})

# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == '__main__':
    print(Problem)
