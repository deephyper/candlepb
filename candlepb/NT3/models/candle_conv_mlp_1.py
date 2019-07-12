import tensorflow as tf

from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.op1d import (Activation, Conv1D,
                                                      Dense, Dropout, Flatten,
                                                      Identity, MaxPooling1D)
from deephyper.search.nas.model.space.struct import AutoOutputStructure

def add_conv_op_(node):
    node.add_op(Identity())
    node.add_op(Conv1D(filter_size=3, num_filters=8))
    node.add_op(Conv1D(filter_size=4, num_filters=8))
    node.add_op(Conv1D(filter_size=5, num_filters=8))
    node.add_op(Conv1D(filter_size=6, num_filters=8))

def add_activation_op_(node):
    node.add_op(Identity())
    node.add_op(Activation(activation='relu'))
    node.add_op(Activation(activation='tanh'))
    node.add_op(Activation(activation='sigmoid'))

def add_pooling_op_(node):
    node.add_op(Identity())
    node.add_op(MaxPooling1D(pool_size=3, padding='same'))
    node.add_op(MaxPooling1D(pool_size=4, padding='same'))
    node.add_op(MaxPooling1D(pool_size=5, padding='same'))
    node.add_op(MaxPooling1D(pool_size=6, padding='same'))

def add_dense_op_(node):
    node.add_op(Identity())
    node.add_op(Dense(units=10))
    node.add_op(Dense(units=50))
    node.add_op(Dense(units=100))
    node.add_op(Dense(units=200))
    node.add_op(Dense(units=250))
    node.add_op(Dense(units=500))
    node.add_op(Dense(units=750))
    node.add_op(Dense(units=1000))

def add_dropout_op_(node):
    node.add_op(Identity())
    node.add_op(Dropout(rate=0.5))
    node.add_op(Dropout(rate=0.4))
    node.add_op(Dropout(rate=0.3))
    node.add_op(Dropout(rate=0.2))
    node.add_op(Dropout(rate=0.1))
    node.add_op(Dropout(rate=0.05))

def create_structure(input_shape=(2,), output_shape=(1,), *args, **kwargs):
    struct = AutoOutputStructure(input_shape, output_shape, regression=False)

    n1 = VariableNode('N')
    add_conv_op_(n1)
    struct.connect(struct.input_nodes[0], n1)

    n2 = VariableNode('N')
    add_activation_op_(n2)
    struct.connect(n1, n2)

    n3 = VariableNode('N')
    add_pooling_op_(n3)
    struct.connect(n2, n3)

    n4 = VariableNode('N')
    add_conv_op_(n4)
    struct.connect(n3, n4)

    n5 = VariableNode('N')
    add_activation_op_(n5)
    struct.connect(n4, n5)

    n6 = VariableNode('N')
    add_pooling_op_(n6)
    struct.connect(n5, n6)

    n7 = ConstantNode(op=Flatten(), name='N')
    struct.connect(n6, n7)

    n8 = VariableNode('N')
    add_dense_op_(n8)
    struct.connect(n7, n8)

    n9 = VariableNode('N')
    add_activation_op_(n9)
    struct.connect(n8, n9)

    n10 = VariableNode('N')
    add_dropout_op_(n10)
    struct.connect(n9, n10)

    n11 = VariableNode('N')
    add_dense_op_(n11)
    struct.connect(n10, n11)

    n12 = VariableNode('N')
    add_activation_op_(n12)
    struct.connect(n11, n12)

    n13 = VariableNode('N')
    add_dropout_op_(n13)
    struct.connect(n12, n13)

    return struct

def test_create_structure():
    from random import random, seed
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    # seed(10)
    shapes = [(2, )]
    structure = create_structure(shapes, (1,), 5)

    ops = [random() for i in range(structure.num_nodes)]
    print('num ops: ', len(ops))
    structure.set_ops(ops)
    structure.draw_graphviz('nt3_model.dot')

    model = structure.create_model()
    print('depth: ', structure.depth)

    model = structure.create_model()
    plot_model(model, to_file='nt3_model.png', show_shapes=True)

    model.summary()

if __name__ == '__main__':
    test_create_structure()
