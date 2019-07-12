import tensorflow as tf

from deephyper.search.nas.model.space.node import ConstantNode
from deephyper.search.nas.model.space.op.op1d import (Activation, Conv1D,
                                                      Dense, Dropout, Flatten,
                                                      Identity, MaxPooling1D)
from deephyper.search.nas.model.space.struct import AutoOutputStructure


def create_structure(input_shape=(2,), output_shape=(1,), *args, **kwargs):
    struct = AutoOutputStructure(input_shape, output_shape, regression=False)

    n1 = ConstantNode(op=Conv1D(filter_size=20, num_filters=128), name='N')
    struct.connect(struct.input_nodes[0], n1)

    n2 = ConstantNode(op=Activation(activation='relu'), name='N')
    struct.connect(n1, n2)

    n3 = ConstantNode(op=MaxPooling1D(pool_size=1, padding='same'), name='N')
    struct.connect(n2, n3)

    n4 = ConstantNode(op=Conv1D(filter_size=10, num_filters=128),name='N')
    struct.connect(n3, n4)

    n5 = ConstantNode(op=Activation(activation='relu'), name='N')
    struct.connect(n4, n5)

    n6 = ConstantNode(op=MaxPooling1D(pool_size=10, padding='same'), name='N')
    struct.connect(n5, n6)

    n7 = ConstantNode(op=Flatten(), name='N')
    struct.connect(n6, n7)

    n8 = ConstantNode(op=Dense(units=200), name='N')
    struct.connect(n7, n8)

    n9 = ConstantNode(op=Activation(activation='relu'), name='N')
    struct.connect(n8, n9)

    n10 = ConstantNode(op=Dropout(rate=0.1), name='N')
    struct.connect(n9, n10)

    n11 = ConstantNode(op=Dense(units=20), name='N')
    struct.connect(n10, n11)

    n12 = ConstantNode(op=Activation(activation='relu'), name='N')
    struct.connect(n11, n12)

    n13 = ConstantNode(op=Dropout(rate=0.1), name='N')
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
