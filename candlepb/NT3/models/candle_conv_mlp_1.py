import tensorflow as tf

from deephyper.search.nas.model.baseline.util.struct import (create_seq_struct,
                                                             create_struct_full_skipco)
from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.structure import KerasStructure
from deephyper.search.nas.model.space.node import VariableNode, ConstantNode
from deephyper.search.nas.model.space.op.basic import Connect
from deephyper.search.nas.model.space.op.op1d import (Conv1D, Dense, Identity, Activation,
                                                      MaxPooling1D, Dropout, Flatten)



def create_cell_conv(input_nodes):
    """Create a cell with convolution.

    Args:
        input_nodes (list(Node)): a list of input_nodes for this cell.

    Returns:
        Cell: the corresponding cell.
    """
    cell = Cell(input_nodes)

    n1 = VariableNode('N1')
    cell.graph.add_edge(input_nodes[0], n1) # fixed input connection
    n1.add_op(Identity())
    n1.add_op(Conv1D(filter_size=2, num_filters=8))
    n1.add_op(Conv1D(filter_size=3, num_filters=8))
    n1.add_op(Conv1D(filter_size=4, num_filters=8))
    n1.add_op(Conv1D(filter_size=5, num_filters=8))
    n1.add_op(Conv1D(filter_size=6, num_filters=8))
    n1.add_op(Conv1D(filter_size=7, num_filters=8))
    n1.add_op(Conv1D(filter_size=8, num_filters=8))
    n1.add_op(Conv1D(filter_size=9, num_filters=8))
    n1.add_op(Conv1D(filter_size=10, num_filters=8))

    n2 = VariableNode('N2')
    n2.add_op(Identity())
    n2.add_op(Activation(activation='relu'))
    n2.add_op(Activation(activation='tanh'))
    n2.add_op(Activation(activation='sigmoid'))

    n3 = VariableNode('N3')
    n3.add_op(Identity())
    n3.add_op(MaxPooling1D(pool_size=2, padding='same'))
    n3.add_op(MaxPooling1D(pool_size=3, padding='same'))
    n3.add_op(MaxPooling1D(pool_size=4, padding='same'))
    n3.add_op(MaxPooling1D(pool_size=5, padding='same'))
    n3.add_op(MaxPooling1D(pool_size=6, padding='same'))
    n3.add_op(MaxPooling1D(pool_size=7, padding='same'))
    n3.add_op(MaxPooling1D(pool_size=8, padding='same'))
    n3.add_op(MaxPooling1D(pool_size=9, padding='same'))
    n3.add_op(MaxPooling1D(pool_size=10, padding='same'))

    block = Block()
    block.add_node(n1)
    block.add_node(n2)
    block.add_node(n3)
    block.add_edge(n1, n2)
    block.add_edge(n2, n3)

    cell.add_block(block)

    cell.set_outputs()
    return cell

def create_cell_mlp(input_nodes):
    """Create a cell with convolution.

    Args:
        input_nodes (list(Node)): a list of input_nodes for this cell.

    Returns:
        Cell: the corresponding cell.
    """
    cell = Cell(input_nodes)

    n1 = ConstantNode(name='N1')
    cell.graph.add_edge(input_nodes[0], n1) # fixed input connection
    n1.set_op(op=Flatten())

    n2 = VariableNode('N2')
    n2.add_op(Identity())
    n2.add_op(Dense(units=10))
    n2.add_op(Dense(units=50))
    n2.add_op(Dense(units=100))
    n2.add_op(Dense(units=200))
    n2.add_op(Dense(units=250))
    n2.add_op(Dense(units=500))
    n2.add_op(Dense(units=750))
    n2.add_op(Dense(units=1000))

    n3 = VariableNode('N3')
    n3.add_op(Identity())
    n3.add_op(Activation(activation='relu'))
    n3.add_op(Activation(activation='tanh'))
    n3.add_op(Activation(activation='sigmoid'))

    n4 = VariableNode('N4')
    n4.add_op(Identity())
    n4.add_op(Dropout(rate=0.5))
    n4.add_op(Dropout(rate=0.4))
    n4.add_op(Dropout(rate=0.3))
    n4.add_op(Dropout(rate=0.2))
    n4.add_op(Dropout(rate=0.1))
    n4.add_op(Dropout(rate=0.05))

    block = Block()
    block.add_node(n1)
    block.add_node(n2)
    block.add_node(n3)
    block.add_node(n4)
    block.add_edge(n1, n2)
    block.add_edge(n2, n3)
    block.add_edge(n3, n4)

    cell.add_block(block)

    cell.set_outputs()
    return cell

def create_structure(input_shape=(2,), output_shape=(1,), *args, **kwargs):
    network = KerasStructure(input_shape, output_shape) #, output_op=AddByPadding)
    input_nodes = network.input_nodes

    # CELL 1
    cell1 = create_cell_conv(input_nodes)
    network.add_cell(cell1)

    # CELL 2
    cell2 = create_cell_conv([cell1.output])
    network.add_cell(cell2)

    # CELL 3
    cell3 = create_cell_mlp([cell2.output])
    network.add_cell(cell3)

    # CELL 4
    cell4 = create_cell_mlp([cell3.output])
    network.add_cell(cell4)

    return network

def test_create_structure():
    from random import random, seed
    from deephyper.search.nas.model.space.structure import KerasStructure
    from deephyper.core.model_utils import number_parameters
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    # seed(10)
    shapes = [(2, )]
    structure = create_structure(shapes, (1,), 5)
    assert type(structure) is KerasStructure

    ops = [random() for i in range(structure.num_nodes)]
    ops = [
            0.16666666666666666,
            0.5,
            0.3333333333333333,
            0.16666666666666666,
            0.0,
            0.5,
            0.0,
            0.16666666666666666,
            0.16666666666666666,
            0.0,
            0.0,
            0.6666666666666666
        ]
    print('num ops: ', len(ops))
    structure.set_ops(ops)
    structure.draw_graphviz('nt3_model.dot')

    model = structure.create_model()

    model = structure.create_model()
    plot_model(model, to_file='nt3_model.png', show_shapes=True)

    model.summary()

    # import numpy as np
    # x0 = np.zeros((1, *shapes[0]))
    # x1 = np.zeros((1, *shapes[1]))
    # x2 = np.zeros((1, *shapes[2]))
    # inpts = [x0, x1, x2]
    # y = model.predict(inpts)

    # for x in inpts:
    #     print(f'shape(x): {np.shape(x)}')
    # print(f'shape(y): {np.shape(y)}')

    # total_parameters = number_parameters()
    # print('total_parameters: ', total_parameters)

    # assert np.shape(y) == (1, 1), f'Wrong output shape {np.shape(y)} should be {(1, 1)}'

if __name__ == '__main__':
    test_create_structure()
