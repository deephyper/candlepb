import tensorflow as tf

from deephyper.search.nas.model.baseline.util.struct import (create_seq_struct,
                                                             create_struct_full_skipco)
from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.node import Node
from deephyper.search.nas.model.space.op.basic import Connect
from deephyper.search.nas.model.space.op.op1d import (Conv1D, Dense, Identity,
                                                      MaxPooling1D,
                                                      dropout_ops)


def create_cell_1(input_nodes):
    """Create a cell with convolution.

    Args:
        input_nodes (list(Node)): a list of input_nodes for this cell.

    Returns:
        Cell: the corresponding cell.
    """
    cell = Cell(input_nodes)

    def create_conv_block(input_nodes):
        # first node of block
        n1 = Node('N1')
        for inpt in input_nodes:
            n1.add_op(Connect(cell.graph, inpt, n1))

        def create_conv_node(name):
            n = Node(name)
            n.add_op(Identity())
            n.add_op(Conv1D(filter_size=3, num_filters=16))
            n.add_op(MaxPooling1D(pool_size=3, padding='same'))
            n.add_op(Dense(10, tf.nn.relu))
            n.add_op(Conv1D(filter_size=5, num_filters=16))
            n.add_op(MaxPooling1D(pool_size=5, padding='same'))
            n.add_op(Dense(100, tf.nn.relu))
            n.add_op(Conv1D(filter_size=10, num_filters=16))
            n.add_op(MaxPooling1D(pool_size=10, padding='same'))
            n.add_op(Dense(1000, tf.nn.relu))
            return n
        # second node of block
        n2 = create_conv_node('N2')

        n3 = create_conv_node('N3')

        block = Block()
        block.add_node(n1)
        block.add_node(n2)
        block.add_node(n3)

        block.add_edge(n1, n2)
        block.add_edge(n2, n3)
        return block

    block1 = create_conv_block(input_nodes)
    block2 = create_conv_block(input_nodes)
    block3 = create_conv_block(input_nodes)

    cell.add_block(block1)
    cell.add_block(block2)
    cell.add_block(block3)

    cell.set_outputs()
    return cell

def create_structure(input_shape=(2,), output_shape=(1,), num_cells=2):
    return create_struct_full_skipco(
        input_shape,
        output_shape,
        create_cell_1,
        num_cells)

def test_create_structure():
    from random import random, seed
    from deephyper.search.nas.model.space.structure import KerasStructure
    from deephyper.core.model_utils import number_parameters
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    # seed(10)
    shapes = [(942, ), (3820, ), (3820, )]
    structure = create_structure(shapes, (1,), 5)
    assert type(structure) is KerasStructure

    ops = [random() for i in range(structure.num_nodes)]
    # ops = [0 for i in range(structure.num_nodes)]
    print('num ops: ', len(ops))
    structure.set_ops(ops)
    structure.draw_graphviz('graph_anl_conv_mlp_2_test.dot')

    model = structure.create_model()

    model = structure.create_model()
    plot_model(model, to_file='graph_anl_conv_mlp_2_test.png', show_shapes=True)

    import numpy as np
    x0 = np.zeros((1, *shapes[0]))
    x1 = np.zeros((1, *shapes[1]))
    x2 = np.zeros((1, *shapes[2]))
    inpts = [x0, x1, x2]
    y = model.predict(inpts)

    for x in inpts:
        print(f'shape(x): {np.shape(x)}')
    print(f'shape(y): {np.shape(y)}')

    total_parameters = number_parameters()
    print('total_parameters: ', total_parameters)

    # assert np.shape(y) == (1, 1), f'Wrong output shape {np.shape(y)} should be {(1, 1)}'

if __name__ == '__main__':
    test_create_structure()
