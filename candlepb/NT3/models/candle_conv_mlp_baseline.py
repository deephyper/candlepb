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

    n1 = ConstantNode(op=Conv1D(filter_size=20, num_filters=128), name='N1')
    cell.graph.add_edge(input_nodes[0], n1) # fixed input connection

    n2 = ConstantNode(op=Activation(activation='relu'), name='N2')

    n3 = ConstantNode(op=MaxPooling1D(pool_size=1, padding='same'), name='N3')

    n4 = ConstantNode(op=Conv1D(filter_size=10, num_filters=128),name='N4')

    n5 = ConstantNode(op=Activation(activation='relu'), name='N5')

    n6 = ConstantNode(op=MaxPooling1D(pool_size=10, padding='same'), name='N6')

    n7 = ConstantNode(op=Flatten(), name='N7')

    n8 = ConstantNode(op=Dense(units=200), name='N8')

    n9 = ConstantNode(op=Activation(activation='relu'), name='N9')

    n10 = ConstantNode(op=Dropout(rate=0.1), name='N10')

    n11 = ConstantNode(op=Dense(units=20), name='N11')

    n12 = ConstantNode(op=Activation(activation='relu'), name='N12')

    n13 = ConstantNode(op=Dropout(rate=0.1), name='N13')

    block = Block()
    block.add_node(n1)
    block.add_node(n2)
    block.add_node(n3)
    block.add_node(n4)
    block.add_node(n5)
    block.add_node(n6)
    block.add_node(n7)
    block.add_node(n8)
    block.add_node(n9)
    block.add_node(n10)
    block.add_node(n11)
    block.add_node(n12)
    block.add_node(n13)
    block.add_edge(n1, n2)
    block.add_edge(n2, n3)
    block.add_edge(n3, n4)
    block.add_edge(n4, n5)
    block.add_edge(n5, n6)
    block.add_edge(n6, n7)
    block.add_edge(n7, n8)
    block.add_edge(n8, n9)
    block.add_edge(n9, n10)
    block.add_edge(n10, n11)
    block.add_edge(n11, n12)
    block.add_edge(n12, n13)

    cell.add_block(block)

    cell.set_outputs()
    return cell

def create_structure(input_shape=(2,), output_shape=(1,), *args, **kwargs):
    network = KerasStructure(input_shape, output_shape) #, output_op=AddByPadding)
    input_nodes = network.input_nodes

    # CELL 1
    cell1 = create_cell_conv(input_nodes)
    network.add_cell(cell1)

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
    print('num ops: ', len(ops))
    structure.set_ops(ops)
    structure.draw_graphviz('nt3_model.dot')

    model = structure.create_model()
    print('depth: ', structure.depth)

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
