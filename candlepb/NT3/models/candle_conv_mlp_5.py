import tensorflow as tf

from deephyper.search.nas.model.baseline.util.struct import (create_seq_struct,
                                                             create_struct_full_skipco)
from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.basic import (AddByPadding, Connect,
                                                       Tensor)
from deephyper.search.nas.model.space.op.op1d import (Activation, Conv1D,
                                                      Dense, Dropout, Flatten,
                                                      Identity, MaxPooling1D)
from deephyper.search.nas.model.space.structure import KerasStructure


def create_conv_node(name):
    n = VariableNode(name)
    n.add_op(Identity())
    # for i in range(2, 11):
    for i in range(3, 7):
        n.add_op(Conv1D(filter_size=i, num_filters=8))
    return n


def create_pool_node(name):
    n = VariableNode(name)
    n.add_op(Identity())
    # for i in range(2, 11):
    for i in range(3, 7)
        n.add_op(MaxPooling1D(pool_size=i, padding='same'))
    return n

def create_act_node(name):
    n = VariableNode(name)
    n.add_op(Identity())
    n.add_op(Activation(activation='relu'))
    n.add_op(Activation(activation='tanh'))
    n.add_op(Activation(activation='sigmoid'))
    return n


def create_cell_conv(feed_forward_node, input_nodes):
    """Create a cell with convolution.

    Args:
        input_nodes (list(Node)): a list of input_nodes for this cell.

    Returns:
        Cell: the corresponding cell.
    """
    cell = Cell(input_nodes)

    nullNode = ConstantNode(op=Tensor([]), name='None')
    def get_skipco_node():
        cnode = VariableNode(name='SkipCo')
        cnode.add_op(Connect(cell.graph, nullNode, cnode)) # SAME
        return cnode

    n1 = create_conv_node('N1')
    cell.graph.add_edge(feed_forward_node, n1) # fixed input connection

    n2 = create_act_node('N2')

    n3 = create_pool_node('N3')

    block = Block()
    # block.add_node(cnode)
    block.add_node(n1)
    block.add_node(n2)
    block.add_node(n3)
    # block.add_edge(cnode, n1)
    block.add_edge(n1, n2)
    block.add_edge(n2, n3)
    cell.add_block(block)

    cnode = get_skipco_node()
    for n in input_nodes:
        cnode.add_op(Connect(cell.graph, n, cnode))
    block2 = Block()
    block2.add_node(cnode)
    cell.add_block(block2)

    mergeNode = ConstantNode(name='merge')
    mergeNode.set_op(AddByPadding(cell.graph, mergeNode, cell.get_blocks_output()))
    cell.set_outputs(node=mergeNode)
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
    skipco_input_nodes = input_nodes[:]
    feed_forward_node = input_nodes[0]

    cell_1 = Cell(input_nodes)

    n1 = create_conv_node('N1')
    cell_1.graph.add_edge(feed_forward_node, n1) # fixed input connection

    n2 = create_act_node('N2')

    n3 = create_pool_node('N3')

    block = Block()
    block.add_node(n1)
    block.add_node(n2)
    block.add_node(n3)
    block.add_edge(n1, n2)
    block.add_edge(n2, n3)
    cell_1.add_block(block)

    mergeNode = ConstantNode(name='merge')
    mergeNode.set_op(AddByPadding(cell_1.graph, mergeNode, cell_1.get_blocks_output()))
    cell_1.set_outputs(node=mergeNode)
    network.add_cell(cell_1)
    skipco_input_nodes.append(cell_1.output)
    feed_forward_node = cell_1.output

    # CONV CELLS
    for i in range(3):
        # CONV CELL i
        cell_i = create_cell_conv(feed_forward_node, skipco_input_nodes)
        network.add_cell(cell_i)
        skipco_input_nodes.append(cell_i.output)
        feed_forward_node = cell_i.output

    # CELL 3
    cell3 = create_cell_mlp([feed_forward_node])
    network.add_cell(cell3)
    feed_forward_node = cell3.output

    # CELL 4
    cell4 = create_cell_mlp([feed_forward_node])
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
    print('num ops: ', len(ops))
    structure.set_ops(ops)
    structure.draw_graphviz('nt3_model.dot')

    model = structure.create_model()

    model = structure.create_model()
    plot_model(model, to_file='nt3_model.png', show_shapes=True)

    model.summary()

if __name__ == '__main__':
    test_create_structure()
