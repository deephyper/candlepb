import tensorflow as tf

from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.node import VariableNode, ConstantNode, MirrorNode
from deephyper.search.nas.model.space.structure import KerasStructure
from deephyper.search.nas.model.space.op.basic import Connect, AddByPadding
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

    def create_block(input_node):

        def add_mlp_ops_to(vnode):
            # n.add_op(Identity())
            # n.add_op(Conv1D(filter_size=3, num_filters=16))
            # n.add_op(MaxPooling1D(pool_size=3, padding='same'))
            # n.add_op(Dense(100, tf.nn.relu))
            # n.add_op(Dense(100, tf.nn.tanh))
            # n.add_op(Conv1D(filter_size=5, num_filters=16))
            # n.add_op(MaxPooling1D(pool_size=5, padding='same'))
            # n.add_op(Dense(500, tf.nn.relu))
            # n.add_op(Dense(500, tf.nn.tanh))
            # n.add_op(Conv1D(filter_size=10, num_filters=16))
            # n.add_op(MaxPooling1D(pool_size=10, padding='same'))
            vnode.add_op(Dense(1000, tf.nn.relu))
            # n.add_op(Dense(1000, tf.nn.tanh))

        # first node of block
        n1 = VariableNode('N1')
        add_mlp_ops_to(n1)
        cell.graph.add_edge(input_node, n1) # fixed input of current block

        # second node of block
        n2 = VariableNode('N2')
        add_mlp_ops_to(n2)

        # third node of the block
        n3 = VariableNode('N3')
        add_mlp_ops_to(n3)

        block = Block()
        block.add_node(n1)
        block.add_node(n2)
        block.add_node(n3)

        block.add_edge(n1, n2)
        block.add_edge(n2, n3)
        return block, (n1, n2, n3)

    block1, _ = create_block(input_nodes[0])
    block2, (vn1, vn2, vn3) = create_block(input_nodes[1])

   # first node of block
    m_vn1 = MirrorNode(node=vn1)
    cell.graph.add_edge(input_nodes[2], m_vn1) # fixed input of current block

    # second node of block
    m_vn2 = MirrorNode(node=vn2)

    # third node of the block
    m_vn3 = MirrorNode(node=vn3)

    block3 = Block()
    block3.add_node(m_vn1)
    block3.add_node(m_vn2)
    block3.add_node(m_vn3)

    block3.add_edge(m_vn1, m_vn2)
    block3.add_edge(m_vn2, m_vn3)

    cell.add_block(block1)
    cell.add_block(block2)
    cell.add_block(block3)

    # addNode = Node('Merging')
    # addNode.add_op(AddByPadding(cell.graph, addNode, cell.get_blocks_output()))
    # addNode.set_op(0)
    # cell.set_outputs(node=addNode)
    cell.set_outputs()
    return cell

def create_cell_2(input_nodes):
    """Create a cell with convolution.

    Args:
        input_nodes (list(Node)): a list of input_nodes for this cell.

    Returns:
        Cell: the corresponding cell.
    """
    cell = Cell(input_nodes)

    def create_block(input_node):

        def create_mlp_node(name):
            n = VariableNode(name)
            # n.add_op(Identity())
            # n.add_op(Conv1D(filter_size=3, num_filters=16))
            # n.add_op(MaxPooling1D(pool_size=3, padding='same'))
            # n.add_op(Dense(100, tf.nn.relu))
            # n.add_op(Dense(100, tf.nn.tanh))
            # n.add_op(Conv1D(filter_size=5, num_filters=16))
            # n.add_op(MaxPooling1D(pool_size=5, padding='same'))
            # n.add_op(Dense(500, tf.nn.relu))
            # n.add_op(Dense(500, tf.nn.tanh))
            # n.add_op(Conv1D(filter_size=10, num_filters=16))
            # n.add_op(MaxPooling1D(pool_size=10, padding='same'))
            n.add_op(Dense(1000, tf.nn.relu))
            # n.add_op(Dense(1000, tf.nn.tanh))
            return n

        # first node of block
        n1 = create_mlp_node('N1')
        cell.graph.add_edge(input_node, n1) # fixed input of current block

        # second node of block
        n2 = create_mlp_node('N2')

        n3 = create_mlp_node('N3')

        block = Block()
        block.add_node(n1)
        block.add_node(n2)
        block.add_node(n3)

        block.add_edge(n1, n2)
        block.add_edge(n2, n3)
        return block

    block = create_block(input_nodes[0])

    cell.add_block(block)

    # addNode = Node('Merging')
    # addNode.add_op(AddByPadding(cell.graph, addNode, cell.get_blocks_output()))
    # addNode.set_op(0)
    # cell.set_outputs(node=addNode)
    cell.set_outputs()
    return cell

def create_structure(input_shape=(2,), output_shape=(1,), *args, **kwargs):

    network = KerasStructure(input_shape, output_shape)
    input_nodes = network.input_nodes

    func = lambda: create_cell_1(input_nodes)
    network.add_cell_f(func)

    func = lambda x: create_cell_2(x)
    network.add_cell_f(func, num=1)

    return network

def test_create_structure():
    from random import random, seed
    from deephyper.search.nas.model.space.structure import KerasStructure
    from deephyper.core.model_utils import number_parameters
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    # seed(10)
    shapes = [(942, ), (3820, ), (3820, )]
    structure = create_structure(shapes, (1,))
    assert type(structure) is KerasStructure

    ops = [random() for i in range(structure.num_nodes)]
    # ops = [0 for i in range(structure.num_nodes)]
    print('num ops: ', len(ops))
    structure.set_ops(ops)
    structure.draw_graphviz('graph_candle_mlp_5.dot')

    model = structure.create_model()

    model = structure.create_model()
    plot_model(model, to_file='graph_candle_mlp_5.png', show_shapes=True)

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

    model.summary()
    # assert np.shape(y) == (1, 1), f'Wrong output shape {np.shape(y)} should be {(1, 1)}'

if __name__ == '__main__':
    test_create_structure()
