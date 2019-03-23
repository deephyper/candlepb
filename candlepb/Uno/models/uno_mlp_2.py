import tensorflow as tf

from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.node import (ConstantNode, MirrorNode,
                                                   VariableNode)
from deephyper.search.nas.model.space.op.basic import (AddByPadding, Connect,
                                                       Tensor)
from deephyper.search.nas.model.space.op.op1d import (Concatenate, Dense,
                                                      Dropout, Identity,
                                                      dropout_ops)
from deephyper.search.nas.model.space.structure import KerasStructure


def create_mlp_node(node):
    # node.add_op(Identity())
    node.add_op(Dense(100, tf.nn.relu))
    # node.add_op(Dense(100, tf.nn.tanh))
    # node.add_op(Dense(100, tf.nn.sigmoid))
    # node.add_op(Dropout(0.3))
    # node.add_op(Dense(500, tf.nn.relu))
    # node.add_op(Dense(500, tf.nn.tanh))
    # node.add_op(Dense(500, tf.nn.sigmoid))
    # node.add_op(Dropout(0.4))
    # node.add_op(Dense(1000, tf.nn.relu))
    # node.add_op(Dense(1000, tf.nn.tanh))
    # node.add_op(Dense(1000, tf.nn.sigmoid))
    # node.add_op(Dropout(0.5))

def set_cell_output_add(cell):
    addNode = ConstantNode(name='Merging')
    addNode.set_op(AddByPadding(cell.graph, addNode, cell.get_blocks_output()))
    cell.set_outputs(node=addNode)

def create_cell_1(input_nodes):
    """Create a cell with convolution.

    Args:
        input_nodes (list(Node)): a list of input_nodes for this cell.

    Returns:
        Cell: the corresponding cell.
    """
    cell = Cell(input_nodes)

    input_dose1 = input_nodes[0]
    input_rnaseq = input_nodes[1]
    input_drug1descriptor = input_nodes[2]
    input_drug1fingerprints = input_nodes[3]

    def create_block_3_nodes(input_node):

        # first node of block
        n1 = VariableNode('N1')
        create_mlp_node(n1)
        cell.graph.add_edge(input_node, n1) # fixed input of current block

        # second node of block
        n2 = VariableNode('N2')
        create_mlp_node(n2)

        # third node of the block
        n3 = VariableNode('N3')
        create_mlp_node(n3)

        block = Block()
        block.add_node(n1)
        block.add_node(n2)
        block.add_node(n3)

        block.add_edge(n1, n2)
        block.add_edge(n2, n3)
        return block, (n1, n2, n3)

    # BLOCK FOR: dose1
    n = ConstantNode(op=Identity(), name='N1', )
    cell.graph.add_edge(input_dose1, n)
    block0 = Block()
    block0.add_node(n)
    cell.add_block(block0)


    # BLOCK FOR: rnaseq
    block3, _ = create_block_3_nodes(input_rnaseq)
    cell.add_block(block3)

    # BLOCK FOR: drug1.descriptor
    block4, _ = create_block_3_nodes(input_drug1descriptor)
    cell.add_block(block4)

    # BLOCK FOR: drug1.fingerprints
    block5, _ = create_block_3_nodes(input_drug1fingerprints)
    cell.add_block(block5)

    # set_cell_output_add(cell)
    cell.set_outputs()
    return cell

def create_mlp_block(cell, input_node, skipco_inputs):
        block = Block()

        nullNode = ConstantNode(op=Tensor([]), name='None')
        def get_skipco_node():
            cnode = VariableNode(name='SkipCo')
            cnode.add_op(Connect(cell.graph, nullNode, cnode)) # SAME
            return cnode

        # first node of block
        n1 = VariableNode('N1')
        create_mlp_node(n1)
        cell.graph.add_edge(input_node, n1) # fixed input of current block
        block.add_node(n1)
        skipco_inputs.append(n1)

        # second node of block
        n2 = VariableNode('N2')
        create_mlp_node(n2)
        block.add_node(n2)

        mergeNode1 = VariableNode(name='Merge')
        mergeNode1.add_op(AddByPadding())
        mergeNode1.add_op(Concatenate())
        block.add_node(mergeNode1)

        block.add_edge(n1, n2)
        block.add_edge(n2, mergeNode1)
        #block.add_edge(n1, mergeNode1) # residual connection
        # replaced by skipco
        cnode = get_skipco_node()
        for inpt in skipco_inputs:
            cnode.add_op(Connect(cell.graph, inpt, cnode))
        block.add_node(cnode)
        block.add_edge(cnode, mergeNode1)

        skipco_inputs.append(n2)
        skipco_inputs.append(mergeNode1)

        n3 = VariableNode('N3')
        create_mlp_node(n3)
        block.add_node(n3)

        block.add_edge(mergeNode1, n3)

        mergeNode2 = VariableNode(name='Merge')
        mergeNode2.add_op(AddByPadding())
        mergeNode2.add_op(Concatenate())
        block.add_node(mergeNode2)

        block.add_edge(n3, mergeNode2)
        #block.add_edge(mergeNode1, mergeNode2) # residual connection
        # replaced by skipco
        cnode = get_skipco_node()
        for inpt in skipco_inputs:
            cnode.add_op(Connect(cell.graph, inpt, cnode))
        block.add_node(cnode)
        block.add_edge(cnode, mergeNode2)

        cell.add_block(block)
        return n1


def create_structure(input_shape=[(2,) for _ in range(4)], output_shape=(1,), num_cell=8, *args, **kwargs):

    network = KerasStructure(input_shape, output_shape) #, output_op=AddByPadding)
    input_nodes = network.input_nodes
    print('input_nodes: ', input_nodes)

    # CELL 1
    cell1 = create_cell_1(input_nodes)
    network.add_cell(cell1)

    # # CELL Middle
    pred_cell = cell1
    skipco_inputs = [input_nodes, input_nodes[0], input_nodes[1], input_nodes[2], input_nodes[3], pred_cell.output]

    cell_last = Cell([pred_cell.output])
    first_node = create_mlp_block(cell_last, pred_cell.output, skipco_inputs)
    set_cell_output_add(cell_last)

    network.add_cell(cell_last)


    return network

def test_create_structure():
    from random import random, seed
    from deephyper.search.nas.model.space.structure import KerasStructure
    from deephyper.core.model_utils import number_parameters
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    # seed(10)
    shapes = [
        (1, ),
        (942, ),
        (5270, ),
        (2048, )
    ]
    structure = create_structure(shapes, (1,))
    assert type(structure) is KerasStructure

    ops = [random() for i in range(structure.num_nodes)]

    print('num ops: ', len(ops))
    print(ops)
    structure.set_ops(ops)
    structure.draw_graphviz('uno_mlp_1.dot')

    model = structure.create_model()
    # model_json = model.to_json()
    # with open('model.json', 'w') as f:
    #     f.write(model_json)
    model = structure.create_model()
    plot_model(model, to_file='uno_mlp_1.png', show_shapes=True)

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

    model.summary()
    # assert np.shape(y) == (1, 1), f'Wrong output shape {np.shape(y)} should be {(1, 1)}'

if __name__ == '__main__':
    test_create_structure()
