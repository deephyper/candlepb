import tensorflow as tf

from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.node import (ConstantNode, MirrorNode,
                                                   VariableNode)
from deephyper.search.nas.model.space.op.basic import Connect, Tensor, AddByPadding
from deephyper.search.nas.model.space.op.op1d import (Concatenate, Dense,
                                                      Dropout, Identity,
                                                      dropout_ops)
from deephyper.search.nas.model.space.structure import KerasStructure

def create_mlp_node(node):
    # node.add_op(Identity())
    # node.add_op(Dense(100, tf.nn.relu))
    # node.add_op(Dense(100, tf.nn.tanh))
    # node.add_op(Dense(100, tf.nn.sigmoid))
    # node.add_op(Dropout(0.3))
    # node.add_op(Dense(500, tf.nn.relu))
    # node.add_op(Dense(500, tf.nn.tanh))
    # node.add_op(Dense(500, tf.nn.sigmoid))
    # node.add_op(Dropout(0.4))
    # node.add_op(Dense(1000, tf.nn.relu))
    # node.add_op(Dense(1000, tf.nn.tanh))
    node.add_op(Dense(1000, tf.nn.sigmoid))
    # node.add_op(Dropout(0.5))

# def set_cell_output_add(cell):
#     addNode = ConstantNode(name='Merging')
#     addNode.set_op(AddByPadding(cell.graph, addNode, cell.get_blocks_output()))
#     cell.set_outputs(node=addNode)

def create_cell_1(input_nodes):
    """Create a cell with convolution.

    Args:
        input_nodes (list(Node)): a list of input_nodes for this cell.

    Returns:
        Cell: the corresponding cell.
    """
    cell = Cell(input_nodes)

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
    # input_nodes[0] == dose1
    n = ConstantNode(op=Identity(), name='N1', )
    cell.graph.add_edge(input_nodes[0], n)
    block0 = Block()
    block0.add_node(n)
    cell.add_block(block0)


    # BLOCK FOR: dose2
    # input_nodes[1] == dose2
    n = ConstantNode(op=Identity(), name='N1', )
    cell.graph.add_edge(input_nodes[1], n)
    block1 = Block()
    block1.add_node(n)
    cell.add_block(block1)


    # BLOCK FOR: response.source
    # input_nodes[2] == response.source
    n = ConstantNode(op=Identity(), name='N1')
    cell.graph.add_edge(input_nodes[2], n)
    block2 = Block()
    block2.add_node(n)
    cell.add_block(block2)


    # BLOCK FOR: rnaseq
    # input_nodes[3] == rnaseq
    block3, _ = create_block_3_nodes(input_nodes[3])
    cell.add_block(block3)

    # BLOCK FOR: drug1.descriptor
    # input_nodes[4] == drug1.descriptor
    block4, (vn1, vn2, vn3) = create_block_3_nodes(input_nodes[4])
    cell.add_block(block4)


    # BLOCK FOR: drug2.descriptor
    # input_nodes[6] == drug2.descriptor
    # first node of block
    m_vn1 = MirrorNode(node=vn1)
    cell.graph.add_edge(input_nodes[6], m_vn1) # fixed input of current block
    # second node of block
    m_vn2 = MirrorNode(node=vn2)
    # third node of the block
    m_vn3 = MirrorNode(node=vn3)

    block6 = Block()
    block6.add_node(m_vn1)
    block6.add_node(m_vn2)
    block6.add_node(m_vn3)
    block6.add_edge(m_vn1, m_vn2)
    block6.add_edge(m_vn2, m_vn3)
    cell.add_block(block6)


    # BLOCK FOR: drug1.fingerprints
    # input_nodes[5] == drug1.fingerprints
    block5, (vn1, vn2, vn3) = create_block_3_nodes(input_nodes[5])
    cell.add_block(block5)


    # BLOCK FOR: drug2.fingerprints
    # input_nodes[7] == drug2.fingerprints
    # first node of block
    m_vn1 = MirrorNode(node=vn1)
    cell.graph.add_edge(input_nodes[7], m_vn1) # fixed input of current block
    # second node of block
    m_vn2 = MirrorNode(node=vn2)
    # third node of the block
    m_vn3 = MirrorNode(node=vn3)

    block7 = Block()
    block7.add_node(m_vn1)
    block7.add_node(m_vn2)
    block7.add_node(m_vn3)
    block7.add_edge(m_vn1, m_vn2)
    block7.add_edge(m_vn2, m_vn3)
    cell.add_block(block7)


    # set_cell_output_add(cell)
    cell.set_outputs()
    return cell

def create_mlp_block(cell, input_node):

        # first node of block
        n1 = VariableNode('N1')
        create_mlp_node(n1)
        cell.graph.add_edge(input_node, n1) # fixed input of current block

        # second node of block
        n2 = VariableNode('N2')
        create_mlp_node(n2)

        n3 = VariableNode('N3')
        create_mlp_node(n3)

        block = Block()
        block.add_node(n1)
        block.add_node(n2)
        block.add_node(n3)

        block.add_edge(n1, n2)
        block.add_edge(n2, n3)
        return block

def create_structure(input_shape=[(2,) for _ in range(8)], output_shape=(1,), num_cell=8, *args, **kwargs):

    network = KerasStructure(input_shape, output_shape) #, output_op=AddByPadding)
    input_nodes = network.input_nodes
    print('input_nodes: ', input_nodes)

    # CELL 1
    cell1 = create_cell_1(input_nodes)
    network.add_cell(cell1)

    # # CELL Middle
    # inputs_skipco = [input_nodes, input_nodes[0], input_nodes[1], input_nodes[2], cell1.output]
    pred_cell = cell1
    # n = num_cell
    # for i in range(n):
    #     cell_i = Cell(input_nodes + [cell1.output])

    #     block1 = create_mlp_block(cell_i, pred_cell.output)
    #     cell_i.add_block(block1)

    #     cnode = VariableNode(name='SkipCo')
    #     nullNode = ConstantNode(op=Tensor([]), name='None')
    #     cnode.add_op(Connect(cell_i.graph, nullNode, cnode)) # SAME

    #     for inpt in inputs_skipco:
    #         cnode.add_op(Connect(cell_i.graph, inpt, cnode))

    #     block2 = Block()
    #     block2.add_node(cnode)
    #     cell_i.add_block(block2)
    #     # set_cell_output_add(cell2)
    #     cell_i.set_outputs()

    #     network.add_cell(cell_i)

    #     # prep. for next iter
    #     inputs_skipco.append(cell_i.output)
    #     pred_cell = cell_i


    # CELL LAST
    cell_last = Cell([pred_cell.output])
    block1 = create_mlp_block(cell_last, pred_cell.output)
    cell_last.add_block(block1)
    # set_cell_output_add(cell3)
    cell_last.set_outputs()
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
        (1, ),
        (9, ),
        (17750, ),
        (5270, ),
        (2048, ),
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
