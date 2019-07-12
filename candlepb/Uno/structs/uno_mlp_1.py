import tensorflow as tf

from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.merge import AddByPadding, Concatenate
from deephyper.search.nas.model.space.op.op1d import Dense, Dropout, Identity
from deephyper.search.nas.model.space.struct import AutoOutputStructure


def add_mlp_op_(node):
    node.add_op(Identity())
    node.add_op(Dense(100, tf.nn.relu))
    node.add_op(Dense(100, tf.nn.tanh))
    node.add_op(Dense(100, tf.nn.sigmoid))
    node.add_op(Dropout(0.3))
    node.add_op(Dense(500, tf.nn.relu))
    node.add_op(Dense(500, tf.nn.tanh))
    node.add_op(Dense(500, tf.nn.sigmoid))
    node.add_op(Dropout(0.4))
    node.add_op(Dense(1000, tf.nn.relu))
    node.add_op(Dense(1000, tf.nn.tanh))
    node.add_op(Dense(1000, tf.nn.sigmoid))
    node.add_op(Dropout(0.5))


def create_structure(input_shape=[(1, ), (942, ), (5270, ), (2048, )], output_shape=(1,), num_cells=2, *args, **kwargs):

    struct = AutoOutputStructure(input_shape, output_shape, regression=True)
    input_nodes = struct.input_nodes

    output_submodels = [input_nodes[0]]

    for i in range(1, 4):
        vnode1 = VariableNode('N1')
        add_mlp_op_(vnode1)
        struct.connect(input_nodes[i], vnode1)

        vnode2 = VariableNode('N2')
        add_mlp_op_(vnode2)
        struct.connect(vnode1, vnode2)

        vnode3 = VariableNode('N3')
        add_mlp_op_(vnode3)
        struct.connect(vnode2, vnode3)

        output_submodels.append(vnode3)

    merge1 = ConstantNode(name='Merge')
    merge1.set_op(Concatenate(struct, merge1, output_submodels))

    vnode4 = VariableNode('N4')
    add_mlp_op_(vnode4)
    struct.connect(merge1, vnode4)

    prev = vnode4

    for i in range(num_cells):
        vnode = VariableNode(f'N{i+1}')
        add_mlp_op_(vnode)
        struct.connect(prev, vnode)

        merge = ConstantNode(name='Merge')
        merge.set_op(AddByPadding(struct, merge, [vnode, prev]))

        prev = merge


    return struct

def test_create_structure():
    from random import random, seed
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

    ops = [random() for i in range(structure.num_nodes)]

    print('num ops: ', len(ops))
    print('size: ', structure.size)
    print(ops)
    structure.set_ops(ops)
    structure.draw_graphviz('uno_mlp_1.dot')

    model = structure.create_model()

    model = structure.create_model()
    plot_model(model, to_file='uno_mlp_1.png', show_shapes=True)


    model.summary()

if __name__ == '__main__':
    test_create_structure()
