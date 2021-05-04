# Copyright 2020 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import random
import collections

import networkx as nx


Genotype = collections.namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


_opname_to_index = {
    'zero': 0,
    'skip_connect': 1,
    'max_pool_3x3': 2,
    'avg_pool_3x3': 3,
    'sep_conv_3x3': 4,
    'sep_conv_5x5': 5,
    'dil_conv_3x3': 6,
    'dil_conv_5x5': 7,
    'addition': 8,
    'input': 9,
    'output': 10,
    'global': 11
}


def get_matrix_and_ops(g, prune=True, keep_dims=False):
    '''
        Args:
            g : should be a Genotype object
    '''

    # Example g:
    # Genotype(normal=[
    #   ('avg_pool_3x3', 1),
    #   ('max_pool_3x3', 0),
    #   ('sep_conv_3x3', 1),
    #   ('avg_pool_3x3', 2),
    #   ('skip_connect', 3),
    #   ('skip_connect', 0),
    #   ('dil_conv_5x5', 2),
    #   ('avg_pool_3x3', 4)],
    #   ...

    graph = nx.DiGraph()

    def add_nodes(ops, concat):
        shift = len(graph.nodes)
        num_nodes = 4
        ops_per_node = 2
        num_inputs = 2

        darts_node_to_ops = { i: [shift + i] for i in range(num_inputs) }
        for i in range(num_inputs):
            graph.add_node(shift + i, label=_opname_to_index['input'])

        ops_iter = iter(ops)

        for node_id in range(num_inputs, num_inputs+num_nodes):
            this_node_ops = []
            for _ in range(ops_per_node):
                op, src = next(ops_iter)

                op_idx = len(graph.nodes)
                graph.add_node(op_idx, label=_opname_to_index[op])
                src_ops = darts_node_to_ops[src]
                for src_idx in src_ops:
                    graph.add_edge(src_idx, op_idx)
                this_node_ops.append(op_idx)

            addition_node = len(graph.nodes)
            graph.add_node(addition_node, label=_opname_to_index['addition'])
            for o_idx in this_node_ops:
                graph.add_edge(o_idx, addition_node)
            darts_node_to_ops[node_id] = [addition_node]

        merge_node_idx = len(graph.nodes)
        graph.add_node(merge_node_idx, label=_opname_to_index['output'])
        for src in concat:
            src_ops = darts_node_to_ops[src]
            for src_idx in src_ops:
                graph.add_edge(src_idx, merge_node_idx)

    add_nodes(g.normal, g.normal_concat)
    add_nodes(g.reduce, g.reduce_concat)

    if prune:
        for n, data in graph.nodes(data=True):
            assert int(data['label']) != 0
            if data['label'] == 1: # skip-connection
                for pred in graph.predecessors(n):
                    for suc in graph.successors(n):
                        graph.add_edge(pred, suc)
                graph.remove_edges_from([(n, suc) for suc in graph.successors(n)])
                graph.remove_edges_from([(pred, n) for pred in graph.predecessors(n)])
                data['label'] = None

    matrix = [[1 if graph.has_edge(src_idx, dst_idx) else 0 for dst_idx in range(len(graph.nodes))] for src_idx in range(len(graph.nodes))]
    labels = [None for _ in range(len(graph.nodes))]
    for n, data in graph.nodes(data=True):
        labels[n] = data['label']

    return matrix, labels


def get_adjacency_and_features(matrix, labels):
    # Add global node
    for row in matrix:
        row.insert(0, 0)

    nodes = len(matrix)
    global_row = [1 if i else 0 for i in range(nodes+1)] # zero followed by ones
    matrix.insert(0, global_row)

    labels.insert(0, _opname_to_index['global'])

    # Add diag matrix
    for idx, row in enumerate(matrix):
        row[idx] = 1

    possible_ops = len(_opname_to_index) - 2 # exclude "zero" and "skip_connect"

    # Create features matrix from labels
    features = [[0 for _ in range(possible_ops)] for _ in range(nodes+1)]
    for idx, op in enumerate(labels):
        if op is not None:
            features[idx][int(op)-2] = 1

    return matrix, features


def get_random_model():
    num_nodes = 4
    num_inputs = 2
    num_ops_per_node = 2
    possible_ops = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'
    ]

    def get_random_cell():
        ops = tuple(random.choices(possible_ops, k=num_ops_per_node * num_nodes))
        inputs = []
        possible_inputs = [i for i in range(num_inputs)]
        for node_idx in range(num_inputs, num_inputs + num_nodes):
            inputs.extend(random.sample(possible_inputs, k=num_ops_per_node))
            possible_inputs.append(node_idx)

        assert len(ops) == len(inputs)
        return tuple(zip(ops, tuple(inputs)))

    normal = get_random_cell()
    reduce = get_random_cell()
    return Genotype(normal, range(num_inputs, num_inputs + num_nodes), reduce, range(num_inputs, num_inputs + num_nodes))
