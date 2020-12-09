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

import copy
import numpy as np


_opname_to_index = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4,
    'input': 5,
    'output': 6,
    'global': 7
}

_opindex_to_name = { value: key for key, value in _opname_to_index.items() }


def get_models_iter(start=None, maxpoints=None):
    def inc():
        for i in range(len(start)):
            start[i] += 1
            if start[i] >= 5:
                start[i] = 0
            else:
                break

    if not start:
        start = [0, 0, 0, 0, 0, 0]
    else:
        if len(start) != 6:
            raise ValueError('Invalid point')
        for p in start:
            if p < 0 or p >= 5:
                raise ValueError('Invalid point')

        inc()

    counter = 0
    while True:
        if maxpoints is not None:
            if counter > maxpoints:
                break
            counter += 1
        yield start
        inc()
        if start == [0, 0, 0, 0, 0, 0]:
            break


def get_total_models():
    return 5**6

def get_matrix_and_ops(g, prune=True, keep_dims=False):
    ''' Return the adjacency matrix and label vector.

        Args:
            g : should be a point from Nasbench102 search space
            prune : remove dangling nodes that only connected to zero ops
            keep_dims : keep the original matrix size after pruning
    '''

    matrix = [[0 for _ in range(8)] for _ in range(8)]
    labels = [None for _ in range(8)]
    labels[0] = 'input'
    labels[-1] = 'output'
    matrix[0][1] = matrix[0][2] = matrix[0][4] = 1
    matrix[1][3] = matrix[1][5] = 1
    matrix[2][6] = 1
    matrix[3][6] = 1
    matrix[4][7] = 1
    matrix[5][7] = 1
    matrix[6][7] = 1

    for idx, op in enumerate(g):
        if op == 0: # zero
            for other in range(8):
                if matrix[other][idx+1]:
                    matrix[other][idx+1] = 0
                if matrix[idx+1][other]:
                    matrix[idx+1][other] = 0
        elif op == 1: # skip-connection:
            to_del = []
            for other in range(8):
                if matrix[other][idx+1]:
                    for other2 in range(8):
                        if matrix[idx+1][other2]:
                            matrix[other][other2] = 1
                            matrix[other][idx+1] = 0
                            to_del.append(other2)
            for d in to_del:
                matrix[idx+1][d] = 0
        else:
            labels[idx+1] = str(op)
        
    if prune:
        visited_fw = [False for _ in range(8)]
        visited_bw = copy.copy(visited_fw)

        def bfs(beg, vis, con_f):
            q = [beg]
            vis[beg] = True
            while q:
                v = q.pop()
                for other in range(8):
                    if not vis[other] and con_f(v, other):
                        q.append(other)
                        vis[other] = True
                
        bfs(0, visited_fw, lambda src, dst: matrix[src][dst]) # forward
        bfs(7, visited_bw, lambda src, dst: matrix[dst][src]) # backward
    
        for v in range(7, -1, -1):
            if not visited_fw[v] or not visited_bw[v]:
                labels[v] = None
                if keep_dims:
                    matrix[v] = [0] * 8
                else:
                    del matrix[v]
                for other in range(len(matrix)):
                    if keep_dims:
                        matrix[other][v] = 0
                    else:
                        del matrix[other][v]
    
        if not keep_dims:
            labels = list(filter(lambda l: l is not None, labels))
            
        assert visited_fw[-1] == visited_bw[0]
        assert visited_fw[-1] == False or matrix
    
        verts = len(matrix)
        assert verts == len(labels)
        for row in matrix:
            assert len(row) == verts
    
    return matrix, labels

def get_adjacency_and_features(matrix, labels):
    # Add global node
    for row in matrix:
        row.insert(0, 0)
    global_row = [0, 1, 1, 1, 1, 1, 1, 1, 1]
    matrix.insert(0, global_row)
    # Add diag matrix
    for idx, row in enumerate(matrix):
        row[idx] = 1
    # Create features matrix from labels
    features = [[0 for _ in range(6)] for _ in range(9)]
    features[0][5] = 1 # global
    features[1][3] = 1 # input
    features[-1][4] = 1 # output
    for idx, op in enumerate(labels):
        if op != None and op != 'input' and op != 'output':
            features[idx+1][int(op)-2] = 1

    return matrix, features

def get_arch_vector_from_arch_str(arch_str):
    ''' Args:
            arch_str : a string representation of a cell architecture,
                for example '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|'
    '''

    nodes = arch_str.split('+')
    nodes = [node[1:-1].split('|') for node in nodes]
    nodes = [[op_and_input.split('~')[0]  for op_and_input in node] for node in nodes]

    # arch_vector is equivalent to a decision vector produced by autocaml when using Nasbench201 backend
    arch_vector = [_opname_to_index[op] for node in nodes for op in node]
    return arch_vector


def get_arch_str_from_arch_vector(arch_vector):
    ops = [_opindex_to_name[opindex] for opindex in arch_vector]
    return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*ops)
