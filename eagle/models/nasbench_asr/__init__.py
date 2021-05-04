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

import nasbench_asr as nbasr


_all_ops = nbasr.search_space.all_ops + ['input', 'output', 'global']
_opname_to_index = { name: idx for idx, name in enumerate(_all_ops) }


_hash_to_pt = {}


def init_module(cache_dir):
    global _hash_to_pt
    del cache_dir
    _hash_to_pt = { nbasr.search_space.get_model_hash(pt): pt for pt in nbasr.search_space.get_all_architectures() }


def get_matrix_and_ops(g, prune=True, keep_dims=False):
    pt = _hash_to_pt[g]
    graph, _ = nbasr.graph_utils.get_model_graph(pt, minimize=prune, keep_dims=keep_dims)
    adj, labels = graph
    adj = adj.tolist()
    return adj, labels


def get_adjacency_and_features(matrix, labels):
    # Add global node
    for row in matrix:
        row.insert(0, 0)

    nodes = len(matrix)
    global_row = [1 if i else 0 for i in range(nodes+1)] # zero followed by ones
    matrix.insert(0, global_row)

    labels.insert(0, 'global')

    # Add diag matrix
    for idx, row in enumerate(matrix):
        row[idx] = 1

    possible_ops = len(_opname_to_index) - 1 # exclude "zero"
    zero_idx = _opname_to_index['zero']

    # Create features matrix from labels
    features = [[0 for _ in range(possible_ops)] for _ in range(nodes+1)]
    for idx, op in enumerate(labels):
        if op is not None:
            op = int(_opname_to_index[op])
            if op >= zero_idx:
                op -= 1
            features[idx][op] = 1

    return matrix, features
