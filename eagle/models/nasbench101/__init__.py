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

import pickle
import pathlib


_opname_to_index = {
    'none': 0,
    'conv3x3-bn-relu': 1,
    'conv1x1-bn-relu': 2,
    'maxpool3x3': 3,
    'input': 4,
    'output': 5,
    'global': 6
}

_model_info = None


def get_model_info(nb101_dataset, cache_dir):
    if cache_dir is not None:
        nbmodels_cache = pathlib.Path(cache_dir) / 'nb101_models_info.pickle'
        if nbmodels_cache.exists():
            with nbmodels_cache.open('rb') as f:
                return pickle.load(f)

    if nb101_dataset:
        import nasbench.api as nbapi
        nasbench = nbapi.NASBench(nb101_dataset)
        model_info = { model_hash: (stats['module_adjacency'], stats['module_operations'])  for model_hash, stats in nasbench.fixed_statistics.items() }
        if cache_dir is not None:
            with nbmodels_cache.open('wb') as f:
                pickle.dump(model_info, f)

        return model_info

    return None


def init_module(cache_dir, nb101_dataset=None):
    global _model_info
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
    _model_info = get_model_info(nb101_dataset, cache_dir)
    if _model_info is None:
        raise ValueError('Cannot find information about NB101 models')


def get_matrix_and_ops(g, prune=True, keep_dims=False):
    ''' Return the adjacency matrix and label vector.

        Args:
            g : should be a point from Nasbench101 search space
            prune : remove dangling nodes that only connected to zero ops
            keep_dims : keep the original matrix size after pruning
    '''
    matrix, labels = _model_info[g]
    matrix = matrix.tolist()

    #sometimes we have none's already in the labels (TODO dunno why)
    #HACK will remove here
    while 'none' in labels:
        labels.remove('none')

    while len(labels) < 7:
        labels.insert(-1,'none')
        for row in matrix:
            row.insert(-1,0)
        matrix.insert(-1,[0]*len(matrix[0]))

    return matrix, labels


def get_adjacency_and_features(matrix, labels):
    # Add global node
    for row in matrix:
        row.insert(0, 0)
    global_row = [0, 1, 1, 1, 1, 1, 1, 1]
    matrix.insert(0, global_row)

    # Add diag matrix
    for idx, row in enumerate(matrix):
        row[idx] = 1
    # Create features matrix from labels
    features = [[0 for _ in range(len(_opname_to_index.keys()))] for _ in range(len(global_row))]

    #global
    features[0][_opname_to_index['global']] = 1

    for idx, l in enumerate(labels):
        op = _opname_to_index[l]
        features[idx+1][int(op)] = 1

    assert len(matrix) == len(global_row)
    assert len(features) == len(global_row)

    return matrix, features
