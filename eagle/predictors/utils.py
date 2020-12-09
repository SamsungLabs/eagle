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

import math
from functools import reduce
from collections import defaultdict

import numpy as np
import torch.nn as nn


class ProductList():
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        i1, i2 = self.unmerge(idx)
        return self.values[i1], self.values[i2]

    def __len__(self):
        return len(self.values) * (len(self.values) - 1)

    def unmerge(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError()

        i1 = idx // (len(self.values) - 1)
        i2 = idx % (len(self.values) - 1)
        if i1 <= i2:
            i2 = (i2 + 1) % len(self.values)
        return i1, i2


#https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)

class SearchSpaceIterator:
    ''' Generic search space iterator that tracks points in the search space and their number '''
    def __init__(self, search_space, shuffle=False):
        '''
        search_space: a single list for now till needed to extend to multiple container case
        shuffle: shuffle
        '''
        self.search_space = search_space
        self.size = reduce(lambda x,y: x*y, self.search_space)
        self.points = (np.arange(0, self.size))
        
        if shuffle:
            np.random.shuffle(self.points)
        self.counter = 0

    def __len__(self):
        return len(self.search_space)

    def point_to_int(self, point):
        '''
        Converts point in search space to an int
        Inverse of get_point_from_int
        '''
        m = 1
        p = 0
        multiplier = self.search_space[::-1]
        for i, s in enumerate(point[::-1]):
            p += s * m
            m *= multiplier[i]
        
        return p
    
    def int_to_point(self, p):
        '''
        Converts p to a point in the search space, starting from right
        Example: 
        [4,4,4], 0 returns [0,0,0]
        [4,4,3], 1 returns [0,0,1]
        [2,3], 4 returns [1,1]
        '''
        assert p >= 0 and p < self.size
        point = [None] * len(self.search_space)
        for i, s in enumerate(self.search_space[::-1]):
            r = p % s
            point[i] = r 
            p = (p - r) // s

        return point[::-1]
    

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.size:
            point = self.int_to_point(self.points[self.counter])
            self.counter += 1
            return point
        else:
            raise StopIteration

def count_ops_edges(spec):
    d = defaultdict(int)
    m, ops = spec
    for o in ops[1:-1]:
        if o is not None:
            d[o] += 1
    nodes = len(m)
    for src in range(nodes):
        for dst in range(nodes):
            if m[src][dst] == 1:
                d['edges'] += 1
    return d

def valid(prediction, truth, leeway=0.05):
    p = prediction.item()
    t = truth.item()
    
    if t >= (1-leeway)*p and t <= (1+leeway)*p:
        return 1

    return 0

def normalize(mx):
    """Row-normalize sparse matrix"""
    mx = np.array(mx)
    rowsum = mx.sum(axis=1)
    r_inv = np.power(rowsum, -1.0).flatten() #use -1.0 as asym matrix
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    a = np.dot(r_mat_inv, mx)
    #a = np.dot(a, r_mat_inv) #skip for asym matrix
    #return a #normalized matrix
    return mx #unnormalized matrix


def init_tensor(tensor, init_type, nonlinearity):
    if tensor is None or init_type is None:
        return

    if init_type =='thomas':
        size = tensor.size(-1)
        stdv = 1. / math.sqrt(size)
        nn.init.uniform_(tensor, -stdv, stdv)
    elif init_type == 'kaiming_normal_in':
        nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_normal_out':
        nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_uniform_in':
        nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_uniform_out':
        nn.init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity=nonlinearity)
    elif init_type == 'orthogonal':
        nn.init.orthogonal_(tensor, gain=nn.init.calculate_gain(nonlinearity))
    else:
        raise ValueError(f'Unknown initialization type: {init_type}')
