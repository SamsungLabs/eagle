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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from .. import utils


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True, weight_init='thomas', bias_init='thomas'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self):
        utils.init_tensor(self.weight, self.weight_init, 'relu')
        utils.init_tensor(self.bias, self.bias_init, 'relu')

    def forward(self, adjacency, features):
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(Module):
    def __init__(self, 
                num_features=0, 
                num_layers=2,
                num_hidden=32,
                dropout_ratio=0,
                weight_init='thomas',
                bias_init='thomas',
                binary_classifier=False,
                augments=0):

        super(GCN, self).__init__()
        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.dropout_ratio = dropout_ratio
        self.gc = nn.ModuleList([GraphConvolution(self.nfeat if i==0 else self.nhid, self.nhid, bias=True, weight_init=weight_init, bias_init=bias_init) for i in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.LayerNorm(self.nhid).double() for i in range(self.nlayer)])
        self.relu = nn.ModuleList([nn.ReLU().double() for i in range(self.nlayer)])
        if not binary_classifier:
            self.fc = nn.Linear(self.nhid + augments, 1).double()
        else:
            if binary_classifier == 'naive':
                self.fc = nn.Linear(self.nhid + augments, 1).double()
            elif binary_classifier == 'oneway' or binary_classifier == 'oneway-hard':
                self.fc = nn.Linear((self.nhid + augments) * 2, 1).double()
            else:
                self.fc = nn.Linear((self.nhid + augments) * 2, 2).double()

            if binary_classifier != 'oneway' and binary_classifier != 'oneway-hard':
                self.final_act = nn.LogSoftmax(dim=1)
            else:
                self.final_act = nn.Sigmoid()

        self.dropout = nn.ModuleList([nn.Dropout(self.dropout_ratio).double() for i in range(self.nlayer)])

        self.binary_classifier = binary_classifier

    def forward_single_model(self, adjacency, features):
        x = self.relu[0](self.bn[0](self.gc[0](adjacency, features)))
        x = self.dropout[0](x)
        for i in range(1,self.nlayer):
            x = self.relu[i](self.bn[i](self.gc[i](adjacency, x)))
            x = self.dropout[i](x)

        return x

    def extract_features(self, adjacency, features, augments=None):
        x = self.forward_single_model(adjacency, features)
        x = x[:,0] # use global node
        if augments is not None:
            x = torch.cat([x, augments], dim=1)
        return x

    def regress(self, features, features2=None):
        if not self.binary_classifier:
            assert features2 is None
            return self.fc(features)

        assert features2 is not None
        if self.binary_classifier == 'naive':
            x1 = self.fc(features)
            x2 = self.fc(features2)
        else:
            x1 = features
            x2 = features2

        x = torch.cat([x1, x2], dim=1)
        if self.binary_classifier != 'naive':
            x = self.fc(x)

        x = self.final_act(x)
        return x

    def forward(self, adjacency, features, augments=None):
        if not self.binary_classifier:
            x = self.forward_single_model(adjacency, features)
            x = x[:,0] # use global node
            if augments is not None:
                x = torch.cat([x, augments], dim=1)
            return self.fc(x)
        else:
            x1 = self.forward_single_model(adjacency[:,0], features[:,0])
            x1 = x1[:,0]
            x2 = self.forward_single_model(adjacency[:,1], features[:,1])
            x2 = x2[:,0]
            if augments is not None:
                a1 = augments[:,0]
                a2 = augments[:,1]
                x1 = torch.cat([x1, a1], dim=1)
                x2 = torch.cat([x2, a2], dim=1)

            if self.binary_classifier == 'naive':
                x1 = self.fc(x1)
                x2 = self.fc(x2)

            x = torch.cat([x1, x2], dim=1)
            if self.binary_classifier != 'naive':
                x = self.fc(x)

            x = self.final_act(x)
            return x

    def reset_last(self):
        self.fc.reset_parameters()

    def final_params(self):
        return self.fc.parameters()
