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

import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, 
                num_nodes=0,
                num_features=0, 
                num_layers=2,
                num_hidden=32,
                dropout_ratio=0):

        super(MLP, self).__init__()
        self.nnodes = num_nodes
        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.dropout_ratio = dropout_ratio
        indepth = self.nnodes * self.nfeat + self.nnodes * self.nnodes
        self.mlp = nn.ModuleList([nn.Linear(indepth if i==0 else self.nhid, self.nhid).double() for i in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.LayerNorm(self.nhid).double() for i in range(self.nlayer)])
        self.relu = nn.ModuleList([nn.ReLU().double() for i in range(self.nlayer)])
        self.fc = nn.Linear(self.nhid, 1).double()
        self.dropout = nn.ModuleList([nn.Dropout(self.dropout_ratio).double() for i in range(self.nlayer)])

    def forward(self, adjacency, features):
        x = torch.cat((torch.flatten(features), torch.flatten(adjacency)), 0)
        x = self.relu[0](self.bn[0](self.mlp[0](x)))
        x = self.dropout[0](x)
        for i in range(1,self.nlayer):
            x = self.relu[i](self.bn[i](self.mlp[i](x)))
            x = self.dropout[i](x)
        return self.fc(x)

    def reset_last(self):
        self.fc.reset_parameters()
