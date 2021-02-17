__author__ = "Martin Willoch Olstad"
__email__ = "martinwilloch@gmail.com"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueFunctionApproximation(nn.Module):
    def __init__(self, act_func=None, out_func=None, input_dim=None, output_dim=None, hidden_dims=None):
        super(ValueFunctionApproximation, self).__init__()

        self.act_func = F.relu if act_func is None else act_func
        self.out_func = F.linear if out_func is None else out_func

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = self.initialize_layers(input_dim, output_dim, hidden_dims)
        self.depth = len(self.layers)

        self.float()

    def forward(self, x):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float()
        for i in range(self.depth):
            x = self.layers[i](x.float())
            x = self.act_func(x) if i != self.depth-1 else x
        return x

    @staticmethod
    def initialize_layers(input_dim, output_dim, hidden_dims):
        layers = []
        if len(hidden_dims) == 0:
            layer = nn.Linear(input_dim, output_dim, bias=True)
            layers.append(layer)
            return nn.ModuleList(layers)
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dims[i], bias=True)
                layers.append(layer)
            elif i == len(hidden_dims):
                layer = nn.Linear(hidden_dims[i - 1], output_dim, bias=True)
                layers.append(layer)
            else:
                layer = nn.Linear(hidden_dims[i - 1], hidden_dims[i], bias=True)
                layers.append(layer)
        return nn.ModuleList(layers)
