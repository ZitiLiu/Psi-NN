# import numpy as np
# coding = utf-8
import torch
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import pandas as pd
from collections import OrderedDict

class Net(torch.nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_d_list = []
        self.loss_b_list = []
        self.loss_rgl_list = []
        self.para_ud_list = []

    def forward(self, input):
        x, y = input[:, 0:1], input[:, 1:2]
        output_1 = self.layers(torch.cat([x, y], dim=1))
        output_2 = self.layers(torch.cat([x, -y], dim=1))
        p, u, v = output_1[:, 0:1] + output_2[:, 0:1], output_1[:, 1:2] + output_2[:, 1:2], output_1[:, 2:3] - output_2[:, 2:3]
        output = torch.cat([p, u, v], dim=1)
        return output