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
        self.loss_teach_list = []
        self.loss_rgl_list = []
        self.para_ud_list = []

    def forward(self, x):
        out = self.layers(x)
        return out
    
