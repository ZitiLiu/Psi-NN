# import numpy as np
# coding = utf-8
import torch
# from torch.utils.data import DataLoader
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D
# import os
# import Vis
# import time
  
class Net(nn.Module):
    def __init__(self, node_num:int, output_num:int = 1):
        super(Net, self).__init__()

        # 假令一个节点的复制数量为n
        self.node_num = node_num
        self.output_num = output_num

        self.fc1 = nn.Linear(1, 2*self.node_num)

        #第二层需要两种w
        self.fc2_1 = nn.Linear(2*self.node_num, self.node_num)
        self.fc2_2 = nn.Linear(2*self.node_num, self.node_num)

        # 第三层一种
        self.fc3 = nn.Linear(self.node_num, self.node_num)

        # 第四层一种
        self.fc4 = nn.Linear(self.node_num, self.output_num)

        #将损失存下来
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_b_list = []
        self.loss_d_list = []
        self.loss_rgl_list = []
        self.para_ud_list = []

    def forward(self, input):

        x, y = input[:,0:1], input[:,1:2]

        u1 = torch.tanh(self.fc1(x))
        u2 = torch.tanh(self.fc1(y))

        u2_1 = torch.tanh(self.fc2_1(u1) + self.fc2_2(u2))
        u2_2 = torch.tanh(self.fc2_1(u2) + self.fc2_2(u1))

        u = torch.tanh(self.fc3(u2_1) + self.fc3(u2_2))

        u = self.fc4(u)
       
        return u
    
    # def forward(self, input):
    #     x, y = input[:, 0:1], input[:, 1:2]
    #     psi = torch.tanh(self.fc1(x) + self.fc2(y))  # 标量势函数
    #     u1 = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]  # ∂ψ/∂y
    #     u2 = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]  # -∂ψ/∂x
    #     return torch.cat([u1, u2], dim=1)
    

# import torch

# # 定义输入网格
# x = torch.linspace(-1, 1, 100, requires_grad=True).reshape(-1, 1)  # 启用 requires_grad
# y = torch.linspace(-1, 1, 100, requires_grad=True).reshape(-1, 1)  # 启用 requires_grad
# grid = torch.cat([x, y], dim=1)

# # 计算网络输出
# net = Net(node_num=10)  # 假设节点数为10
# u = net(grid)
# u1, u2 = u[:, 0:1], u[:, 1:2]

# # 计算散度
# u1_x = torch.autograd.grad(u1.sum(), grid, create_graph=True)[0][:, 0:1]  # ∂u1/∂x
# u2_y = torch.autograd.grad(u2.sum(), grid, create_graph=True)[0][:, 1:2]  # ∂u2/∂y

# divergence = u1_x + u2_y  # 散度

# # 检查散度是否接近零
# print("Max divergence:", divergence.abs().max().item())



