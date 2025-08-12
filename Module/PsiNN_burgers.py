# coding = utf-8
import torch
# from torch.utils.data import DataLoader
import torch.nn as nn
  
class Net(nn.Module):
    def __init__(self, node_num:int, output_num: int = 1):
        super(Net, self).__init__()

        # 假令一个节点的复制数量为n
        self.node_num = node_num

        #第一层需要两种w
        self.fc1_1 = nn.Linear(1, self.node_num)
        # self.fc1_2 = nn.Linear(1, self.node_num)
        # 为了保证偏置的相同，这里将偏置置为0
        self.fc1_3 = nn.Linear(1, self.node_num, bias=False)    
        # self.fc1_4 = nn.Linear(1, self.node_num, bias=False)

        #第二层需要两种w
        self.fc2_1 = nn.Linear(self.node_num, self.node_num)
        self.fc2_2 = nn.Linear(self.node_num, self.node_num, bias=False)
        self.fc2_3 = nn.Linear(self.node_num, self.node_num)    
        # self.fc2_4 = nn.Linear(self.node_num, self.node_num)  

        #第三层需要一种w
        self.fc3_1 = nn.Linear(self.node_num, 2*self.node_num)
        self.fc3_2 = nn.Linear(self.node_num, 2*self.node_num, bias=False)

        #最后一层需要一种，这里由于是奇函数所以去掉最终的bias
        self.fc4_1 = nn.Linear(2*self.node_num, output_num)
 

        #将损失存下来
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_b_list = []
        self.loss_d_list = []
        self.loss_rgl_list = []
        self.para_ud_list = []

    # 定义数据的流动方式，要放在类的缩进里面
    def forward(self, input):

        x,y = input[:,0:1], input[:,1:2]

        u1_1 = torch.tanh(self.fc1_1(x) + self.fc1_3(y))
        u1_2 = torch.tanh(self.fc1_1(x) - self.fc1_3(y))
        # u1_3 = torch.tanh(self.fc1_2(x) + self.fc1_4(y))
        # u1_4 = torch.tanh(self.fc1_2(x) - self.fc1_4(y))

        #第二层
        u2_1 = torch.tanh(self.fc2_1(u1_1) + self.fc2_3(u1_2))
        u2_2 = torch.tanh(self.fc2_3(u1_1) + self.fc2_1(u1_2))

        u2_3 = torch.tanh(self.fc2_2(u1_1) - self.fc2_2(u1_2)) # 多的一个环节
        # u2_3 = torch.tanh(self.fc2_2(u1_3) + self.fc2_4(u1_4))
        # u2_4 = torch.tanh(self.fc2_4(u1_3) + self.fc2_2(u1_4))
        
        #第三层
        u3_1 = torch.tanh(self.fc3_1(u2_1) - self.fc3_1(u2_2) + self.fc3_2(u2_3))
        # u3_2 = torch.tanh(self.fc3_2(u2_3) + self.fc3_2(u2_4))



        #最后的输出
        u = self.fc4_1(u3_1) 
       
        return u