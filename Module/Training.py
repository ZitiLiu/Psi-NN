# coding = utf-8
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import os
import importlib
import time
import itertools
import Module.PINN as PINN
import Module.SingleVis as SingleVis
import Module.GroupVis as GroupVis

torch.manual_seed(1234)  # 设置随机种子

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available")
else:
    device = torch.device('cpu')

class model():
    def __init__(self, ques_name, ini_num) :

        # 使用文件进行初始化    
        self.ques_name = ques_name
        self.ini_num = ini_num

        self.ini_file_path = f'./Config/{ques_name}_{ini_num}.csv'

        data = pd.read_csv(self.ini_file_path, header=None, names=['key', 'value'], usecols=[0, 1]) 

        self.model_ini_dict = {}
        for index, row in data.iterrows():
            key = row['key']
            value = row['value']

            # 含有min/max的一般是函数值，要变为float方便计算
            if 'min' in key or 'max' in key:    
                self.model_ini_dict[key] = float(value)

            # 含有num的一般是某某数量
            elif 'num' in key or 'state' in key:  
                self.model_ini_dict[key] = int(value)

            # 含有state的为int值，反正可以判断真假   
            elif 'state' in key:
                self.model_ini_dict[key] = int(value)

            # 默认是字符串
            else:   
                self.model_ini_dict[key] = str(value)        

        # 是否记录每步
        self.pace_record_state = self.model_ini_dict['pace_record_state']

        # 此处用于定义节点数
        self.node_num = self.model_ini_dict['node_num']

        self.coord_num = self.model_ini_dict['coord_num'] if 'coord_num' in self.model_ini_dict else self.model_ini_dict['input_num'] #兼容一下老Config文件
        self.output_num = self.model_ini_dict['output_num']


        # 定义模型学习率
        self.learning_rate = float(self.model_ini_dict['learning_rate']) if 'learning_rate' in self.model_ini_dict else 1e-4

        # 定义教师模型
        self.model_ini_dict['model'] = self.model_ini_dict['model'].split(' ')

        # 提前设定好这个参数空间的值！！！！
        self.x_min = self.model_ini_dict['x_min']
        self.x_max = self.model_ini_dict['x_max']
        self.y_min = self.model_ini_dict['y_min']
        self.y_max = self.model_ini_dict['y_max']
        self.z_min = self.model_ini_dict['z_min'] if 'z_min' in self.model_ini_dict else 0.
        self.z_max = self.model_ini_dict['z_max'] if 'z_max' in self.model_ini_dict else 0.

        # 设定一下可调节参数（组）
        # 将字符串按分号分隔
        self.para_ctrl_list = self.model_ini_dict['para_ctrl'].split(';')
        # 将每个分隔后的字符串按逗号分隔，并将每个元素转换为float
        self.para_ctrl_list = [list(map(float, item.split(','))) for item in self.para_ctrl_list]
        # 设定求解参数数量
        self.para_ctrl_num = len(self.para_ctrl_list)
        
        # 是否将参数作为追加输入
        self.para_ctrl_add = int(self.model_ini_dict['para_ctrl_add']) if 'para_ctrl_add' in self.model_ini_dict else False
        # 如果追加输入，则添加参数数量
        self.input_num = self.coord_num + self.para_ctrl_num if self.para_ctrl_add else self.coord_num

        # 读取隐藏层的节点组数
        self.hidden_layers_group = list(map(float, self.model_ini_dict['hidden_layers_group'].split(',')))
        self.layer = [self.input_num, self.output_num]
        self.layer[1:1] = list(map(lambda x: x * self.node_num, self.hidden_layers_group))  # 这里是插入的意思
        self.layer = list(map(int, self.layer))

        # 设定使用的数据库下标
        # self.model_ini_dict['data_serial'] = list(map(int, self.model_ini_dict['data_serial'].split(',')))
        self.model_ini_dict['data_serial'] = list(self.model_ini_dict['data_serial'].split(','))

        self.data_serial = self.model_ini_dict['data_serial']
        
        # 整体的计算场边界上节点数
        self.grid_node_num = self.model_ini_dict['grid_node_num']  

        # 监督值的有无，直接根据文件名标识，不要重复定义
        self.monitor_state = True if 'inv' in self.ques_name or 'global' in self.ques_name else False

        # 正则化的有无
        self.regular_state = self.model_ini_dict['regularization_state']

        # 是否为加载模型
        self.load_state = self.model_ini_dict['load_state']

        # 迭代步数组数     
        self.step_num = self.model_ini_dict['step_num'] if self.model_ini_dict['step_num'] < 10 else 1

        # 边界上使用的节点数
        self.bun_node_num = self.model_ini_dict['bun_node_num']

        # 画图使用的节点数
        self.figure_node_num = self.model_ini_dict['figure_node_num']    

        # 是否蒸馏学习这里就不多重定义了，直接根据文件名标识
        self.distill_state = True if 'distill' in self.ques_name else False
        # 蒸馏学习应该是也有正逆问题的，所以不能删减inv标识

        # self.distill_state = self.model_ini_dict['distill_state'] if 'distill_state' in self.model_ini_dict else False

        # 由于蒸馏学习是PINN，给个使用的模型参数，跟上面是一样的
        if self.distill_state:
            # 显示蒸馏状态
            print(f'Distill state: {self.distill_state}')
            self.layer_student = [self.input_num, self.output_num]
            self.hidden_layers_group_student = list(map(float, self.model_ini_dict['hidden_layers_group_student'].split(',')))
            self.layer_student[1:1] = list(map(lambda x: x * self.node_num, self.hidden_layers_group_student))

            self.layer_student = list(map(int, self.layer_student))  

        # 迭代率下降
        self.milestone = list(map(int, self.model_ini_dict['milestone'].split(','))) if 'milestone' in self.model_ini_dict else None
        self.gamma = float(self.model_ini_dict['gamma']) if 'gamma' in self.model_ini_dict else 0.5

        # 记录的间隔步数
        self.pace_record_gap = list(map(int, self.model_ini_dict['pace_record_gap'].split(','))) if 'pace_record_gap' in self.model_ini_dict else 100

        # 一些步数之后记录间隔变大
        self.pace_record_skip = list(map(int, self.model_ini_dict['pace_record_skip'].split(','))) if 'pace_record_skip' in self.model_ini_dict else self.train_steps /2 
        ## 结果发现实际上模型的体积比较小，所以都存了得了

        # 学生网络是否直接学习加载的现有网络
        self.load_study_state = self.model_ini_dict['load_study_state'] if 'load_study_state' in self.model_ini_dict else False

        # 教师和学生步数
        if int(self.model_ini_dict['step_num']) > 10000:
            self.train_steps = int(self.model_ini_dict['step_num']) 
        elif int(self.model_ini_dict['step_num']) < 1000:
            self.train_steps = int(self.model_ini_dict['train_steps'])
        else:
            self.train_steps = 100000

        # 比例
        self.train_ratio = float(self.model_ini_dict['train_ratio']) if 'train_ratio' in self.model_ini_dict else 0.5

        # 在此处统一定义存储路径
        self.save_desti = f'./Results/{self.ques_name}_{str(self.ini_num)}/'

        # 消融实验追加
        # 学生网络不加正则化
        self.study_regularization_state = self.model_ini_dict['study_regularization_state'] if 'study_regularization_state' in self.model_ini_dict else True

        # k值控制，加入这个值之后分离教师输出和观测值损失
        self.k_value = float(self.model_ini_dict['k_value']) if 'k_value' in self.model_ini_dict else 0.0


        # 这里表示要不要加流动中的p值参数，默认是加
        if 'Flow' in self.ques_name:
            self.flow_p_add = int(self.model_ini_dict['flow_p_add']) if 'flow_p_add' in self.model_ini_dict else 1
            self.cylinder_weight = float(self.model_ini_dict['cylinder_weight']) if 'cylinder_weight' in self.model_ini_dict else 1
            self.bcs_weight = float(self.model_ini_dict['bcs_weight']) if 'bcs_weight' in self.model_ini_dict else 1


    # 这里定义一下计算场
    def mesh_init(self):
        if self.coord_num == 3:
            self.x = np.linspace(self.x_min, self.x_max, self.grid_node_num).reshape([-1,1])
            self.y = np.linspace(self.y_min, self.y_max, self.grid_node_num).reshape([-1,1])
            self.z = np.linspace(self.z_min, self.z_max, self.grid_node_num).reshape([-1,1])
            self.x, self.y, self.z = np.meshgrid(self.x, self.y, self.z)
            self.x = torch.tensor(self.x,requires_grad=True).float().to(device).reshape([-1,1])
            self.y = torch.tensor(self.y,requires_grad=True).float().to(device).reshape([-1,1])
            self.z = torch.tensor(self.z,requires_grad=True).float().to(device).reshape([-1,1])
        
        elif 'Flow' in self.ques_name:
            # 流动问题的计算点就是flow
            fluid_data = pd.read_csv(f'./Database/flow/fluid_data.csv').values
            self.x = torch.tensor(fluid_data[:,0], requires_grad=True).float().to(device).reshape([-1,1])
            self.y = torch.tensor(fluid_data[:,1], requires_grad=True).float().to(device).reshape([-1,1])
            # y值需要减少 0.2以对齐坐标轴
            # in order to align the coordinate axis, we need to subtract 0.2 from the y values
            self.y -= 0.2

        else:   
            self.x = torch.linspace(self.x_min, self.x_max, self.grid_node_num, requires_grad=True).float().to(device)
            self.y = torch.linspace(self.y_min, self.y_max, self.grid_node_num, requires_grad=True).float().to(device)
            self.x, self.y = torch.meshgrid(self.x, self.y, indexing='ij')
            self.x = self.x.reshape([-1, 1])
            self.y = self.y.reshape([-1, 1])

            # 实际上，这里只需要把网格点算出来后，在他们前面加入一个维度就行了，这里给出所有可能的参数组合
            if self.para_ctrl_add:
                # 使用itertools.product生成所有可能的组合
                combinations = list(itertools.product(*self.para_ctrl_list))

                # 将每个组合转换为torch.Tensor
                self.para_ctrl_tensors = [torch.tensor(combination, dtype=torch.float).to(device) for combination in combinations]

    # 边界条件损失
    def net_b(self):    
        loss_b = torch.tensor(0.).to(device)

        # 流动的边界条件
        if 'Flow' in self.ques_name:
            # 首先读取边界数据，x,y,p,u,v排列的
            cylinder_data = pd.read_csv(f'./Database/flow/cylinder_data.csv').values
            inlet_data = pd.read_csv(f'./Database/flow/inlet_data.csv').values
            outlet_data = pd.read_csv(f'./Database/flow/outlet_data.csv').values
            wall_data = pd.read_csv(f'./Database/flow/wall_data.csv').values
            '''
            根据边界数据进行计算，约束如下
            1, 入口速度u 有固定值 v = 0 直接减
            2， 上下边界，以及壁面速度u=v = 0
            3， 出口压力p = 0
            '''
            # 首先把所有的y值减少0.2以对齐坐标轴
            # in order to align the coordinate axis, we need to subtract 0.2 from the y values
            inlet_data[:,1] -= 0.2
            wall_data[:,1] -= 0.2
            outlet_data[:,1] -= 0.2
            cylinder_data[:,1] -= 0.2

            # 入口边界
            xy_in = torch.tensor(inlet_data[:,0:2], requires_grad=True).float().to(device)
            uv_in = torch.tensor(inlet_data[:,3:5], requires_grad=True).float().to(device)

            loss_b_in = ((self.net(xy_in)[:, 1:3] - uv_in)**2).mean() # 入口速度u有固定值，v=0


            # 圆柱面边界
            xy_cylinder = torch.tensor(cylinder_data[:,0:2], requires_grad=True).float().to(device)

            loss_b_cylinder_uv = ((self.net(xy_cylinder)[:, 1:3])**2).mean()  # 圆柱面速度为0

            # 加入壁面压力损失项
            p_cylinder = torch.tensor(cylinder_data[:,2], requires_grad=True).float().to(device)
            if self.flow_p_add:
                # 圆柱面压力固定
                loss_b_cylinder_p = ((self.net(xy_cylinder)[:, 0] - p_cylinder)**2).mean()
            else:
                loss_b_cylinder_p = 0

            # 上下边界
            xy_wall = torch.tensor(wall_data[:,0:2], requires_grad=True).float().to(device)
            loss_b_wall = ((self.net(xy_wall)[:, 1:3])**2).mean()  # 上下边界速度为0

            # 出口压力
            xy_out = torch.tensor(outlet_data[:,0:2], requires_grad=True).float().to(device)
            p_out = torch.tensor(outlet_data[:,2], requires_grad=True).float().to(device)
            # loss_b_out = ((self.net(xy_out)[:, 0] - p_out)**2).mean()  # 出口压力为0
            loss_b_out = ((self.net(xy_out)[:, 0])**2).mean()  # 出口压力为0

            # 圆柱面系数已在Config文件中定义
            loss_b += loss_b_in + self.cylinder_weight * (loss_b_cylinder_uv + loss_b_cylinder_p) + loss_b_wall + loss_b_out
            # print(f'Flow boundary loss: {loss_b.item()}')
            
            return loss_b, loss_b_in, loss_b_cylinder_uv, loss_b_cylinder_p, loss_b_wall, loss_b_out
        else:

            if 'Poisson' in self.ques_name:
                self.bun_node_num = 1000

            #x最小,y任意
            y_b = torch.linspace(self.y_min, self.y_max, self.bun_node_num, requires_grad=True).float().to(device).reshape([-1,1])
            x_b = torch.full_like(y_b, self.x_min, requires_grad=True).float().to(device).reshape([-1,1])
            u_b = self.net(torch.cat([x_b, y_b], dim=1))

            # y=最小,x任意
            x_down = torch.linspace(self.x_min, self.x_max, self.bun_node_num, requires_grad=True).float().to(device).reshape([-1,1])
            y_down = torch.full_like(x_down, self.y_min, requires_grad=True).float().to(device).reshape([-1,1])
            u_down = self.net(torch.cat([x_down, y_down], dim=1))

            # y=最大,x任意
            x_up = torch.linspace(self.x_min, self.x_max, self.bun_node_num, requires_grad=True).float().to(device).reshape([-1,1])
            y_up = torch.full_like(x_up, self.y_max, requires_grad=True).float().to(device).reshape([-1,1])
            u_up = self.net(torch.cat([x_up, y_up], dim=1))

            # x最大,y任意
            y_f = torch.linspace(self.y_min, self.y_max, self.bun_node_num, requires_grad=True).float().to(device).reshape([-1,1])
            x_f = torch.full_like(y_f, self.x_max, requires_grad=True).float().to(device).reshape([-1,1])
            u_f = self.net(torch.cat([x_f, y_f], dim=1))

            if 'Burgers' in self.ques_name:
                u_b_moni = -torch.sin(torch.pi * y_b) # burgers
                loss_b += torch.mean((u_b - u_b_moni)**2)

                if 'half' in self.ques_name:
                    # 只算一半，边界条件ydown就是-1
                    x_down = torch.linspace(self.x_min, self.x_max, self.bun_node_num, requires_grad=True).float().to(device).reshape([-1,1])
                    y_down = -torch.ones_like(x_down, requires_grad=True).float().to(device).reshape([-1,1])
                    u_down = self.net(torch.cat([x_down, y_down], dim=1))
                    
                u_down_moni = torch.zeros_like(u_down)  #burgers
                loss_b += torch.mean((u_down - u_down_moni)**2)

                u_up_moni = torch.zeros_like(u_up)  #burgers
                loss_b += torch.mean((u_up - u_up_moni)**2)
         #     loss_b += torch.mean((u_up - u_up_moni)**2)

            elif 'Laplace' in self.ques_name:
                u_b_moni = (x_b**3 - 3*x_b*y_b**2)  #laplace
                loss_b += torch.mean((u_b - u_b_moni)**2)

                u_down_moni = (x_down**3 - 3*x_down*y_down**2)
                loss_b += torch.mean((u_down - u_down_moni)**2)

                u_up_moni = (x_up**3 - 3*x_up*y_up**2)
                loss_b += torch.mean((u_up - u_up_moni)**2)

                u_f_moni = (x_f**3 - 3*x_f*y_f**2)
                loss_b += torch.mean((u_f - u_f_moni)**2)
            
            elif 'Poisson' in self.ques_name:
                x_total = torch.cat([x_b, x_down, x_up, x_f], dim=0)
                y_total = torch.cat([y_b, y_down, y_up, y_f], dim=0)
                u_total = self.net(torch.cat([x_total, y_total], dim=1))
                loss_b += torch.mean((u_total)**2)
            
        return loss_b
    
    def net_f(self):
        
        loss_f = torch.tensor(0.).to(device)
        u = self.net(torch.cat([self.x, self.y], dim=1)).to(device)

        # print(f'ques_name: {self.ques_name}')
        if 'Flow' in self.ques_name:

            rho = 1.0
            mu = 0.02

            p,u,v = torch.split(u, 1, dim=1)
            u_x = torch.autograd.grad(u, self.x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_y = torch.autograd.grad(u, self.y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, self.x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, self.y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

            v_x = torch.autograd.grad(v, self.x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
            v_y = torch.autograd.grad(v, self.y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
            v_xx = torch.autograd.grad(v_x, self.x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
            v_yy = torch.autograd.grad(v_y, self.y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

            p_x = torch.autograd.grad(p, self.x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
            p_y = torch.autograd.grad(p, self.y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

            # 连续性方程
            eq0 = u_x + v_y

            # x方向动量方程
            eq1 = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)

            # y方向动量方程
            eq2 = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)

            # 方程损失，需要转化为标量
            loss_f += torch.mean((eq0)**2) + torch.mean((eq1)**2) + torch.mean((eq2)**2)

            return loss_f


        u_x = torch.autograd.grad(u, self.x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]    #这就是自动微分的一整个公式，直接照着抄就行了
        u_xx = torch.autograd.grad(u_x, self.x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, self.y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, self.y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        u_yyy = torch.autograd.grad(u_yy, self.y, grad_outputs=torch.ones_like(u_yy), retain_graph=True, create_graph=True)[0]

        #方程误差
        if 'Burgers' in self.ques_name:
            if 'inv' in self.ques_name:
                loss_f = torch.mean((u_x + u*u_y - self.para_undetermin[0]* u_yy)**2)
                # print(loss_f)
            else:
                # print(self.para_ctrl_list)
                loss_f = torch.mean((u_x + u*u_y - self.para_ctrl_list[0][0] / torch.pi * u_yy)**2)
                # loss_f = torch.mean((u_x + u*u_y - 0.01 / torch.pi * u_yy)**2)
        
        elif 'Laplace' in self.ques_name:
            if 'inv' in self.ques_name:
                loss_f = torch.mean((u_xx + self.para_undetermin[0] * u_yy)**2)
            else:
                loss_f = torch.mean((u_xx + u_yy)**2)

        elif 'Poisson' in self.ques_name:
            k = torch.arange(1, 5).to(device)
            f = sum([1/2*((-1)**(k+1))*(k**2) * (torch.sin(k * torch.pi * (self.x)) * torch.sin(k * torch.pi * (self.y))) for k in k])


            if 'inv' in self.ques_name:
                loss_f = torch.mean((u_xx + self.para_undetermin[0] * u_yy - f )**2)
            else:
                loss_f = torch.mean((u_xx + u_yy - f)**2)
        # else:
        #     raise ValueError('The input ' + self.ques_name + ' is unintegrated or the question name is incorrect. Please check again.')
        
        return loss_f
        
    def net_rgl(self, mode = 'teacher', object = 'all', reg_type ='l2', weight_rgl = 1e-3):
        loss_rgl = torch.tensor(0.).to(device)

        if mode == 'teacher':
            parameters_rgl = self.net.named_parameters()
        elif mode == 'student':
            if not self.study_regularization_state:
                return loss_rgl
            parameters_rgl = self.net_student.named_parameters()

        if object == 'all':
            for name, param in parameters_rgl:
                if reg_type == 'l2':
                    loss_rgl += weight_rgl * torch.norm(param, p=2)
                elif reg_type == 'l1':
                    loss_rgl += weight_rgl * torch.norm(param
                    , p=1)
                

        elif object == 'weight':
            for name, param in parameters_rgl:
                if 'weight' in name:
                    if reg_type == 'l2':
                        loss_rgl += weight_rgl * torch.norm(param, p=2)
                    elif reg_type == 'l1':
                        loss_rgl += weight_rgl * torch.norm(param, p=1)
                    elif reg_type == 'growl':
                    # 按行计算 2 范数
                        row_norms = torch.norm(param, p=2, dim=1)
                        
                        # 按行范数降序排列
                        sorted_row_norms, _ = torch.sort(row_norms, descending=True)
                        
                        # 如果没有提供 lambda_vals，则自动生成
                        lambda_vals = torch.linspace(1, 0.1, steps=sorted_row_norms.size(0)).to(device)
                        
                        # 确保 lambda_vals 的长度与行数匹配
                        lambda_vals = lambda_vals[:sorted_row_norms.size(0)]
                        
                        # 计算 GrOWL 正则化项
                        loss_rgl += torch.sum(lambda_vals * sorted_row_norms)
        return loss_rgl

    # 已知全场数据的监督误差（知道解析式或者有数据）
    def net_global(self, state:bool=False):

        loss_global = torch.tensor(0.).to(device)

        if 'Laplace' in self.ques_name:
            # u = self.net(self.x, self.y).to(device)
            u = self.net(torch.cat([self.x, self.y], dim=1)).to(device)
            loss_global += torch.mean((u - (self.x)**3 + 3 * self.x * self.y **2) **2)

        elif 'Poisson' in self.ques_name:
            u = self.net(torch.cat([self.x, self.y], dim=1)).to(device)
            
            u_moni = 0.5 / (2*torch.pi**2) * ((torch.sin(torch.pi * (self.x)) * torch.sin(torch.pi * (self.y)))- (2 * torch.sin(2 * torch.pi * (self.x)) * torch.sin(2 * torch.pi * (self.y))) + (3 * torch.sin(3 * torch.pi * (self.x)) * torch.sin(3 * torch.pi * (self.y))) - (4 * torch.sin(4 * torch.pi * (self.x)) * torch.sin(4 * torch.pi * (self.y))))
            
            # lf 是low frequency的意思
            if 'lf' in self.ques_name:
                # print('lf')
                u_moni = torch.sin(torch.pi * (self.x)) * torch.sin(torch.pi * (self.y)) + torch.sin(2*torch.pi * (self.x)) * torch.sin(2*torch.pi * (self.y))

            loss_global += torch.mean((u_moni - u) ** 2)

        else:
            # print(f'Reading precise database for {self.ques_name}: ./Database/{self.ques_name}_data.csv')
            self.precise_database = pd.read_csv('./Database/'+self.ques_name + '_data.csv').values
            self.x_monitor = self.precise_database[: , 0:self.coord_num].reshape([-1,self.coord_num])
            self.u_monitor = self.precise_database[: , self.coord_num : self.output_num + self.coord_num].reshape([-1,self.output_num])
            self.x_monitor = torch.tensor(self.x_monitor,requires_grad=True).float().to(device)
            self.u_monitor = torch.tensor(self.u_monitor,requires_grad=True).float().to(device)

            u = self.net(self.x_monitor).to(device)

            loss_global += torch.mean((u - self.u_monitor)**2)
            
        return loss_global, state


    # 有无监督值可以根据state判断。
    def net_d(self, mode = 'teacher'):
        loss_d = torch.tensor(0.).to(device)

        # 这里采用一个临时的将名称分离的策略
        ques_name = self.ques_name.split('_')[0]

        
        # 由于要堆叠，所以这里先弄一个进去
        # current_read = pd.read_csv(f'./Database/{self.ques_name}_data_{self.data_serial[0]}.csv', header=None).values
        current_read = pd.read_csv(f'./Database/{ques_name}_inv_data_{self.data_serial[0]}.csv', header=None).values

        self.database = current_read
        for i in range(1, len(self.data_serial)):
            # current_read = pd.read_csv(f'./Database/{self.ques_name}_data_{self.data_serial[i]}.csv', header=None).values
            current_read = pd.read_csv(f'./Database/{ques_name}_inv_data_{self.data_serial[i]}.csv', header=None).values
            self.database = np.vstack([self.database,current_read])
        
        # 由于这些监督值往往在存储的时候已经是meshgrid之后的状态，所以这里就不再进行meshgrid
        self.input_monitor = self.database[:,0:self.input_num].reshape([-1,self.input_num])
        self.u_monitor = self.database[:,self.input_num:].reshape([-1,self.output_num])
        self.input_monitor = torch.tensor(self.input_monitor,requires_grad=True).float().to(device)
        self.u_monitor = torch.tensor(self.u_monitor,requires_grad=True).float().to(device)



        if mode == 'student':

            # 这里进行融合
            # 先计算该点教师的值
            if self.k_value > 0:

                self.teacher_monitor_value = self.net(self.input_monitor)

                fai = 1 - torch.tanh(self.k_value * torch.abs(self.teacher_monitor_value - self.u_monitor))

                # loss_d = torch.mean((self.net_student(self.input_monitor) - fai * self.teacher_monitor_value - (1-fai) * self.u_monitor)**2)
                loss_d = torch.mean(((1-fai) * (self.net_student(self.input_monitor) - self.u_monitor))**2)

                return loss_d
            else:
                return loss_d
            # loss_d += torch.mean((self.net_student(self.input_monitor) - self.u_monitor)**2)
            # return loss_d

        if self.net.__module__.split('.')[-1] == 'PINN_post_divfree':
            output = self.net(self.input_monitor)
            output = torch.autograd.grad(output, self.input_monitor, grad_outputs=torch.ones_like(output), retain_graph=True, create_graph=True)[0]
            u = torch.cat((-output[:,1:2], output[:,0:1]), dim=1)
            loss_d += torch.mean((u - self.u_monitor)**2)
        else:
            # loss_d += torch.mean((self.net(self.x_monitor,self.y_monitor) - self.u_monitor)**2)
            # print(self.input_monitor.shape)
            loss_d += torch.mean((self.net(self.input_monitor) - self.u_monitor)**2)
        
        return loss_d
    

    def net_teach(self, weight_teach = 1):

               
        if self.para_ctrl_add:
            current_para_ctrl_tensors = [para_ctrl_tensor.repeat(self.x.shape[0], 1) for para_ctrl_tensor in self.para_ctrl_tensors]
            for i in range (len(self.para_ctrl_tensors)):
                u_teacher = self.net(torch.cat([self.x, self.y, current_para_ctrl_tensors[i]], dim=1))
                u_student = self.net_student(torch.cat([self.x, self.y, current_para_ctrl_tensors[i]], dim=1))
                return torch.mean((u_teacher - u_student)**2) * weight_teach

        if self.coord_num == 3:
            # print(type(self.net))
            # u_teacher = self.net(self.x,self.y,self.z).to(device)
            u_teacher = self.net(torch.cat([self.x, self.y, self.z], dim=1)).to(device)
            # u_student = self.net_student(self.x,self.y,self.z).to(device)
            u_student = self.net_student(torch.cat([self.x, self.y, self.z], dim=1)).to(device)
        else:
            xy_cat = torch.cat([self.x, self.y], dim=1)

            # 现在需要区分教师模型中已有观测值的坐标点，这里由于包含self.input_monitor，所以调用的时候必须要把net_d放在前面
            if self.k_value > 0:
                # 计算每个xy_cat行是否与input_monitor任一行完全相同
                mask = torch.ones(xy_cat.shape[0], dtype=torch.bool, device=xy_cat.device)
                for row in self.input_monitor:
                    same = torch.all(torch.isclose(xy_cat, row, atol=1e-8), dim=1)
                    mask = mask & (~same)  # 只保留不相等的行

                # 去除这些行
                xy_cat = xy_cat[mask]


            u_teacher = self.net(xy_cat).to(device)
            u_student = self.net_student(xy_cat).to(device)
        
        return torch.mean((u_teacher - u_student)**2) * weight_teach

  
    def train_adam(self):
        self.para_undetermin = torch.zeros(self.para_ctrl_num, requires_grad=True).float().to(device)
        self.para_undetermin = torch.nn.Parameter(self.para_undetermin)

        if 'Poisson' in self.ques_name:
            self.learning_rate = 1e-3
        
        self.optimizer = optim.Adam(list(self.net.parameters()) + [self.para_undetermin], lr=self.learning_rate)

       
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestone, gamma=self.gamma)

        if self.distill_state:
            self.optimizer_student = optim.Adam(list(self.net_student.parameters()), lr=self.learning_rate) 

        self.current_time = time.time()
        self.time_list = [0.]

        for iter_group in range(self.step_num):  

            for iter_inner in range(self.train_steps): 

                self.optimizer.zero_grad() 

                if self.load_study_state:
                    break
                self.loss_f = self.net_f()           
                if 'inv' in self.ques_name:
                    if 'Poisson' in self.ques_name:
                        self.loss_d = self.net_global()[0]
                    else:
                        self.loss_d = self.net_d()
                else:
                    self.loss_d = torch.tensor(0.).to(device)
                self.loss_b = torch.tensor(0.).to(device) if self.monitor_state else self.net_b()
                self.loss_rgl = self.net_rgl(object='all', reg_type='l2') if self.regular_state else torch.tensor(0.).to(device)

                if self.monitor_state: 
                    self.loss = self.loss_d + self.loss_f
                else:
                    if 'Flow' in self.ques_name :
                        self.loss = self.loss_f + self.loss_b[0]
                    else:
                        self.loss = self.loss_f + self.loss_b

                if self.regular_state:
                    self.loss += self.loss_rgl
                

                if 'Flow' in self.ques_name:
                    loss = self.loss_f + self.bcs_weight * self.loss_b[0]
                    loss.backward(retain_graph=True) 
                else:
                    self.loss.backward(retain_graph=True)

                self.optimizer.step()    
                self.scheduler.step()


                self.net.iter += 1
                self.net.iter_list.append(self.net.iter)
                self.net.loss_list.append(self.loss.item())
                self.net.loss_f_list.append(self.loss_f.item())
                if 'Flow' in self.ques_name:
                    self.loss_b_origion = self.loss_b[1] + self.loss_b[2] + self.loss_b[3] + self.loss_b[4] + self.loss_b[5]
                    self.net.loss_b_list.append(self.loss_b_origion.item())
                else:
                    self.net.loss_b_list.append(self.loss_b.item())
                self.net.loss_d_list.append(self.loss_d.item())
                self.net.loss_rgl_list.append(self.loss_rgl.item())

                if self.monitor_state:
                    self.net.para_ud_list.append(self.para_undetermin.tolist())
  
                if self.net.iter -1 in self.pace_record_skip:
                    iter_index_teacher = self.pace_record_skip.index(self.net.iter -1)
                    current_gap_teacher = self.pace_record_gap[iter_index_teacher]
                
                if 'Flow' in self.ques_name:
                    self.loss_dict = {'Iter':self.net.iter, 'Loss':self.loss.item(), 'Loss_f':self.loss_f.item(), 'Loss_b':self.loss_b_origion.item(), 'Loss_d':self.loss_d.item(), 'Loss_rgl':self.loss_rgl.item()}
                else:
                    self.loss_dict = {'Iter':self.net.iter, 'Loss':self.loss.item(), 'Loss_f':self.loss_f.item(), 'Loss_b':self.loss_b.item(), 'Loss_d':self.loss_d.item(), 'Loss_rgl':self.loss_rgl.item()}

                if self.net.iter % current_gap_teacher == 0:
                    total_iter = self.step_num * self.train_steps  
                    loss_str = ', '.join([f'{key}: {int(value) if key == "Iter" else value:.5e}' for key, value in self.loss_dict.items() if key != "Iter" and value != 0])
                    iter_str = f'Iter: {{{self.net.iter}/{total_iter}}}'  
                    print(f'{iter_str}, {loss_str}')
                    if self.pace_record_state:
                        self.model_save(str(self.net.iter))

                    if 'Flow' in self.ques_name:
                        print(f"loss_b_in: {self.loss_b[1]:.5e}, loss_b_cylinder_uv: {self.loss_b[2]:.5e}, loss_b_cylinder_p: {self.loss_b[3]:.5e}, loss_b_wall: {self.loss_b[4]:.5e}, loss_b_out: {self.loss_b[5]:.5e}")
        
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if current_lr != self.original_lr:
                        print(f"Learning rate changed from {self.original_lr:.6f} to {current_lr:.6f}")
                    self.original_lr = current_lr


                    
                
                self.time_list[0] += time.time() - self.current_time
                self.current_time = time.time()

            if self.distill_state:


                for iter_inner in range(int(self.train_steps * self.train_ratio)):
                    
                    self.optimizer_student.zero_grad()

                    self.loss_student_d = self.net_d(mode='student')

                    self.loss_teach = self.net_teach()
                    self.loss_student_rgl = self.net_rgl(mode='student', object='weight')

                    self.loss_student = self.loss_student_d + self.loss_teach + self.loss_student_rgl 

                    self.loss_student.backward(retain_graph=True)

                    self.optimizer_student.step()

                    self.net_student.iter += 1
                    self.net_student.iter_list.append(self.net_student.iter)
                    self.net_student.loss_list.append(self.loss_student.item())
                    self.net_student.loss_teach_list.append(self.loss_teach.item())
                    self.net_student.loss_d_list.append(self.loss_student_d.item())
                    self.net_student.loss_rgl_list.append(self.loss_student_rgl.item())

                
                    if self.net_student.iter - 1  in self.pace_record_skip:
                        iter_index_student = self.pace_record_skip.index(self.net_student.iter - 1)
                        current_gap_student = self.pace_record_gap[iter_index_student]
                    if self.net_student.iter % current_gap_student == 0: 
                        total_iter_student = int(self.step_num * self.train_steps * self.train_ratio) 
                        iter_str_student = f'Iter (student): {{{self.net_student.iter}/{total_iter_student}}}'

                        loss_str_student = ', '.join([f'{key}: {value:.5e}' for key, value in {
                            'loss_student': self.loss_student.item(),
                            'loss_teach': self.loss_teach.item(),
                            'loss_rgl': self.loss_student_rgl.item(),
                            'loss_student_d': self.loss_student_d.item()
                        }.items() if value != 0])
                        print(f'{iter_str_student}, {loss_str_student}')
                        
                        if self.pace_record_state:
                            self.model_save(str(self.net_student.iter), mode='student')

                if len(self.time_list) == 1:
                    self.time_list.append(0.)
                self.time_list[1] += time.time() - self.current_time
                self.current_time = time.time()
                
        print(f'\nTime occupied: {(self.time_list[0]):.5e} s.\n')
        if self.distill_state:
            print(f'\nTime occupied (student): {(self.time_list[1]):.5e} s.\n')
    def model_save(self, suffix:str ='', mode:str='teacher'):

        if not os.path.exists(f'./Results/'):
            os.mkdir(f'./Results/')


        if not os.path.exists(self.save_desti):
            os.mkdir(self.save_desti)
        if not os.path.exists(f'{self.save_desti}/Models/'):       
            os.mkdir(f'{self.save_desti}/Models/')

        if mode == 'teacher':
            in_net = self.net
            suffix_mode = ''
        elif mode == 'student':
            in_net = self.net_student
            suffix_mode = '_student'
        else:
            raise ValueError("Invalid mode. Choose either 'teacher' or 'student'.")
        
        if suffix == '':
            torch.save(in_net.state_dict(), f"{self.save_desti}/Models/{self.ques_name}_{self.ini_num}_{in_net.__module__.split('.')[-1]}{suffix_mode}.pth")
        elif self.pace_record_state:
            torch.save(in_net.state_dict(), f"{self.save_desti}/Models/{self.ques_name}_{self.ini_num}_{in_net.__module__.split('.')[-1]}{suffix_mode}_step_{suffix}.pth")

        # 复制控制参数（Config内容）
        # if not os.path.exists(f'{self.save_desti}{self.ques_name}_{self.ini_num}.csv'):
        self.control_paras = pd.read_csv(self.ini_file_path)
        self.control_paras.to_csv(f'{self.save_desti}{self.ques_name}_{self.ini_num}.csv', index=False)
        


        # 存储时间，最后一步才存
        if suffix == '':
            self.time_save = pd.DataFrame({
                'Question': [self.ques_name],
                'Number': [self.ini_num],
                'Module': [in_net.__module__.split('.')[-1]],
                'Training Time': [self.time_list[0]],
                'Student Training Time': [self.time_list[1]] if self.distill_state else [0.]
            })
            file_path = self.save_desti + 'Clock time.csv'
            if not os.path.isfile(file_path):
                self.time_save.to_csv(self.save_desti + 'Clock time.csv', mode='a', index=False)    # mode='a'表示追加写入
            else:
                self.time_save.to_csv(self.save_desti + 'Clock time.csv', mode='a', index=False, header=False)

        if mode == 'teacher':

            loss_data_dict = {
                'iter': self.net.iter_list,
                'loss': self.net.loss_list,
                'loss_f': self.net.loss_f_list,
                'loss_b': self.net.loss_b_list,
                'loss_d': self.net.loss_d_list,
                'loss_rgl': self.net.loss_rgl_list
            }

            loss_data_dict = {key: value for key, value in loss_data_dict.items() if value != 0}


            df_loss_data = pd.DataFrame(loss_data_dict)

            df_loss_data = df_loss_data.loc[:, (df_loss_data != 0).any(axis=0)]
            
        if self.distill_state and mode == 'student':
            loss_student_data_dict = {
                'iter': self.net_student.iter_list,
                'loss': self.net_student.loss_list,
                'loss_teach': self.net_student.loss_teach_list,
                'loss_rgl': self.net_student.loss_rgl_list,
                'loss_student_d': self.net_student.loss_d_list
            }

            loss_student_data_dict = {key: value for key, value in loss_student_data_dict.items() if value != 0}

            df_loss_student_data = pd.DataFrame(loss_student_data_dict)

            df_loss_student_data = df_loss_student_data.loc[:, (df_loss_student_data != 0).any(axis=0)]
        
        if not os.path.exists(self.save_desti + '/Loss/'):       
            os.mkdir(self.save_desti + '/Loss/')

        if mode == 'teacher':
            df_loss_data.to_csv(f"{self.save_desti}/Loss/{self.ques_name}_{str(self.ini_num)}_loss_{self.net.__module__.split('.')[-1]}.csv", index=False) 
            # print(f'\n Teacher model loss data saved.\n')

        if self.distill_state and mode == 'student':
            df_loss_student_data.to_csv(f"{self.save_desti}/Loss/{self.ques_name}_{str(self.ini_num)}_loss_{self.net_student.__module__.split('.')[-1]}_student.csv", index=False)

        # 存储算出来的参数
        if self.monitor_state:
            if mode == 'teacher':
                iter_list = np.array(self.net.iter_list).reshape([-1,1])
                para_ud = np.array(np.hstack([iter_list, self.net.para_ud_list]))
                # para_ud = np.transpose(para_ud)
                para_ud_columns = ['iter']
                for i in range(self.para_ctrl_num):
                    para_ud_columns.append('parameters_'+str(i+1))
                df_para_ud = pd.DataFrame(para_ud, columns = para_ud_columns)
                if not os.path.exists(self.save_desti + '/Parameters/'):       
                    os.mkdir(self.save_desti + '/Parameters/')
                df_para_ud.to_csv(f"{self.save_desti}/Parameters/{self.ques_name}_{str(self.ini_num)}_paras_{self.net.__module__.split('.')[-1]}.csv", index=False, mode='a' if self.load_state else 'w')


    def result_show(self):
        x = np.linspace(self.x_min, self.x_max, self.figure_node_num).reshape([-1,1])
        y = np.linspace(self.y_min, self.y_max, self.figure_node_num).reshape([-1,1])
        z = np.linspace(self.z_min, self.z_max, self.figure_node_num).reshape([-1,1]) if self.coord_num == 3 else None
        if self.coord_num == 3:
            x, y, z = np.meshgrid(x, y, z)
        elif 'Flow' in self.ques_name :
            x, y = self.x.detach().cpu().numpy(), self.y.detach().cpu().numpy()
        else:
            x, y = np.meshgrid(x, y)
        
        input = torch.tensor(np.concatenate([x.reshape([-1,1]), y.reshape([-1,1])], axis=1),
        dtype=torch.float32, requires_grad=True).float().to(device) if self.coord_num == 2 else torch.tensor(np.concatenate([x.reshape([-1,1]), y.reshape([-1,1]), z.reshape([-1,1])], axis=1),
        dtype=torch.float32, requires_grad=True).float().to(device)
        u = self.net(input)


        if self.net.__module__.split('.')[-1] == 'PINN_post_divfree':
            output = torch.autograd.grad(u, input, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u = torch.cat([-output[:,1:2], output[:,0:1]], dim=1)

        if self.distill_state:
            u_student = self.net_student(input)
            u_student = u_student.detach().cpu().numpy()

        input = input.detach().cpu().numpy()

        u = u.detach().cpu().numpy()

        u_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti, self.net.__module__.split('.')[-1], input, u)
        u_vis.figure_2d() if self.coord_num == 2 else u_vis.figure_3d()
        if not self.load_study_state:
            u_vis.loss_vis()

        if self.distill_state:
            u_student_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti, self.net_student.__module__.split('.')[-1], input, u_student, mode='student')
            u_student_vis.figure_2d() if self.coord_num == 2 else u_student_vis.figure_3d()
            u_student_vis.loss_vis()

        if self.monitor_state:
            u_vis.para_vis()

    def workflow(self):
        self.mesh_init()
        self.train_adam()
        self.model_save() 
        if self.distill_state:
            self.model_save(mode='student')
        if not self.para_ctrl_add:
            self.result_show()

    def train(self): 

        model_define_trigger = 0
        
        if len(self.model_ini_dict['model']) > 1:
            group = GroupVis.Vis(self.ques_name, self.ini_num, self.save_desti)

        for i in range (len(self.model_ini_dict['model'])):

            self.original_lr = 1e-3 if 'Poisson' in self.ques_name else self.learning_rate

            model_define_trigger = 1
            module = importlib.import_module(f"Module.{self.model_ini_dict['model'][i]}")
            NetClass = getattr(module, 'Net')

            if 'PINN' in self.model_ini_dict['model'][i]:
                self.net = NetClass(self.layer).float().to(device)
            else:
                self.net = NetClass(self.node_num, self.output_num).float().to(device)

            if self.load_state:
                load_path = f"./Results/{self.ques_name}_{self.ini_num}/Models/{self.ques_name}_{self.ini_num}_{self.net.__module__.split('.')[-1]}.pth"
                self.net.load_state_dict(torch.load(load_path))

            if self.distill_state:
                self.net_student = PINN.Net(self.layer_student).float().to(device)
            
            print(f'\nRunning Model: {self.model_ini_dict["model"][i]}\n')

            self.workflow()


            if len(self.model_ini_dict['model']) > 1:
                group.loss_read(self.net.__module__.split('.')[-1])
                if self.monitor_state:
                    group.para_read(self.net.__module__.split('.')[-1])

        if len(self.model_ini_dict['model']) > 1:
            group.loss_vis()
            if self.monitor_state:
                group.para_vis()

        if model_define_trigger == 0:
            raise ValueError('The model name is incorrect. Please check again.')