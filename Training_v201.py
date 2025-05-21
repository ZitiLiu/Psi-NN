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

torch.manual_seed(1234)  

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class model():
    def __init__(self, ques_name, ini_num) :
 
        self.ques_name = ques_name
        self.ini_num = ini_num

        self.ini_file_path = f'./Config/{ques_name}_{ini_num}.csv'

        data = pd.read_csv(self.ini_file_path, header=None, names=['key', 'value'], usecols=[0, 1]) 

        self.model_ini_dict = {}
        for index, row in data.iterrows():
            key = row['key']
            value = row['value']

            if 'min' in key or 'max' in key:    
                self.model_ini_dict[key] = float(value)

            elif 'num' in key or 'state' in key:  
                self.model_ini_dict[key] = int(value)

            elif 'state' in key:
                self.model_ini_dict[key] = int(value)

            else:   
                self.model_ini_dict[key] = str(value)        

        self.pace_record_state = self.model_ini_dict['pace_record_state']

        self.node_num = self.model_ini_dict['node_num']

        self.coord_num = self.model_ini_dict['coord_num'] if 'coord_num' in self.model_ini_dict else self.model_ini_dict['input_num'] 
        self.output_num = self.model_ini_dict['output_num']


        self.learning_rate = 1e-4

        self.model_ini_dict['model'] = self.model_ini_dict['model'].split(' ')

        self.x_min = self.model_ini_dict['x_min']
        self.x_max = self.model_ini_dict['x_max']
        self.y_min = self.model_ini_dict['y_min']
        self.y_max = self.model_ini_dict['y_max']
        self.z_min = self.model_ini_dict['z_min'] if 'z_min' in self.model_ini_dict else 0.
        self.z_max = self.model_ini_dict['z_max'] if 'z_max' in self.model_ini_dict else 0.

        self.para_ctrl_list = self.model_ini_dict['para_ctrl'].split(';')
        self.para_ctrl_list = [list(map(float, item.split(','))) for item in self.para_ctrl_list]
        self.para_ctrl_num = len(self.para_ctrl_list)
        
        self.para_ctrl_add = int(self.model_ini_dict['para_ctrl_add']) if 'para_ctrl_add' in self.model_ini_dict else False
        self.input_num = self.coord_num + self.para_ctrl_num if self.para_ctrl_add else self.coord_num

        self.hidden_layers_group = list(map(float, self.model_ini_dict['hidden_layers_group'].split(',')))
        self.layer = [self.input_num, self.output_num]
        self.layer[1:1] = list(map(lambda x: x * self.node_num, self.hidden_layers_group))  
        self.layer = list(map(int, self.layer))

        self.model_ini_dict['data_serial'] = list(self.model_ini_dict['data_serial'].split(','))

        self.data_serial = self.model_ini_dict['data_serial']
        
        self.grid_node_num = self.model_ini_dict['grid_node_num']  

        self.monitor_state = True if 'inv' in self.ques_name or 'global' in self.ques_name else False

        self.regular_state = self.model_ini_dict['regularization_state']

        self.load_state = self.model_ini_dict['load_state']

        self.step_num = self.model_ini_dict['step_num'] if self.model_ini_dict['step_num'] < 10 else 1

        self.bun_node_num = self.model_ini_dict['bun_node_num']

        self.figure_node_num = self.model_ini_dict['figure_node_num']    

        self.distill_state = True if 'distill' in self.ques_name else False

        if self.distill_state:
            print(f'Distill state: {self.distill_state}')
            self.layer_student = [self.input_num, self.output_num]
            self.hidden_layers_group_student = list(map(float, self.model_ini_dict['hidden_layers_group_student'].split(',')))
            self.layer_student[1:1] = list(map(lambda x: x * self.node_num, self.hidden_layers_group_student))

            self.layer_student = list(map(int, self.layer_student))  

        self.milestone = list(map(int, self.model_ini_dict['milestone'].split(','))) if 'milestone' in self.model_ini_dict else None
        self.gamma = float(self.model_ini_dict['gamma']) if 'gamma' in self.model_ini_dict else 0.5

        self.pace_record_gap = list(map(int, self.model_ini_dict['pace_record_gap'].split(','))) if 'pace_record_gap' in self.model_ini_dict else 100

        self.pace_record_skip = list(map(int, self.model_ini_dict['pace_record_skip'].split(','))) if 'pace_record_skip' in self.model_ini_dict else self.train_steps /2 

        self.load_study_state = self.model_ini_dict['load_study_state'] if 'load_study_state' in self.model_ini_dict else False

        if int(self.model_ini_dict['step_num']) > 10000:
            self.train_steps = int(self.model_ini_dict['step_num']) 
        elif int(self.model_ini_dict['step_num']) < 1000:
            self.train_steps = int(self.model_ini_dict['train_steps'])
        else:
            self.train_steps = 100000

        self.train_ratio = float(self.model_ini_dict['train_ratio']) if 'train_ratio' in self.model_ini_dict else 0.5

        self.save_desti = f'./Results/{self.ques_name}_{str(self.ini_num)})/'


    def mesh_init(self):
        if self.coord_num == 3:
            self.x = np.linspace(self.x_min, self.x_max, self.grid_node_num).reshape([-1,1])
            self.y = np.linspace(self.y_min, self.y_max, self.grid_node_num).reshape([-1,1])
            self.z = np.linspace(self.z_min, self.z_max, self.grid_node_num).reshape([-1,1])
            self.x, self.y, self.z = np.meshgrid(self.x, self.y, self.z)
            self.x = torch.tensor(self.x,requires_grad=True).float().to(device).reshape([-1,1])
            self.y = torch.tensor(self.y,requires_grad=True).float().to(device).reshape([-1,1])
            self.z = torch.tensor(self.z,requires_grad=True).float().to(device).reshape([-1,1])
        
        else:  
            self.x = torch.linspace(self.x_min, self.x_max, self.grid_node_num, requires_grad=True).float().to(device)
            self.y = torch.linspace(self.y_min, self.y_max, self.grid_node_num, requires_grad=True).float().to(device)
            self.x, self.y = torch.meshgrid(self.x, self.y, indexing='ij')
            self.x = self.x.reshape([-1, 1])
            self.y = self.y.reshape([-1, 1])
            if self.para_ctrl_add:
                combinations = list(itertools.product(*self.para_ctrl_list))
                self.para_ctrl_tensors = [torch.tensor(combination, dtype=torch.float).to(device) for combination in combinations]

    def net_b(self):    
        loss_b = torch.tensor(0.).to(device)

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

            u_down_moni = torch.zeros_like(u_down)  #burgers
            loss_b += torch.mean((u_down - u_down_moni)**2)

            u_up_moni = torch.zeros_like(u_up)  #burgers
            loss_b += torch.mean((u_up - u_up_moni)**2)

        elif 'AC' in self.ques_name:
            u_b_moni = (y_b**2) * torch.cos(torch.pi * y_b) # AC
            loss_b += torch.mean((u_b - u_b_moni)**2)

            u_up = self.net(x_up, y_up)
            loss_b += torch.mean((u_up - u_down)**2)

            u_up_y = torch.autograd.grad(u_up, y_up, grad_outputs=torch.ones_like(u_up), retain_graph=True, create_graph=True)[0] #AC
            u_down_y = torch.autograd.grad(u_down, y_down, grad_outputs=torch.ones_like(u_down), retain_graph=True, create_graph=True)[0]   #AC
            loss_b += torch.mean((u_down_y - u_up_y)**2)
            

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
    
    # 方程损失
    def net_f(self):

        loss_f = torch.tensor(0.).to(device)
        # self.para_ctrl = torch.tensor(self.para_ctrl,requires_grad=True).float().to(device)

        #############此处让global类先不加loss_f
        if 'global' in self.ques_name:
            return loss_f
        
        u = self.net(torch.cat([self.x, self.y], dim=1)).cuda()

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
            # 源项
            k = torch.arange(1, 5).to(device)
            f = sum([1/2*((-1)**(k+1))*(k**2) * (torch.sin(k * torch.pi * (self.x)) * torch.sin(k * torch.pi * (self.y))) for k in k])


            if 'inv' in self.ques_name:
                loss_f = torch.mean((u_xx + self.para_undetermin[0] * u_yy - f )**2)
            else:
                loss_f = torch.mean((u_xx + u_yy - f)**2)
        
        return loss_f
    
    def net_rgl(self, mode = 'teacher', object = 'all', reg_type ='l2', weight_rgl = 1e-3):
        loss_rgl = torch.tensor(0.).to(device)

        if mode == 'teacher':
            parameters_rgl = self.net.named_parameters()
        elif mode == 'student':
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
            u = self.net(torch.cat([self.x, self.y], dim=1)).cuda()
            loss_global += torch.mean((u - (self.x)**3 + 3 * self.x * self.y **2) **2)

        elif 'Poisson' in self.ques_name:
            u = self.net(torch.cat([self.x, self.y], dim=1)).cuda()
            
            u_moni = 0.5 / (2*torch.pi**2) * ((torch.sin(torch.pi * (self.x)) * torch.sin(torch.pi * (self.y)))- (2 * torch.sin(2 * torch.pi * (self.x)) * torch.sin(2 * torch.pi * (self.y))) + (3 * torch.sin(3 * torch.pi * (self.x)) * torch.sin(3 * torch.pi * (self.y))) - (4 * torch.sin(4 * torch.pi * (self.x)) * torch.sin(4 * torch.pi * (self.y))))
            
            if 'lf' in self.ques_name:
                u_moni = torch.sin(torch.pi * (self.x)) * torch.sin(torch.pi * (self.y)) + torch.sin(2*torch.pi * (self.x)) * torch.sin(2*torch.pi * (self.y))

            loss_global += torch.mean((u_moni - u) ** 2)

        else:
            self.precise_database = pd.read_csv('./Database/'+self.ques_name + '_data.csv').values
            self.x_monitor = self.precise_database[: , 0:self.coord_num].reshape([-1,self.coord_num])
            self.u_monitor = self.precise_database[: , self.coord_num : self.output_num + self.coord_num].reshape([-1,self.output_num])
            self.x_monitor = torch.tensor(self.x_monitor,requires_grad=True).float().to(device)
            self.u_monitor = torch.tensor(self.u_monitor,requires_grad=True).float().to(device)

            u = self.net(self.x_monitor).cuda()

            loss_global += torch.mean((u - self.u_monitor)**2)
            
        return loss_global, state

    def net_d(self):
        loss_d = torch.tensor(0.).to(device)
        ques_name = self.ques_name.split('_')[0]

        current_read = pd.read_csv(f'./Database/{ques_name}_inv_data_{self.data_serial[0]}.csv', header=None).values

        self.database = current_read
        for i in range(1, len(self.data_serial)):
            current_read = pd.read_csv(f'./Database/{ques_name}_inv_data_{self.data_serial[i]}.csv', header=None).values
            self.database = np.vstack([self.database,current_read])
        
        self.input_monitor = self.database[:,0:self.input_num].reshape([-1,self.input_num])
        self.u_monitor = self.database[:,self.input_num:].reshape([-1,self.output_num])
        self.input_monitor = torch.tensor(self.input_monitor,requires_grad=True).float().to(device)
        self.u_monitor = torch.tensor(self.u_monitor,requires_grad=True).float().to(device)

        if self.net.__module__.split('.')[-1] == 'PINN_post_divfree':
            output = self.net(self.input_monitor)
            output = torch.autograd.grad(output, self.input_monitor, grad_outputs=torch.ones_like(output), retain_graph=True, create_graph=True)[0]
            u = torch.cat((-output[:,1:2], output[:,0:1]), dim=1)
            loss_d += torch.mean((u - self.u_monitor)**2)
        else:
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
            u_teacher = self.net(torch.cat([self.x, self.y, self.z], dim=1)).cuda()
            u_student = self.net_student(torch.cat([self.x, self.y, self.z], dim=1)).cuda()
        else:
            u_teacher = self.net(torch.cat([self.x, self.y], dim=1)).cuda()
            u_student = self.net_student(torch.cat([self.x, self.y], dim=1)).cuda()
        
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

                if 'global' in self.ques_name:
                    self.loss_d = self.net_global()[0]
                elif 'inv' in self.ques_name:
                    if 'Poisson' in self.ques_name:
                        self.loss_d = self.net_global()[0]
                    else:
                        self.loss_d = self.net_d()
                else:
                    self.loss_d = torch.tensor(0.).to(device)

                self.loss_b = torch.tensor(0.).to(device) if self.monitor_state else self.net_b()

                self.loss_rgl = self.net_rgl(object='all', reg_type='l2') if self.regular_state else torch.tensor(0.).to(device)


                if self.monitor_state: 
                    if 'global' in self.ques_name:
                        self.loss = self.loss_d.clone() 
                    else:
                        self.loss = self.loss_d + self.loss_f
                else: 
                    self.loss = self.loss_f + self.loss_b

                if self.regular_state:
                    self.loss += self.loss_rgl
                

                self.loss.backward(retain_graph=True)

                self.optimizer.step()      

                self.scheduler.step()


                self.net.iter += 1
                self.net.iter_list.append(self.net.iter)
                self.net.loss_list.append(self.loss.item())
                self.net.loss_f_list.append(self.loss_f.item())
                self.net.loss_b_list.append(self.loss_b.item())
                self.net.loss_d_list.append(self.loss_d.item())
                self.net.loss_rgl_list.append(self.loss_rgl.item())

                if self.monitor_state:
                    self.net.para_ud_list.append(self.para_undetermin.tolist())
  
                if self.net.iter -1 in self.pace_record_skip:
                    iter_index_teacher = self.pace_record_skip.index(self.net.iter -1)
                    current_gap_teacher = self.pace_record_gap[iter_index_teacher]

                self.loss_dict = {'Iter':self.net.iter, 'Loss':self.loss.item(), 'Loss_f':self.loss_f.item(), 'Loss_b':self.loss_b.item(), 'Loss_d':self.loss_d.item(), 'Loss_rgl':self.loss_rgl.item()}

                if self.net.iter % current_gap_teacher == 0:
                    total_iter = self.step_num * self.train_steps  
                    loss_str = ', '.join([f'{key}: {int(value) if key == "Iter" else value:.5e}' for key, value in self.loss_dict.items() if key != "Iter" and value != 0])
                    iter_str = f'Iter: {{{self.net.iter}/{total_iter}}}'  
                    print(f'{iter_str}, {loss_str}')
                    if self.pace_record_state:
                        self.model_save(str(self.net.iter))

        
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if current_lr != self.original_lr:
                        print(f"Learning rate changed from {self.original_lr:.6f} to {current_lr:.6f}")
                    self.original_lr = current_lr


                    
                
                self.time_list[0] += time.time() - self.current_time
                self.current_time = time.time()

            if self.distill_state:


                for iter_inner in range(int(self.train_steps * self.train_ratio)):
                    
                    self.optimizer_student.zero_grad()

                    self.loss_teach = self.net_teach()
                    self.loss_student_rgl = self.net_rgl(mode='student', object='weight')
                    self.loss_student = self.loss_teach + self.loss_student_rgl

                    self.loss_student.backward(retain_graph=True)

                    self.optimizer_student.step()

                    self.net_student.iter += 1
                    self.net_student.iter_list.append(self.net_student.iter)
                    self.net_student.loss_list.append(self.loss_student.item())
                    self.net_student.loss_teach_list.append(self.loss_teach.item())
                    self.net_student.loss_rgl_list.append(self.loss_student_rgl.item())

                    

                    if self.net_student.iter - 1  in self.pace_record_skip:
                        iter_index_student = self.pace_record_skip.index(self.net_student.iter - 1)
                        current_gap_student = self.pace_record_gap[iter_index_student]

                    
                    if self.net_student.iter % current_gap_student == 0: 
                        total_iter_student = int(self.step_num * self.train_steps * self.train_ratio)  
                        iter_str_student = f'Iter (student): {{{self.net_student.iter}/{total_iter_student}}}'  
                        loss_str_student = f'loss_student: {self.loss_student.item():.5e}, loss_teach: {self.loss_teach.item():.5e}, loss_rgl: {self.loss_student_rgl.item():.5e}'
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

        self.control_paras = pd.read_csv(self.ini_file_path)
        self.control_paras.to_csv(f'{self.save_desti}{self.ques_name}_{self.ini_num}.csv', index=False)
        


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
                self.time_save.to_csv(self.save_desti + 'Clock time.csv', mode='a', index=False) 
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
            loss_student_data = np.array([self.net_student.iter_list, self.net_student.loss_list, self.net_student.loss_teach_list, self.net_student.loss_rgl_list])
            loss_student_data = np.transpose(loss_student_data)
            df_loss_student_data = pd.DataFrame(loss_student_data, columns=['iter', 'loss', 'loss_teach', 'loss_rgl'])
        
        if not os.path.exists(self.save_desti + '/Loss/'):       
            os.mkdir(self.save_desti + '/Loss/')

        if mode == 'teacher':
            df_loss_data.to_csv(f"{self.save_desti}/Loss/{self.ques_name}_{str(self.ini_num)}_loss_{self.net.__module__.split('.')[-1]}.csv", index=False) 

        if self.distill_state and mode == 'student':
            df_loss_student_data.to_csv(f"{self.save_desti}/Loss/{self.ques_name}_{str(self.ini_num)}_loss_{self.net_student.__module__.split('.')[-1]}_student.csv", index=False)

        if self.monitor_state:
            if mode == 'teacher':
                iter_list = np.array(self.net.iter_list).reshape([-1,1])
                para_ud = np.array(np.hstack([iter_list, self.net.para_ud_list]))
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
        u_vis.loss_vis()

        if self.distill_state:
            u_student_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti, self.net_student.__module__.split('.')[-1], input, u_student, mode='student')
            u_student_vis.figure_2d() if self.coord_num == 2 else u_student_vis.figure_3d()
            u_student_vis.loss_vis()

        if self.monitor_state:
            u_vis.para_vis()
            if self.distill_state:
                u_student_vis.para_vis()

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

            if 'PINN' in self.model_ini_dict['model'][i] and 'AsPINN' not in self.model_ini_dict['model'][i]:
                self.net = NetClass(self.layer).float().to(device)
            else:
                self.net = NetClass(self.node_num).float().to(device)

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