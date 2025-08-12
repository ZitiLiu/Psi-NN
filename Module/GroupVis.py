import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Vis():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['xtick.labelsize'] = 10    #轴标签大小
    plt.rcParams['ytick.labelsize'] = 10    
    plt.rcParams['axes.titlesize'] = 17     #标题字体大小
    plt.rcParams['axes.labelsize'] = 16     #轴名称大小
    plt.rcParams['axes.linewidth'] = 1      #轴粗细
    def __init__(self, ques_name, ini_num, file_desti, x=[], y=[],u=[]):
        self.x = x
        self.y = y 
        self.u = u
        self.ques_name = ques_name
        self.ini_num = ini_num
        self.file_desti =  file_desti 
        self.module_num = 0
        self.group_loss = []
        self.group_name = []
        self.group_para = []

        
        new_model_name = '$\Psi$-NN'
        # 自动获取当前文件夹下所有.py文件名（不含扩展名）
        module_dir = os.path.dirname(os.path.abspath(__file__))
        py_files = [f[:-3] for f in os.listdir(module_dir) if f.endswith('.py') and f != os.path.basename(__file__)]


        # 自动映射
        self.label_name = {}
        mapping = {
                    'PINN': 'PINN',
                    'AblationNN_1': 'Ablation_1'
                }
        for k in py_files:
            if 'PINN_post' in k:
                self.label_name[k] = 'PINN post'
            elif 'PsiNN' in k:
                self.label_name[k] = new_model_name
            else:
                self.label_name[k] = mapping.get(k, k)

        self.colors = {
            'PINN':'#00CC00',
            'PINN post':'blue',
            new_model_name:'red',
            'ablation_1':"#9900ff"
        }


    def loss_read(self, module_name):
        # 几个模型的迭代步数不一定需要一样长，只需要loss的顺序相同就可以了
        self.loss_desti = self.file_desti + '/Loss/'
        self.module_name = module_name
        self.group_loss.append([self.group_loss,pd.read_csv(self.loss_desti + self.ques_name + '_' + str(self.ini_num) + '_loss_' + self.module_name + '.csv').values])
        self.loss_header = pd.read_csv(self.loss_desti + self.ques_name + '_'+str(self.ini_num)+'_loss_' + self.module_name + '.csv',  nrows=0).columns
        self.loss_num =  len(self.loss_header)
        self.group_name.append(self.module_name)
        # print(self.group_name)
        self.module_num+=1

    
    def loss_vis(self):
        for j in range (len(self.loss_header)-1):
            plt.figure(figsize=(6.6,6)) #调整图像大小
            for i in range (self.module_num):


                # print(self.group_name[i])
                # print(self.label_name)

                plt.plot(self.group_loss[i][1][:,0], self.group_loss[i][1][:,j+1], label=self.group_name[i], color = self.colors[self.label_name[self.group_name[i]]], alpha=0.8)
                font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}


                plt.grid()
                plt.legend()
                plt.yscale('log')
                plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
                plt.xlabel(self.loss_header[0], fontdict=font)
                plt.ylabel(self.loss_header[j+1], fontdict=font)
            plt.title(self.ques_name + ' ' + self.loss_header[j+1] + ' ' + 'Comparison', pad=8)
            plt.savefig(self.loss_desti + self.ques_name + '_' +str(self.ini_num) +'_'+ self.loss_header[j+1] + '_' + 'comparison' + '.png', bbox_inches='tight')
            plt.close()     

    # 进行要训练的参数的读取
    def para_read(self, module_name):
        self.para_desti = self.file_desti + '/Parameters/'
        self.module_name = module_name
        self.group_para.append([self.group_para, pd.read_csv(self.para_desti + self.ques_name + '_' + str(self.ini_num) + '_paras_' + self.module_name + '.csv').values])
        self.para_header = pd.read_csv(self.para_desti + self.ques_name + '_'+str(self.ini_num)+'_paras_' + self.module_name + '.csv',  nrows=0).columns
        self.para_num =  len(self.para_header)
        # 如果要读取参数，那么一定一起读取了损失函数，所以这里不用重复append，所以一定要先读取损失函数哟
        # self.group_name.append(self.module_name)
        # self.module_num+=1
    
    def para_vis(self):
        for j in range (len(self.para_header)-1):
            plt.figure(figsize=(4.4,4)) #调整图像大小
            for i in range (self.module_num):
                plt.plot(self.group_para[i][1][:,0], self.group_para[i][1][:,j+1], label=self.group_name[i], color = self.colors[self.label_name[self.group_name[i]]])
                # plt.yscale('log')
                font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
                plt.grid()
                plt.legend()
                # plt.yscale('log')
                plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
                plt.xlabel(self.para_header[0], fontdict=font) 
                plt.ylabel(self.para_header[j+1], fontdict=font)
            plt.title(self.ques_name + ' ' + self.para_header[j+1] + ' ' + 'Comparison', pad=8)
            plt.savefig(self.para_desti + self.ques_name + '_' +str(self.ini_num) +'_'+ self.para_header[j+1] + '_' + 'comparison' + '.png', bbox_inches='tight')
            plt.close()    