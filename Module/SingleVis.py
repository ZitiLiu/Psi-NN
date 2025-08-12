import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
class Vis():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['xtick.labelsize'] = 10    #轴标签大小
    plt.rcParams['ytick.labelsize'] = 10    
    plt.rcParams['axes.titlesize'] = 17     #标题字体大小
    plt.rcParams['axes.labelsize'] = 16     #轴名称大小
    plt.rcParams['axes.linewidth'] = 1      #轴粗细

    def __init__(self, ques_name, ini_num, file_desti, module_name, input =[], u = [], mode: str = 'teacher'):
        input_num = input.T.shape[0]
        if input_num == 2:
            self.x, self.y = input[:,0], input[:,1]
        elif input_num == 3:
            self.x, self.y, self.z = input[:,0], input[:,1], input[:,2]
        self.u = u
        self.ques_name = ques_name
        self.ini_num = ini_num
        self.file_densti =  file_desti 
        self.module_name = module_name
        if mode == 'student':
            self.module_name += '_student'

    def loss_vis(self):
        self.loss_desti = self.file_densti + '/Loss/'
        df = pd.read_csv(f'{self.loss_desti}{self.ques_name}_{self.ini_num}_loss_{self.module_name}.csv').values
        header = pd.read_csv(f'{self.loss_desti}{self.ques_name}_{self.ini_num}_loss_{self.module_name}.csv', nrows=0).columns

        # 由于有的时候前面的iter是续算的，所以这里要重新弄一个iter列表出来
        iter = np.arange(0, len(df[:,0]), 1)

        for j in range (len(header)-1):
            # 零的值就跳过
            if df[0,j+1] == 0:
                continue
            plt.figure(figsize=(3.85, 3.5)) #调整图像大小
            # plt.plot(df[:,0],df[:,j+1])
            plt.plot(iter , df[:,j+1])
            plt.yscale('log')
            ax = plt.gca()
            ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')    # x 轴用科学记数法
            plt.grid()
            plt.xlabel(header[0])
            plt.ylabel(header[j+1])
            # plt.margins(0) #设置坐标轴紧凑
            plt.title(self.ques_name + ' ' + header[j+1] + ' ' +self.module_name, pad=10)
            plt.savefig(self.loss_desti + self.ques_name +'_'+str(self.ini_num)+'_' + header[j+1] + '_' + self.module_name + '.png', bbox_inches='tight')
            plt.close()
    
    def figure_2d(self):
        self.figure_desti = self.file_densti + '/Figure/'
        if not os.path.exists(self.figure_desti):
            os.mkdir(self.figure_desti)
        
        for i in range(len(self.u.T)):
            print(f"Drawing {self.ques_name} {self.module_name} figure {i+1}...")

            if 'Flow' in self.ques_name:
                fig, ax = plt.subplots(figsize=(4.4, 2)) # 流动算例专属长度
                x = self.x.reshape([-1,])
                y = (self.y + 0.2).reshape([-1,])  # in order to align the coordinate axis, we need to add 0.2 from the y values

                # 圆柱参数
                center_x, center_y, radius = 0.2, 0.2, 0.05

                # 三角剖分
                triang = tri.Triangulation(x, y)

                # 屏蔽圆柱区域内的三角形
                mask = []
                for tri_idx in triang.triangles:
                    # 计算三角形重心
                    xc = x[tri_idx].mean()
                    yc = y[tri_idx].mean()
                    # 判断重心是否在圆柱内
                    if ((xc - center_x)**2 + (yc - center_y)**2) < radius**2:
                        mask.append(True)
                    else:
                        mask.append(False)
                triang.set_mask(mask)

                cf = plt.tripcolor(triang, self.u[:,i], cmap='rainbow', vmin=0 if i < 2 else -0.6, vmax=4 if i == 0 else 1.3 if i == 1 else 0.6)

            else: 
                fig, ax = plt.subplots(figsize=(3.85, 3.5))        #这个就是图像的大小  
                cf = plt.scatter(self.x, self.y, c=self.u[:,i], alpha=1 - 0.1, edgecolors='none', cmap='rainbow',marker='s', s=int(8))  #s就是size也就是点的大小
            plt.xlabel('x', style='italic')
            plt.ylabel('y', style='italic')
            # plt.grid()
            plt.margins(0) #设置坐标轴紧凑
            plt.title (self.ques_name + ' ' + self.module_name, pad=10)
            fig.colorbar(cf, fraction=0.046, pad=0.04)
            plt.savefig(f"{self.figure_desti}{self.ques_name}_figure_{i+1}_{self.module_name}.png", bbox_inches='tight')
            plt.close()

    def figure_3d(self):
        self.figure_desti = self.file_densti + '/Figure/'
        if not os.path.exists(self.figure_desti):
            os.mkdir(self.figure_desti)
        
        for i in range(len(self.u.T)):
            print(f"Drawing {self.ques_name} {self.module_name} figure {i+1}...")
            fig = plt.figure(figsize=(4.4, 4))  
            cf = fig.add_subplot(111, projection='3d')
            scatter = cf.scatter(self.x, self.y, self.z, c= self.u[:,i], cmap='rainbow', edgecolors='none', vmin=self.u[:,i].min(), vmax=self.u[:,i].max())
            # cf.plot(self.x, self.y, c=self.u, alpha=1 - 0.1, edgecolors='none', cmap='rainbow',marker='s', s=int(8))  #s就是size也就是点的大小
            cf.set_xlabel('x', style='italic')
            cf.set_ylabel('y', style='italic')
            cf.set_zlabel('z', style='italic')
            cf.view_init(elev=20, azim=160)
            # cf.set_title(self.ques_name + ' ' + self.module_name)
            # fig.colorbar(cf, fraction=0.046, pad=0.04)
            colorbar = plt.colorbar(scatter, fraction=0.04, pad=0.2)  # 在3d坐标轴中添加颜色条  
            colorbar.set_label('T')  # 设置颜色条的标签  
            plt.savefig(f"{self.figure_desti}{self.ques_name}_figure_{i+1}_{self.module_name}.png", bbox_inches='tight')
            plt.close()

    def para_vis(self):
        self.para_desti = self.file_densti + '/Parameters/'
        df = pd.read_csv(self.para_desti + self.ques_name + '_' + str(self.ini_num) + '_paras_' + self.module_name + '.csv').values
        header = pd.read_csv(self.para_desti + self.ques_name + '_' + str(self.ini_num) + '_paras_' + self.module_name + '.csv',  nrows=0).columns

        iter = np.arange(0, len(df[:,0]), 1)

        for j in range (len(header)-1):
            plt.plot(iter , df[:,j+1])
            # plt.plot(df[:,0],df[:,j+1])
            # plt.yscale('log')
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
            # plt.grid()
            plt.xlabel(header[0], fontdict=font)
            plt.ylabel(header[j+1], fontdict=font)
            plt.title(self.ques_name + ' ' + header[j+1] + ' ' +self.module_name)
            plt.savefig(self.para_desti + self.ques_name +'_'+str(self.ini_num)+'_' + header[j+1] + '_' +self.module_name + '.png', bbox_inches='tight')
            plt.close()

    
    