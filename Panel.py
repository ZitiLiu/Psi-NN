import Module.Training as Training

import torch
torch.random.manual_seed(1234)

# 串行所有任务，注释掉如果你不需要
# Serially run all tasks, comment out if you don't need any of them.

task_1 = Training.model('Laplace', 'EXP')
task_1.train()

task_2 = Training.model('Burgers_inv', 'EXP')
task_2.train()

task_3 = Training.model('Poisson', 'EXP')
task_3.train()

task_4 = Training.model('Flow', 'EXP')
task_4.train()

task_5 = Training.model('Burgers_inv_distill', 'EXP')
task_5.train()


