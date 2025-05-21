import Module.Training_v201 as Training

import torch
torch.random.manual_seed(1234)

task_1 = Training.model('Burgers', 'EXP')
task_1.train()
