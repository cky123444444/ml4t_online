import torch
import torch.nn as nn
import torch.nn.functional as F
from common_models import *
from model_utils import *
#使用小时级别数据，样本总数200w，模型参数量400，预测目标为(1小时，4小时)的收益率和波动率
# class DragonNetBase(nn.Module):
#     def __init__(self, input_dim=111):
#         super().__init__()
#         self.bottom_tower = DNN(name='bottom_tower', input_dim=input_dim, nn_dims=[256, 128, 64],last_layer_use_activation=True)

#         self.ctrl_return_1h_tower = DNN(name='ctrl_return_1h_tower', input_dim=64, nn_dims=[32, 16, 1],last_layer_use_activation=False)

#         self.ctrl_sigma_1h_tower = DNN(name='ctrl_sigma_1h_tower', input_dim=64, nn_dims=[32, 16, 1], last_layer_use_activation=False)

#         self.trmt_return_1h_tower = DNN(name='trmt_return_1h_tower', input_dim=64, nn_dims=[32, 16, 1],last_layer_use_activation=False)

#         self.trmt_sigma_1h_tower = DNN(name='trmt_sigma_1h_tower', input_dim=64, nn_dims=[32, 16, 1],last_layer_use_activation=False)
        
#         self.t_tower = DNN(name='t_tower', input_dim=64, nn_dims=[32, 16, 1], last_layer_use_activation=False)
#         # self.bottom_tower = DNN(name='bottom_tower', input_dim=111, hidden_dim1=256, hidden_dim2=128, output_dim=64, last_layer_activation=True)
#         # self.ctrl_return_1h_tower = DNN(name='ctrl_return_1h_tower', input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=1, last_layer_activation=False)
#         # self.ctrl_sigma_1h_tower = DNN(name='ctrl_sigma_1h_tower', input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=1, last_layer_activation=False)
#         # self.trmt_return_1h_tower = DNN(name='trmt_return_1h_tower', input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=1, last_layer_activation=False)
#         # self.trmt_sigma_1h_tower = DNN(name='trmt_sigma_1h_tower', input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=1, last_layer_activation=False)
#         # self.t_tower = DNN(name='t_tower', input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=1, last_layer_activation=False)
#         self.return_1h_epsilon = EpsilonLayer(name='return_1h_eps')
#         self.sigma_1h_epsilon = EpsilonLayer(name='sigma_1h_eps')

#     def forward(self, inputs):
#         base_out = self.bottom_tower(inputs)
#         ctrl_return_1h_out = self.ctrl_return_1h_tower(base_out)
#         ctrl_sigma_1h_out = self.ctrl_sigma_1h_tower(base_out)
#         trmt_return_1h_out = self.trmt_return_1h_tower(base_out)
#         trmt_sigma_1h_out = self.trmt_sigma_1h_tower(base_out)
#         t_out = self.t_tower(base_out)
#         return (ctrl_return_1h_out, 
#                 trmt_return_1h_out, 
#                 ctrl_sigma_1h_out, 
#                 trmt_sigma_1h_out, 
#                 t_out,
#                 self.return_1h_epsilon,
#                 self.sigma_1h_epsilon)
    
class DragonNetLarge(nn.Module):
    def __init__(self, input_dim=111):
        super().__init__()
        self.bottom_tower = DNN(name='bottom_tower', input_dim=input_dim, nn_dims=[128, 256, 512, 256, 128],last_layer_use_activation=True)

        self.ctrl_return_1h_tower = DNN(name='ctrl_return_1h_tower', input_dim=128, nn_dims=[64, 32, 16, 1],last_layer_use_activation=False)

        self.ctrl_sigma_1h_tower = DNN(name='ctrl_sigma_1h_tower', input_dim=128, nn_dims=[64, 32, 16, 1], last_layer_use_activation=False)

        self.trmt_return_1h_tower = DNN(name='trmt_return_1h_tower', input_dim=128, nn_dims=[64, 32, 16, 1],last_layer_use_activation=False)

        self.trmt_sigma_1h_tower = DNN(name='trmt_sigma_1h_tower', input_dim=128, nn_dims=[64, 32, 16, 1],last_layer_use_activation=False)
        
        self.t_tower = DNN(name='t_tower', input_dim=128, nn_dims=[64, 32, 16, 1], last_layer_use_activation=False)

        # self.return_1h_epsilon = EpsilonLayer(name='return_1h_eps')
        # self.sigma_1h_epsilon = EpsilonLayer(name='sigma_1h_eps')

    def forward(self, inputs):
        base_out = self.bottom_tower(inputs)
        ctrl_return_1h_out = self.ctrl_return_1h_tower(base_out)
        ctrl_sigma_1h_out = self.ctrl_sigma_1h_tower(base_out)
        trmt_return_1h_out = self.trmt_return_1h_tower(base_out)
        trmt_sigma_1h_out = self.trmt_sigma_1h_tower(base_out)
        t_out = self.t_tower(base_out)
        return (ctrl_return_1h_out, 
                trmt_return_1h_out, 
                ctrl_sigma_1h_out, 
                trmt_sigma_1h_out, 
                t_out)
                # self.return_1h_epsilon,
                # self.sigma_1h_epsilon)
    
    

# class DragonNetTriple(nn.Module):
#     def __init__(self, input_dim=111):
#         super().__init__()
#         self.bottom_tower = DNN(name='bottom_tower', input_dim=input_dim, nn_dims=[128, 256, 512, 256, 128],last_layer_use_activation=True)

#         self.ctrl_return_1h_tower = DNN(name='ctrl_return_1h_tower', input_dim=128, nn_dims=[64, 32, 16, 1],last_layer_use_activation=False)

#         self.ctrl_sigma_1h_tower = DNN(name='ctrl_sigma_1h_tower', input_dim=128, nn_dims=[64, 32, 16, 1], last_layer_use_activation=False)

#         self.trmt_return_1h_tower = DNN(name='trmt_return_1h_tower', input_dim=128, nn_dims=[64, 32, 16, 1],last_layer_use_activation=False)

#         self.trmt_sigma_1h_tower = DNN(name='trmt_sigma_1h_tower', input_dim=128, nn_dims=[64, 32, 16, 1],last_layer_use_activation=False)
        
#         self.t_tower = DNN(name='t_tower', input_dim=128, nn_dims=[64, 32, 16, 3], last_layer_use_activation=False)


#     def forward(self, inputs):
#         base_out = self.bottom_tower(inputs)
#         ctrl_return_1h_out = self.ctrl_return_1h_tower(base_out)
#         ctrl_sigma_1h_out = self.ctrl_sigma_1h_tower(base_out)
#         trmt_return_1h_out = self.trmt_return_1h_tower(base_out)
#         trmt_sigma_1h_out = self.trmt_sigma_1h_tower(base_out)
#         t_out = self.t_tower(base_out)
#         return (ctrl_return_1h_out, 
#                 trmt_return_1h_out, 
#                 ctrl_sigma_1h_out, 
#                 trmt_sigma_1h_out, 
#                 t_out)
        

