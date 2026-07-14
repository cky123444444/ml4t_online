import torch
import torch.nn as nn
import torch.nn.functional as F

# class DNN(nn.Module):
#     def __init__(self, name, nn_dims,
#                  initializer=None,
#                  last_layer_use_activation=False,
#                  activation=nn.ReLU()):
#         super(DNN, self).__init__()
#         self.name = name

#         # 如果没传初始化器，默认使用 xavier_uniform
#         # if initializer is None:
#         #     def initializer(weight):
#         #         if weight is not None:
#         #             nn.init.xavier_uniform_(weight)

#         layers = []
#         for i in range(len(nn_dims)):
#             linear = nn.LazyLinear(nn_dims[i])

#             # 保存初始化方法，等 LazyLinear 初始化完成后再初始化
#             def apply_init(m):
#                 if isinstance(m, nn.LazyLinear):
#                     # 确保参数已初始化才可访问
#                     if hasattr(m, 'weight') and m.weight is not None:
#                         initializer(m.weight)
#                     if hasattr(m, 'bias') and m.bias is not None:
#                         nn.init.constant_(m.bias, 0.0)

#             layers.append(linear)

#             if i != len(nn_dims) - 1 or last_layer_use_activation:
#                 layers.append(activation)

#         self.net = nn.Sequential(*layers)
#         #self.initializer = apply_init  # 保存以备后续调用

#     def get_input(self, inputs, inputs_dict):
#         input_keys = inputs.get('dnn_input', [])
#         input_tensors = [inputs_dict[k] for k in input_keys]
#         return torch.cat(input_tensors, dim=1)

#     def forward(self, inputs):
#         # LazyLinear 初始化发生在第一次 forward 时，因此先 forward 再初始化
#         output = self.net(inputs)

#         # 确保初始化只应用一次
#         # if hasattr(self, 'initializer'):
#         #     self.net.apply(self.initializer)
#         #     del self.initializer  # 删除，防止后续重复初始化

#         return output

# class DNN(nn.Module):
#     def __init__(self, name, input_dim, hidden_dim1, hidden_dim2, output_dim, last_layer_activation=False):
#         super(DNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim1)   # 第一层
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2) # 第二层
#         self.fc3 = nn.Linear(hidden_dim2, output_dim)  # 第三层（输出层）
#         self.name = name
#         self.last_layer_activation = last_layer_activation
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.zeros_(self.fc1.bias)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.zeros_(self.fc2.bias)
#         nn.init.xavier_uniform_(self.fc3.weight)
#         nn.init.zeros_(self.fc3.bias)
#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # 第一层 + ReLU
#         x = F.relu(self.fc2(x))  # 第二层 + ReLU
#         x = self.fc3(x)          # 输出层（无激活或按需要加 softmax/sigmoid）
#         if self.last_layer_activation:
#             x = F.relu(x)
#         return x

class DNN(nn.Module):
    def __init__(self, name, input_dim, nn_dims, last_layer_use_activation=False):
        super(DNN, self).__init__()
        self.name = name
        self.last_layer_use_activation = last_layer_use_activation

        # 构建所有线性层
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in nn_dims:
            layer = nn.Linear(prev_dim, dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            prev_dim = dim

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 最后一层是否加激活由参数控制
            if i < len(self.layers) - 1:
                x = F.relu(x)
            elif self.last_layer_use_activation:
                x = F.relu(x)
        return x
    
class EpsilonLayer(nn.Module):
    """
    A learnable scalar multiplier layer (PyTorch version).
    
    Multiplies input tensor by a trainable epsilon parameter.
    """

    def __init__(self, name, initializer=None):
        super(EpsilonLayer, self).__init__()
        self.name = 'epsilon_layer_' + name
        self.initializer = initializer or torch.nn.init.normal_
        self.epsilon = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.initializer(self.epsilon)

    def build(self, input_shape):
        """
        Lazily initialize epsilon parameter based on input shape.
        Should be called manually before training if needed.
        """
        device = next(self.parameters(), torch.tensor(0)).device
        self.epsilon = nn.Parameter(torch.empty(1, 1, device=device))
        

    def forward(self, inputs):

        output = inputs * self.epsilon

        return output
    
class GateNetwork(nn.Module):
    """PLE门控网络，将多个expert的输出按学习到的权重加权融合"""

    def __init__(self, name, input_dim, num_experts):
        super(GateNetwork, self).__init__()
        self.name = name
        self.gate = nn.Linear(input_dim, num_experts)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, gate_input, expert_outputs):
        """
        Args:
            gate_input: (batch, input_dim) 门控输入，用于计算expert权重
            expert_outputs: list of (batch, expert_dim) tensors
        Returns:
            (batch, expert_dim) 加权融合后的输出
        """
        gate_weights = F.softmax(self.gate(gate_input), dim=1)  # (batch, num_experts)
        expert_stack = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, expert_dim)
        output = torch.bmm(gate_weights.unsqueeze(1), expert_stack).squeeze(1)  # (batch, expert_dim)
        return output