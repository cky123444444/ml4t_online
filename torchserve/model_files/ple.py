import torch
import torch.nn as nn
import torch.nn.functional as F
from common_models import *
from model_utils import *


class PLENet(nn.Module):
    """
    Progressive Layered Extraction (PLE) 模型
    多层提取网络，每层包含 shared experts + ctrl experts + trmt experts + 门控网络
    t_network 拆分为 t_ctrl 和 t_trmt，分别预测下跌概率和上涨概率
    """

    def __init__(self, input_dim=111, expert_out_dim=128, num_shared_experts=2, num_task_experts=2, num_levels=2, expert_hidden_dim=256):
        super().__init__()
        self.num_levels = num_levels
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts

        # === 逐层构建 extraction network ===
        self.shared_experts = nn.ModuleList()
        self.ctrl_experts = nn.ModuleList()
        self.trmt_experts = nn.ModuleList()
        self.ctrl_gates = nn.ModuleList()
        self.trmt_gates = nn.ModuleList()
        self.shared_gates = nn.ModuleList()

        for level in range(num_levels):
            level_input_dim = input_dim if level == 0 else expert_out_dim

            # Shared experts
            self.shared_experts.append(nn.ModuleList([
                DNN(name=f'shared_expert_l{level}_{i}',
                    input_dim=level_input_dim,
                    nn_dims=[expert_hidden_dim, expert_out_dim],
                    last_layer_use_activation=True)
                for i in range(num_shared_experts)
            ]))

            # Ctrl (下跌) experts
            self.ctrl_experts.append(nn.ModuleList([
                DNN(name=f'ctrl_expert_l{level}_{i}',
                    input_dim=level_input_dim,
                    nn_dims=[expert_hidden_dim, expert_out_dim],
                    last_layer_use_activation=True)
                for i in range(num_task_experts)
            ]))

            # Trmt (上涨) experts
            self.trmt_experts.append(nn.ModuleList([
                DNN(name=f'trmt_expert_l{level}_{i}',
                    input_dim=level_input_dim,
                    nn_dims=[expert_hidden_dim, expert_out_dim],
                    last_layer_use_activation=True)
                for i in range(num_task_experts)
            ]))

            # Gate: ctrl sees ctrl_experts + shared_experts
            self.ctrl_gates.append(
                GateNetwork(name=f'ctrl_gate_l{level}',
                            input_dim=level_input_dim,
                            num_experts=num_task_experts + num_shared_experts))

            # Gate: trmt sees trmt_experts + shared_experts
            self.trmt_gates.append(
                GateNetwork(name=f'trmt_gate_l{level}',
                            input_dim=level_input_dim,
                            num_experts=num_task_experts + num_shared_experts))

            # Gate: shared sees all experts
            self.shared_gates.append(
                GateNetwork(name=f'shared_gate_l{level}',
                            input_dim=level_input_dim,
                            num_experts=num_shared_experts + num_task_experts * 2))

        # === Task towers (维度与 DragonNetLarge 一致) ===
        self.ctrl_return_1h_tower = DNN(name='ctrl_return_1h_tower', input_dim=expert_out_dim, nn_dims=[64, 32, 16, 1], last_layer_use_activation=False)

        self.ctrl_sigma_1h_tower = DNN(name='ctrl_sigma_1h_tower', input_dim=expert_out_dim, nn_dims=[64, 32, 16, 1], last_layer_use_activation=False)

        self.trmt_return_1h_tower = DNN(name='trmt_return_1h_tower', input_dim=expert_out_dim, nn_dims=[64, 32, 16, 1], last_layer_use_activation=False)

        self.trmt_sigma_1h_tower = DNN(name='trmt_sigma_1h_tower', input_dim=expert_out_dim, nn_dims=[64, 32, 16, 1], last_layer_use_activation=False)

        # t_network 拆分为两个，分别放入ctrl/trmt tower
        self.t_ctrl_tower = DNN(name='t_ctrl_tower', input_dim=expert_out_dim, nn_dims=[64, 32, 16, 1], last_layer_use_activation=False)
        self.t_trmt_tower = DNN(name='t_trmt_tower', input_dim=expert_out_dim, nn_dims=[64, 32, 16, 1], last_layer_use_activation=False)

    def forward(self, inputs):
        ctrl_input = inputs
        trmt_input = inputs
        shared_input = inputs

        for level in range(self.num_levels):
            shared_outs = [expert(shared_input) for expert in self.shared_experts[level]]
            ctrl_outs = [expert(ctrl_input) for expert in self.ctrl_experts[level]]
            trmt_outs = [expert(trmt_input) for expert in self.trmt_experts[level]]

            ctrl_input = self.ctrl_gates[level](ctrl_input, ctrl_outs + shared_outs)
            trmt_input = self.trmt_gates[level](trmt_input, trmt_outs + shared_outs)
            shared_input = self.shared_gates[level](shared_input, shared_outs + ctrl_outs + trmt_outs)

        # Return/Sigma 走各自专属路径
        ctrl_return_1h_out = self.ctrl_return_1h_tower(ctrl_input)
        ctrl_sigma_1h_out = self.ctrl_sigma_1h_tower(ctrl_input)
        trmt_return_1h_out = self.trmt_return_1h_tower(trmt_input)
        trmt_sigma_1h_out = self.trmt_sigma_1h_tower(trmt_input)
        # t_ctrl 从 ctrl 表征预测下跌概率，t_trmt 从 trmt 表征预测上涨概率
        t_ctrl_out = self.t_ctrl_tower(ctrl_input)
        t_trmt_out = self.t_trmt_tower(trmt_input)

        return (ctrl_return_1h_out,
                trmt_return_1h_out,
                ctrl_sigma_1h_out,
                trmt_sigma_1h_out,
                t_ctrl_out,
                t_trmt_out)
