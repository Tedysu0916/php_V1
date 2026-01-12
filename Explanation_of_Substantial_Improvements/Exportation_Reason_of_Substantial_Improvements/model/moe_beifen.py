import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], input_gate: nn.Module, task_gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.input_gate = input_gate
        self.task_gate = task_gate
        self.args = moe_args
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs: torch.Tensor, task_param) -> torch.Tensor:
        input_gate_logits = self.input_gate(inputs)
        task_gate_logits = self.task_gate(task_param)

        gate_logits = (1 - self.alpha) * input_gate_logits + self.alpha * task_gate_logits

        # gate_logits = input_gate_logits

        weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok
        )

        # calculate aux_loss
        weights_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(inputs.dtype)
        average_weight = torch.mean(weights_softmax, dim=[0, 1])

        # use top 2 to cal
        indices_top2 = F.one_hot(selected_experts, num_classes=self.args.num_experts).sum(dim=2)
        average_count = torch.mean(indices_top2.float(), dim=[0, 1]).to(inputs.dtype)

        # cal aux loss, Load-Balancing Loss
        l_aux = torch.mean(average_weight * average_count) * self.args.num_experts

        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)

        results = torch.zeros_like(inputs)

        for i, expert in enumerate(self.experts):
            idx_1, idx_2, nth_expert = torch.where(selected_experts == i)
            results[idx_1, idx_2] += weights[idx_1, idx_2, nth_expert, None] * expert(inputs[idx_1, idx_2])

        return results, l_aux.float()

# class MoeLayer(nn.Module):
#     def __init__(self, experts: List[nn.Module], input_gate: nn.Module, task_gate: nn.Module, moe_args: MoeArgs,d_model: int):
#         super().__init__()
#         assert len(experts) > 0
#         self.experts = nn.ModuleList(experts)
#         self.input_gate = input_gate
#         self.task_gate = task_gate
#         self.args = moe_args
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#
#     def forward(self, inputs: torch.Tensor, task_param) -> torch.Tensor:
#         input_gate_logits = self.input_gate(inputs)
#         task_gate_logits = self.task_gate(task_param)
#
#         gate_logits = (1 - self.alpha) * input_gate_logits + self.alpha * task_gate_logits
#
#         # gate_logits = input_gate_logits
#
#         weights, selected_experts = torch.topk(
#             gate_logits, self.args.num_experts_per_tok
#         )
#
#         # calculate aux_loss
#         weights_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(inputs.dtype)
#         average_weight = torch.mean(weights_softmax, dim=[0, 1])
#
#         # use top 2 to cal
#         indices_top2 = F.one_hot(selected_experts, num_classes=self.args.num_experts).sum(dim=2)
#         average_count = torch.mean(indices_top2.float(), dim=[0, 1]).to(inputs.dtype)
#
#         # cal aux loss, Load-Balancing Loss
#         l_aux = torch.mean(average_weight * average_count) * self.args.num_experts
#
#         weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)
#
#         results = torch.zeros_like(inputs)
#         for i, expert in enumerate(self.experts):
#             idx_1, idx_2, nth_expert = torch.where(selected_experts == i)
#             results[idx_1, idx_2] += weights[idx_1, idx_2, nth_expert, None] * expert(inputs[idx_1, idx_2])
#
#         return results, l_aux.float()
# class MoeLayer(nn.Module):
#     def __init__(self, experts: List[nn.Module], input_gate: nn.Module, task_gate: nn.Module, moe_args: MoeArgs):
#         super().__init__()
#         assert len(experts) > 0
#         self.experts = nn.ModuleList(experts)
#         self.input_gate = input_gate
#         # self.d_model = d_model
#         self.task_gate = task_gate
#         self.args = moe_args
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#         self.expert_biases = nn.Parameter(torch.zeros(self.args.num_experts))
#
#     def forward(self, inputs: torch.Tensor, task_param) -> torch.Tensor:
#         input_gate_logits = self.input_gate(inputs)
#         task_gate_logits = self.task_gate(task_param)
#         input_gate_logits = (1 - self.alpha) * input_gate_logits + self.alpha * task_gate_logits
#         gate_probs = torch.sigmoid(input_gate_logits)
#         #
#         gate_logits = input_gate_logits + self.expert_biases
#         _, top_k_indices = torch.topk(gate_logits, self.args.num_experts_per_tok, dim=-1)
#         # weights, selected_experts = torch.topk(
#         #     gate_logits, self.args.num_experts_per_tok
#         # )
#         top_k_probs = gate_probs.gather(-1, top_k_indices)
#
#         # normalize to sum to 1
#         top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
#
#         # get the routed expert outputs
#         batch_size, seq_len, _ = inputs.shape
#         expert_outputs = torch.stack([expert(inputs) for expert in self.experts])
#         indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
#         expert_outputs = expert_outputs.gather(0, indices.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
#
#         final_output = (expert_outputs * top_k_probs.unsqueeze(-1)).sum(dim=-2)
#         f_i = gate_probs.sum(dim=[0, 1]) / (gate_probs.size(0) * gate_probs.size(1))
#         P_i = gate_probs.mean(dim=[0, 1])
#         l_aux = torch.sum(f_i * P_i)
#
#         return final_output, l_aux.float()
        # # calculate aux_loss
        # weights_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(inputs.dtype)
        # average_weight = torch.mean(weights_softmax, dim=[0,1])
        #
        # # use top 2 to cal
        # indices_top2  = F.one_hot(selected_experts, num_classes=self.args.num_experts).sum(dim=2)
        # average_count = torch.mean(indices_top2.float(), dim=[0,1]).to(inputs.dtype)
        #
        # # cal aux loss, Load-Balancing Loss
        # l_aux = torch.mean(average_weight * average_count) * self.args.num_experts
        #
        # weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)
        #
        #
        # results = torch.zeros_like(inputs)
        # for i, expert in enumerate(self.experts):
        #     idx_1, idx_2, nth_expert = torch.where(selected_experts == i)
        #     results[idx_1, idx_2] += weights[idx_1, idx_2, nth_expert, None] * expert(inputs[idx_1,idx_2])
        #
        # return results, l_aux.float()
