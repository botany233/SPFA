import torch
import torch.nn.functional as F
from typing import Union

class Environment():
    def __init__(self,
                 initial_indexes:list[int],
                 flops_list:Union[list[float], None] = None,
                 end_choice = False,
                 max_step = -1,
                 flops_scale: float = 1.0):
        self.end_choice = end_choice
        self.max_step = max_step
        if flops_list is None:
            self.flops_list = None
        else:
            self.flops_list = torch.exp(-torch.tensor(flops_list) * flops_scale)
            self.flops_list /= torch.sum(self.flops_list) / len(flops_list)
        self.initial_indexes = torch.tensor(initial_indexes, dtype=torch.int64)

    def reset(self, wsi_features:torch.Tensor, logits:torch.Tensor):
        self.wsi_features = wsi_features
        self.logits = logits
        self.num_choice = wsi_features.shape[0]
        self.current_step = 0
        self.wsi_features_state = self.wsi_features[self.initial_indexes]
        self.logits_state = self.logits[self.initial_indexes]
        return self.wsi_features_state, self.logits_state

    def step(self, action:int):
        self.current_step += 1
        end = self.current_step == self.max_step
        done = self.end_choice and self.num_choice == action

        if self.flops_list is None or done:
            reward = 1
        else:
            reward = self.flops_list[action]

        if not done:
            self.wsi_features_state = torch.cat((self.wsi_features_state, self.wsi_features[action].unsqueeze(0)), dim=0)
            self.logits_state = torch.cat((self.logits_state, self.logits[action].unsqueeze(0)), dim=0)
        return self.wsi_features_state, self.logits_state, reward, done, end

class Environmentv2():
    def __init__(self,
                 initial_indexes:list[int],
                 flops_list:Union[list[float], None] = None,
                 end_choice = False,
                 max_step = -1,
                 flops_scale: float = 1.0):
        self.end_choice = end_choice
        self.max_step = max_step
        if flops_list is None:
            self.flops_list = None
        else:
            self.flops_list = torch.exp(-torch.tensor(flops_list) * flops_scale)
            self.flops_list /= torch.mean(self.flops_list)
        self.initial_indexes = torch.tensor(initial_indexes, dtype=torch.int64)

    def reset(self, wsi_features:torch.Tensor, logits:torch.Tensor):
        self.wsi_features = wsi_features
        self.logits = logits
        self.num_choice = wsi_features.shape[0]
        self.current_step = 0
        self.wsi_features_state = torch.zeros_like(self.wsi_features)
        self.logits_state = torch.zeros_like(self.logits)
        self.wsi_features_state[self.initial_indexes] = self.wsi_features[self.initial_indexes]
        self.logits_state[self.initial_indexes] = self.logits[self.initial_indexes]
        return self.wsi_features_state, self.logits_state

    def step(self, action:int):
        self.current_step += 1
        end = self.current_step == self.max_step
        done = self.end_choice and self.num_choice == action

        if self.flops_list is None or done:
            reward = 1
        else:
            reward = self.flops_list[action]

        if not done:
            self.wsi_features_state[action] = self.wsi_features[action]
            self.logits_state[action] = self.logits[action]
        return self.wsi_features_state, self.logits_state, reward, done, end