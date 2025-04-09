import torch
import torch.nn.functional as F
from typing import Union

class Environment():
    def __init__(self,
                 initial_indexes:list[int],
                 flops_list:Union[list[float], None] = None,
                 end_choice = False,
                 max_step = -1,
                 flops_scale = 1.0):
        self.end_choice = end_choice
        self.max_step = max_step
        if flops_list is None:
            self.flops_list = None
        else:
            self.flops_list = torch.exp(-torch.tensor(flops_list) * flops_scale)
            self.flops_list /= torch.sum(self.flops_list) / len(flops_list)
        self.initial_indexes = torch.tensor(initial_indexes, dtype=torch.int64)

    def reset(self, wsi_features:torch.Tensor):
        self.infos = wsi_features
        self.num_choice = wsi_features.shape[0]
        self.current_step = 0
        self.state = self.infos[self.initial_indexes]
        return self.state

    def step(self, action:int):
        self.current_step += 1
        end = self.current_step == self.max_step
        done = self.end_choice and self.num_choice == action

        if self.flops_list is None or done:
            reward = 1
        else:
            reward = self.flops_list[action]

        if not done:
            self.state = torch.cat((self.state, self.infos[action].unsqueeze(0)), dim=0)
        return self.state, reward, done, end