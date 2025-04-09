import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LinearPool", "MeanPool", "Voting"]

class LinearPool(nn.Module):
    def __init__(self,
                 n_model:int,
                 n_class:int,
                 model_dim=512):
        super().__init__()
        self.linear_1 = nn.Linear(n_model, 1)
        self.norm = nn.LayerNorm(model_dim)
        self.linear_2 = nn.Linear(model_dim, n_class)

    def forward(self, x):
        x = x.transpose(0, 1) #[D, N]
        x = self.linear_1(x)[:, 0] #[D]
        x = self.norm(x)
        x = self.linear_2(x) #[class]
        return x

class MeanPool(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.mean(x, dim=0)
        x = self.linear(x)
        return x

class Voting(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        vote = torch.argmax(x, dim=-1)
        vote_counts = torch.bincount(vote, minlength=x.shape[1])
        result = torch.argmax(vote_counts)
        output = torch.zeros_like(vote_counts)
        output[result] = 1
        return output

if __name__ == "__main__":
    # model = LinearPool(10, 5)
    # x = torch.randn((10, 512))
    # with torch.no_grad():
    #     y = model(x)
    # print(y.shape)

    # model = MeanPool(512, 5)
    # x = torch.randn((15, 512))
    # with torch.no_grad():
    #     y = model(x)
    # print(y.shape)

    model = Voting()
    x = torch.randn((15, 6))
    with torch.no_grad():
        y = model(x)
    print(y.shape)
    print(y)