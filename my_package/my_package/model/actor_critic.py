import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

__all__ = ["Criticv1", "Actorv1"]

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout=0.1
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x + self.attn(self.norm(x))
        return x[0]

class Actorv1(nn.Module):
    def __init__(self, state_dim, action_dim, model_dim=512):
        super().__init__()
        self.model_dim = model_dim
        self._fc1 = nn.Sequential(nn.Linear(state_dim, model_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, model_dim))
        self.layer1 = TransLayer(dim=model_dim)
        self.layer2 = TransLayer(dim=model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self._fc2 = nn.Linear(model_dim, action_dim)

    def forward(self, h):
        h = self._fc1(h) #[n, 512]
        h = torch.cat((self.cls_token, h), dim=0)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.norm(h)[0]
        logits = self._fc2(h)
        return logits

class Criticv1(nn.Module):
    def __init__(self, state_dim, action_dim, model_dim=512):
        super().__init__()
        self._fc1 = nn.Sequential(nn.Linear(state_dim, model_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, model_dim))
        self.layer1 = TransLayer(dim=model_dim)
        self.layer2 = TransLayer(dim=model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self._fc2 = nn.Linear(model_dim, action_dim)

    def forward(self, h:torch.Tensor):
        h = self._fc1(h) #[n, 512]
        h = torch.cat((self.cls_token, h), dim=0)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.norm(h)[0]
        logits = self._fc2(h)
        return logits

if __name__ == "__main__":
    x = torch.randn((200, 11))
    model = Actorv1(11, 20)
    with torch.no_grad():
        results = model(x)
    print(results.shape)

    action = results.argmax(dim=-1).item()
    print(action)

    model = Criticv1(11, 20)
    with torch.no_grad():
        results = model(x)
    print(results.shape)