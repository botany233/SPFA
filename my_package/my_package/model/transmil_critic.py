import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from torchvision import transforms

__all__ = ["TransMILCritic"]

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x + self.attn(self.norm(x))
        return x[0]

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        _, C = x.shape
        cls_token, feat_token = x[0], x[1:]
        cnn_feat = feat_token.transpose(0, 1).view(C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(1).transpose(0, 1)
        x = torch.cat((cls_token.unsqueeze(0), x), dim=0)
        return x

class TransMILCritic(nn.Module):
    def __init__(self, n_classes, input_dim=2048, model_dim=512):
        super().__init__()
        self.model_dim = model_dim
        self.pos_layer = PPEG(dim=model_dim)
        self._fc1 = nn.Sequential(nn.Linear(input_dim, model_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=model_dim)
        self.layer2 = TransLayer(dim=model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self._fc2 = nn.Linear(model_dim, self.n_classes)

    def forward(self, h):
        h = self._fc1(h) #[n, 512]
        H, _ = h.shape
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:add_length, :]], dim = 1)
        h = torch.cat((self.cls_token, h), dim=0)
        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)
        h = self.norm(h)[:, 0]
        logits = self._fc2(h)
        return logits

if __name__ == "__main__":
    x = torch.randn((4, 200, 2048))
    model = TransMIL(n_classes=10)
    with torch.no_grad():
        results = model(x)
    print(results.shape)