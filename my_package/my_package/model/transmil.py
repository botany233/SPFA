import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from torchvision import transforms

__all__ = ["TransMIL"]

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
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes, input_dim=2048, model_dim=512):
        super(TransMIL, self).__init__()
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
        h_dim = h.dim()
        if h_dim == 2: h = h.unsqueeze(0)
        h = self._fc1(h) #[B, n, 512]
        B, H, _ = h.shape
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim = 1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)
        h = self.norm(h)[:, 0]
        logits = self._fc2(h)
        if h_dim == 2: logits = logits[0]
        return logits

if __name__ == "__main__":
    from thop import profile
    model = TransMIL(6, 512)
    # x = torch.randn((100, 512))
    # with torch.no_grad():
    #     y = model(x)
    # print(y.shape)

    dummy_input = torch.randn(1000, 512)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False) # type: ignore
    print(flops/1e9, params/1e9)