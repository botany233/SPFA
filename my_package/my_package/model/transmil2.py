import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from torchvision import transforms

__all__ = ["TransMIL2"]

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
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
        self.norm = torch.nn.BatchNorm2d(dim)

    def forward(self, x, location):
        C = x.shape[-1]
        cls_token, feat_token = x[0], x[1:]
        min_row, max_row = int(torch.min(location[:, 0])), int(torch.max(location[:, 0]))
        min_col, max_col = int(torch.min(location[:, 1])), int(torch.max(location[:, 1]))
        num_row = max_row - min_row + 1
        num_col = max_col - min_col + 1
        location[:, 0] -= min_row
        location[:, 1] -= min_col

        cnn_feat = torch.zeros(num_row, num_col, C, dtype=x.dtype, device=x.device)
        cnn_feat[location[:, 0], location[:, 1]] = feat_token
        cnn_feat = cnn_feat.permute(2, 0, 1)
        cnn_feat = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        cnn_feat = self.norm(cnn_feat.unsqueeze(0))[0]
        feat_token = cnn_feat.permute(1, 2, 0)[location[:, 0], location[:, 1]] + feat_token
        x = torch.cat((cls_token.unsqueeze(0), feat_token), dim=0)
        return x

class TransMIL2(nn.Module):
    def __init__(self, n_classes, input_dim = 2048, model_dim = 512):
        super().__init__()
        self.pos_layer = PPEG(model_dim)
        self._fc1 = nn.Sequential(nn.Linear(input_dim, model_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, model_dim))
        self.layer1 = TransLayer(dim=model_dim)
        self.layer2 = TransLayer(dim=model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self._fc2 = nn.Linear(model_dim, n_classes)

    def forward(self, x, locations):
        x = self._fc1(x) #[n, 512]
        x = torch.cat((self.cls_token, x), dim=0) #[n+1, 512]

        x = self.layer1(x) #[n+1, 512]
        x = self.pos_layer(x, locations) #[n+1, 512]
        x = self.layer2(x) #[n+1, 512]

        x = self.norm(x)[0] #[512, ]
        logits = self._fc2(x) #[n_classes, ]
        return logits

if __name__ == "__main__":
    model = TransMIL2(n_classes=10, input_dim=512)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    loss_function = nn.CrossEntropyLoss()

    x = torch.randn((500, 512))
    location = torch.randint(0, 100, (500, 2), dtype=torch.int32)
    label = torch.tensor(0)

    for i in range(5):
        score = model(x, location)
        loss = loss_function(score, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())