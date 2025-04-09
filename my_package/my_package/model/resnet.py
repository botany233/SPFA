import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from typing import List, Literal, Union

__all__ = ["Resnet18", "Resnet34", "Resnet50", "MLP"]

def Resnet18(class_num:Union[int, List[int], None] = None, pretrain = True):
    model = list(resnet18(weights = ResNet18_Weights.DEFAULT if pretrain else None).children())[:-1]
    model.append(nn.Flatten(1))
    if class_num is not None:
        if isinstance(class_num, int):
            class_num = [class_num, ]
        class_num.insert(0, 512)
        model.append(MLP(class_num))
    model = nn.Sequential(*model) # type: ignore
    return model

def Resnet34(class_num:Union[int, List[int], None] = None, pretrain = True):
    model = list(resnet34(weights = ResNet34_Weights.DEFAULT if pretrain else None).children())[:-1]
    model.append(nn.Flatten(1))
    if class_num is not None:
        if isinstance(class_num, int):
            class_num = [class_num, ]
        class_num.insert(0, 512)
        model.append(MLP(class_num))
    model = nn.Sequential(*model) # type: ignore
    return model

def Resnet50(class_num:Union[int, List[int], None] = None, pretrain = True):
    model = list(resnet50(weights = ResNet50_Weights.DEFAULT if pretrain else None).children())[:-1]
    model.append(nn.Flatten(1))
    if class_num is not None:
        if isinstance(class_num, int):
            class_num = [class_num, ]
        class_num.insert(0, 2048)
        model.append(MLP(class_num))
    model = nn.Sequential(*model) # type: ignore
    return model

def MLP(dims:List[int], sep: nn.Module = nn.ReLU()):
    assert len(dims) > 1
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(sep)
    return nn.Sequential(*layers[:-1])

if __name__ == "__main__":
    model = Resnet34(20)
    x = torch.randn((20, 3, 256, 256))
    with torch.no_grad():
        y = model(x)
    print(y.shape)