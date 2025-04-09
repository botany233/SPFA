import torch

class RealTransformer():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return torch.tensor(self.transform(x)["pixel_values"][0])