import torch

class RealTransformer():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x, return_tensors="pt")['pixel_values'][0]