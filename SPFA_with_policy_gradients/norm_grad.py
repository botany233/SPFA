import torch

def scale_grad(params, scale:float) -> None:
    for param in params:
        if param.grad is not None:
            param.grad *= scale