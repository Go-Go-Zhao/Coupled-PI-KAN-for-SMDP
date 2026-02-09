import torch

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))

def mse(pred, target):
    return torch.mean((pred - target) ** 2)

def mae(pred, target):
    return torch.mean(torch.abs(pred - target))

def nrmse(pred, target):
    return rmse(pred, target) / (target.max() - target.min() + 1e-8)
