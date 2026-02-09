import torch

class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x):
        self.min = x.min(dim=0)[0]
        self.max = x.max(dim=0)[0]

    def transform(self, x):
        return 2 * (x - self.min) / (self.max - self.min + 1e-8) - 1

    def inverse(self, x):
        return (x + 1) / 2 * (self.max - self.min) + self.min
