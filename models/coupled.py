import torch
import torch.nn as nn
from .KAN import KANNet

class CoupledPIKAN(nn.Module):
    def __init__(self, degree=3, num_internal_knots=5):
        super().__init__()
        # dims 参考表3：Φ1(6,64,64,1), Φ2(7,64,64,2), Φ3(6,64,64,3)
        self.phi1 = KANNet([6, 64, 64, 1], degree=degree, num_internal_knots=num_internal_knots)
        self.phi2 = KANNet([7, 64, 64, 2], degree=degree, num_internal_knots=num_internal_knots)
        self.phi3 = KANNet([6, 64, 64, 3], degree=degree, num_internal_knots=num_internal_knots)

    @torch.no_grad()
    def freeze_except(self, which: str):
        for name, p in self.named_parameters():
            p.requires_grad = (name.startswith(which))

    def forward_phi1(self, x1):
        return self.phi1(x1)  # (B,1) -> T2

    def forward_phi2(self, x2):
        return self.phi2(x2)  # (B,2) -> (T3,T4)

    def forward_phi3(self, x3):
        return self.phi3(x3)  # (B,3) -> (T5,T6,T8)
