import torch
import torch.nn as nn
from KAN import KAN
from config import NET_CONFIG


class CoupledPIKAN(nn.Module):
    def __init__(self):
        super(CoupledPIKAN, self).__init__()
        c = NET_CONFIG

        # Sub-network 1: Predict T2
        # Input: [t, T2_prev, F1, I, Ta, T4_prev] (Example dim=6)
        self.phi1 = KAN([c['phi1_in'], c['hidden'], c['hidden'], c['phi1_out']],
                        grid_size=c['grid_size'], spline_order=c['spline_order'])

        # Sub-network 2: Predict T3, T4
        # Input: [t, T3_prev, T4_prev, V1, I, Ta, T2_pred] (Example dim=7)
        self.phi2 = KAN([c['phi2_in'], c['hidden'], c['hidden'], c['phi2_out']],
                        grid_size=c['grid_size'], spline_order=c['spline_order'])

        # Sub-network 3: Predict T5, T6, T8
        # Input: [t, T6_prev, T8_prev, V1, I, Ta] (Example dim=6)
        self.phi3 = KAN([c['phi3_in'], c['hidden'], c['hidden'], c['phi3_out']],
                        grid_size=c['grid_size'], spline_order=c['spline_order'])

    def forward_phi1(self, x):
        return self.phi1(x)

    def forward_phi2(self, x):
        return self.phi2(x)

    def forward_phi3(self, x):
        return self.phi3(x)
