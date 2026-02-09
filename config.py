# config.py
import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Table 2: Model parameters
PARAMS = {
    'A1': 0.007,          # Area
    'c_p1': 3800.0,       # HTF heat capacity J/(kg C) (approximate for antifreeze)
    'rho': 1020.0,        # HTF density kg/m3
    'H': 5.88,            # Solar field thermal losses
    'L1': 1.95,           # Absorber tube length
    'L2': 9.75,           # Equivalent length
    'V': 1.5,             # Tank volume m3
    'U1': 3.6,            # Upper tank loss
    'U2': 3.8,            # Lower tank loss
    'beta': 0.11,         # Irradiance param
    # 转换因子根据需要添加
}

# Table 3: Network parameters
NET_CONFIG = {
    'phi1_in': 6,
    'phi1_out': 1, # T2
    'phi2_in': 7,
    'phi2_out': 2, # T3, T4
    'phi3_in': 6,
    'phi3_out': 3, # T5, T6, T8
    'hidden': 64,
    'grid_size': 5, # G in paper
    'spline_order': 3, # d in paper
    'lr': 1e-3,
    'epochs': 300,
    'lambda_f': 1.0, # Initial physics weight
    'lambda_y': 1.0  # Data weight
}