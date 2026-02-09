# main.py
import torch
from torch.utils.data import DataLoader, Dataset
from models import CoupledPIKAN
from Train import train_coupled_model
from config import DEVICE, NET_CONFIG


# Dummy Dataset class since we don't have the csv
class SMDPDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        # Random synthetic data mimicking the ranges in paper
        self.t = torch.linspace(0, 1, size).unsqueeze(1)
        self.X1 = torch.randn(size, NET_CONFIG['phi1_in'])  # [t, T2_prev, F1, I, Ta, T4_prev]
        self.X2_base = torch.randn(size, NET_CONFIG['phi2_in'] - 1)  # Excluding T2_pred
        self.T6_prev = torch.randn(size, 1)

        # Targets
        self.T2 = torch.randn(size, 1)
        self.T3 = torch.randn(size, 1)
        self.T4 = torch.randn(size, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            't': self.t[idx],
            'X1': self.X1[idx],
            'X2_base': self.X2_base[idx],
            'T6_prev': self.T6_prev[idx],
            'T2': self.T2[idx],
            'T3': self.T3[idx],
            'T4': self.T4[idx]
        }


def main():
    print(f"Initializing Coupled PI-KAN on {DEVICE}...")

    # 1. Initialize Model
    model = CoupledPIKAN().to(DEVICE)

    # 2. Load Data
    dataset = SMDPDataset(size=360)  # 360 mins as per paper
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. Train
    train_coupled_model(model, dataloader)

    # 4. Save
    torch.save(model.state_dict(), "coupled_pi_kan.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()