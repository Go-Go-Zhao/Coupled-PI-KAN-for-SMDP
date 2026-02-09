import torch
from torch.utils.data import Dataset

class SMDPDataset(Dataset):
    """
    raw_data: Tensor [N, 11]
    columns = [t,T2,T3,T4,T5,T6,T8,I,Ta,F1,V1]
    """
    def __init__(self, raw_data: torch.Tensor):
        assert raw_data.shape[1] == 11
        self.data = raw_data

    def __len__(self):
        return self.data.shape[0] - 1  # need t-1

    def __getitem__(self, idx):
        d0 = self.data[idx]       # t-1
        d1 = self.data[idx + 1]   # t

        # ---------- unpack ----------
        t  = d1[0:1]

        T2_0, T3_0, T4_0 = d0[1], d0[2], d0[3]
        T5_0, T6_0       = d0[4], d0[5]

        T2_1, T3_1, T4_1 = d1[1], d1[2], d1[3]
        T5_1, T6_1, T8_1 = d1[4], d1[5], d1[6]

        I, Ta = d1[7], d1[8]
        F1, V1 = d1[9], d1[10]

        # ---------- X1 ----------
        X1 = torch.tensor([
            t.item(),
            T2_0.item(),
            I.item(),
            Ta.item(),
            F1.item(),
            T3_0.item()
        ])

        yT2 = torch.tensor([T2_1.item()])

        # ---------- X2 ----------
        X2 = torch.tensor([
            t.item(),
            T3_0.item(),
            T4_0.item(),
            T2_1.item(),   # Φ1 output (teacher forcing)
            T6_0.item(),
            V1.item(),
            Ta.item()
        ])

        yT3T4 = torch.tensor([T3_1.item(), T4_1.item()])

        # ---------- X3 ----------
        X3 = torch.tensor([
            t.item(),
            T5_0.item(),
            T6_0.item(),
            T4_1.item(),   # Φ2 output (teacher forcing)
            F1.item(),
            Ta.item()
        ])

        yT5T6T8 = torch.tensor([T5_1.item(), T6_1.item(), T8_1.item()])

        return {
            "t": t,
            "X1": X1,
            "yT2": yT2,
            "X2": X2,
            "yT3T4": yT3T4,
            "X3": X3,
            "yT5T6T8": yT5T6T8,
            # physics helpers
            "T1": torch.tensor([T3_0]),  # T1 ≈ T3
            "F1": torch.tensor([F1]),
            "I": torch.tensor([I]),
            "Ta": torch.tensor([Ta]),
            "V1": torch.tensor([V1])
        }
