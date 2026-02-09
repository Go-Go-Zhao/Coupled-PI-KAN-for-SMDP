# train.py
import torch
import torch.optim as optim
from physics import solar_field_ode, storage_tank_ode, compute_gradients
from config import NET_CONFIG, DEVICE


def train_coupled_model(model, dataloader):
    optimizer1 = optim.Adam(model.phi1.parameters(), lr=NET_CONFIG['lr'])
    optimizer2 = optim.Adam(model.phi2.parameters(), lr=NET_CONFIG['lr'])
    optimizer3 = optim.Adam(model.phi3.parameters(), lr=NET_CONFIG['lr'])

    epochs = NET_CONFIG['epochs']

    # Sequential Training Strategy (Algorithm 1)
    # Step 2: Train Phi1 (T2)
    print("Step 2: Training Sub-network 1 (T2)...")
    model.phi1.train()
    model.phi2.eval()  # Freeze others
    model.phi3.eval()

    for epoch in range(30):  # Short epochs for demo step
        for batch in dataloader:
            # Unpack batch: t, u(F1, V1), d(I, Ta), y_prev
            # Construct X1 inputs for Phi1
            t = batch['t'].requires_grad_(True).to(DEVICE)
            X1 = batch['X1'].to(DEVICE)  # Should contain [t, T2_prev, F1, I, Ta, T4_prev]
            T2_true = batch['T2'].to(DEVICE)

            optimizer1.zero_grad()

            # Predict T2
            T2_pred = model.forward_phi1(X1)

            # Physics Loss
            dT2_dt = compute_gradients(T2_pred, t)
            # Create input dict for physics
            phy_inputs = {
                'I': X1[:, 3:4], 'Ta': X1[:, 4:5], 'F1': X1[:, 2:3],
                'T1': X1[:, 5:6]  # Approx T4
            }
            loss_phy = solar_field_ode(t, T2_pred, phy_inputs, dT2_dt)

            # Data Loss
            loss_data = torch.nn.functional.mse_loss(T2_pred, T2_true)

            # Adaptive Weighting 
            # Real impl requires calculating grad norms
            w_phy = 0.5

            total_loss = loss_data + w_phy * loss_phy
            total_loss.backward()
            optimizer1.step()

    # Step 3: Train Phi2 (T3, T4)
    print("Step 3: Training Sub-network 2 (T3, T4)...")
    model.phi1.eval()  # Freeze Phi1
    model.phi2.train()

    for epoch in range(30):
        for batch in dataloader:
            t = batch['t'].requires_grad_(True).to(DEVICE)
            X1 = batch['X1'].to(DEVICE)
            X2_partial = batch['X2_base'].to(DEVICE)  # [t, T3_prev, T4_prev, V1, I, Ta]
            T3_true = batch['T3'].to(DEVICE)
            T4_true = batch['T4'].to(DEVICE)

            # Get T2 from frozen Phi1
            with torch.no_grad():
                T2_pred = model.forward_phi1(X1)

            # Construct input for Phi2: Cat X2_partial + T2_pred
            X2 = torch.cat([X2_partial, T2_pred], dim=1)

            optimizer2.zero_grad()
            out_phi2 = model.forward_phi2(X2)
            T3_pred, T4_pred = out_phi2[:, 0:1], out_phi2[:, 1:2]

            # Physics Loss
            dT3_dt = compute_gradients(T3_pred, t)
            dT4_dt = compute_gradients(T4_pred, t)

            phy_inputs = {
                'T2_pred': T2_pred, 'T6_prev': batch['T6_prev'].to(DEVICE),
                'V1': X2_partial[:, 3:4], 'Ta': X2_partial[:, 5:6]
            }
            loss_phy = storage_tank_ode(t, T3_pred, T4_pred, phy_inputs, dT3_dt, dT4_dt)

            loss_data = 0.5 * torch.nn.functional.mse_loss(T3_pred, T3_true) + \
                        0.5 * torch.nn.functional.mse_loss(T4_pred, T4_true)

            total_loss = loss_data + 0.1 * loss_phy
            total_loss.backward()
            optimizer2.step()

    # Step 4: Train Phi3 (similarly...)
    print("Step 4: Training Sub-network 3 (T5, T6, T8)...")
    # ... Implementation similar to above ...

    print("Training Cycle Completed.")

